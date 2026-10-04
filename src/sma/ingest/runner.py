"""Daily ingest runner.

Iterates over configured sources, isolates each one's exceptions, and writes
a per-source row to ingest_log. The CLI in __main__.py wires this up to the
store, the universe, and the configured source set.

Concurrent fetch / serial insert (2026-09-02): measurement over the last ~10
nightly runs showed the ~11-13 min wall-clock (the global writer_lock hold
time) is ~90% finnhub_news + finnhub_fundamentals, which SHARE one rate-limit
bucket (55 req/min) -- a provider-enforced floor that more threads cannot
shrink (they'd just contend for the same 55/min quota; see ratelimit.py).
The real, smaller win is overlapping the OTHER overlay sources' independent
network latency (alpaca_news, newsapi, edgar; each ~10-25s) with that floor
instead of paying it serially on top -- worth roughly 10-15% of total
wall-clock, not the dramatic cut a naive "parallelize everything" story
would suggest.

PRICE_SOURCES (yfinance, alpaca) are excluded from concurrent fetch and keep
running fully serially, exactly as before. This is a deliberate simplicity
choice: yfinance's fetch() reads the store mid-fetch (adj-drift detection,
cross-source quarantine) and does its OWN nested full-history refetch+insert
for resynced tickers -- none of that decomposes cleanly into "network phase"
then "insert phase", and it is by far the highest-stakes code path in this
module (it guards the money-path price data). Rather than split it,
PRICE_SOURCES stay off the parallel path entirely.

Every other source already follows one shape: build a list of rows via
network calls, then a handful of `store.conn.executemany(...)` inserts, with
no store reads. Those sources fetch concurrently in threads, each handed a
`_DeferredInsertStore` stand-in that BUFFERS insert statements instead of
executing them (a real `store.conn` read from a worker thread would touch the
single shared DuckDB connection unsafely, and DuckDB connections are not
safe for concurrent writes either way). Once every thread in the batch
completes, the runner replays each source's buffered statements serially on
the real connection, in fixed `sources`-list order -- deterministic
regardless of which thread actually finished fetching first.
"""

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from loguru import logger

from sma.ingest.sources._finnhub_retry import redact_secrets
from sma.ingest.sources.base import IngestResult, Source
from sma.ingest.store import Store


class _BufferedConn:
    """Stand-in for `store.conn` handed to a source's fetch() while it runs
    concurrently with other sources. Captures INSERT/UPDATE/etc statements
    instead of executing them, so the single shared DuckDB connection is only
    ever touched by the runner's own thread. Only sources in the parallel
    group receive this -- PRICE_SOURCES keep the real store and stay serial.

    Only `executemany` is supported: every non-price source builds a rows
    list and inserts it via one (or a couple of) executemany calls, with no
    store reads. A `.execute()` call (a SELECT, or anything else) means a
    source is trying to read the store mid-fetch -- not safe from a worker
    thread, and not something any current non-price source needs -- so it
    fails loudly instead of silently returning stale/empty data.
    """

    def __init__(self) -> None:
        self.statements: list[tuple[str, list]] = []

    def executemany(self, sql: str, params) -> None:
        self.statements.append((sql, list(params)))

    def execute(self, *args, **kwargs):
        raise RuntimeError(
            "a concurrently-fetched source tried to read the store "
            "(store.conn.execute(...)); only PRICE_SOURCES may read the "
            "store mid-fetch, and they run on the serial path -- if a new "
            "source genuinely needs this, exclude it from parallel fetch"
        )


class _DeferredInsertStore:
    """Stand-in for `store` passed to a parallel-fetch source's fetch()."""

    def __init__(self) -> None:
        self.conn = _BufferedConn()

    def replay(self, real_conn) -> None:
        """Execute every buffered statement, in the order they were issued,
        against the real connection. Called from the runner's own thread
        only, after the fetch phase for this batch has fully completed."""
        for sql, params in self.conn.statements:
            real_conn.executemany(sql, params)


def _log_redacted_crash(what: str, e: BaseException) -> None:
    """A source's exception can quote a URL with its key in the query string
    (Finnhub token=, NewsAPI apiKey=). Log the message and traceback through
    redact_secrets instead of logger.exception, whose traceback would print
    them raw (and, with loguru's diagnose, local variables too)."""
    import traceback

    logger.error(
        "{}: {}\n{}", what, redact_secrets(e),
        redact_secrets("".join(traceback.format_exception(e))),
    )


@dataclass
class IngestRunner:
    store: Store
    sources: list[Source]
    universe: list[str]
    # Per-source final outcome of the last run(): name -> {"status", "rows_inserted"}.
    # Consumed by ingest_run() for the sentinel's `sources` block (which was
    # hardcoded {} until 2026-06-09 — a failed night was undiagnosable from
    # the sentinel alone).
    results: dict = field(default_factory=dict)

    def _fetch_one(self, run_id: int, asof_date: date, src: Source, *, log_start: bool) -> None:
        """Run one source, record its ingest_log row + in-memory result.

        `log_start=False` is the retry path: the (run_id, source) row already
        exists from the first attempt, so only the final outcome is UPDATEd
        (log_run_end), keeping exactly one ingest_log row per (run_id, source).
        """
        if log_start:
            self.store.log_run_start(run_id, source=src.name)
        try:
            result: IngestResult = src.fetch(
                self.universe,
                asof_date,
                self.store,
                run_id,
            )
            status, rows, error = result.status, result.rows_inserted, result.error
            logger.info("source {}: status={} rows={}", src.name, status, rows)
        except Exception as e:
            _log_redacted_crash(f"source {src.name} crashed", e)
            status, rows, error = "error", 0, redact_secrets(e)
        self.store.log_run_end(
            run_id,
            source=src.name,
            rows_inserted=rows,
            status=status,
            error=error,
        )
        self.results[src.name] = {"status": status, "rows_inserted": rows}

    def _fetch_group_concurrent(
        self, run_id: int, asof_date: date, sources: list[Source], *, log_start: bool
    ) -> None:
        """Fetch `sources` concurrently (network I/O only, via a per-source
        `_DeferredInsertStore`), then replay each source's buffered inserts
        serially on the real connection and log its result -- in `sources`
        list order, deterministic regardless of thread completion order.

        Mirrors `_fetch_one`'s failure isolation: a crash inside fetch() OR
        during its later replay is caught per-source and becomes status=
        'error', exactly like the serial path, and never prevents any other
        source in the batch from being replayed/logged.
        """
        if not sources:
            return
        if log_start:
            for src in sources:
                self.store.log_run_start(run_id, source=src.name)

        def _worker(src: Source):
            shim = _DeferredInsertStore()
            try:
                result = src.fetch(self.universe, asof_date, shim, run_id)
                return result, shim, None
            except Exception as e:
                _log_redacted_crash(f"source {src.name} crashed", e)
                return None, shim, e

        with ThreadPoolExecutor(max_workers=len(sources)) as pool:
            futures = {src.name: pool.submit(_worker, src) for src in sources}
            outcomes = {name: fut.result() for name, fut in futures.items()}

        for src in sources:
            result, shim, exc = outcomes[src.name]
            if exc is None:
                try:
                    shim.replay(self.store.conn)
                except Exception as e:
                    _log_redacted_crash(f"source {src.name} insert failed", e)
                    exc = e
            if exc is not None:
                status, rows, error = "error", 0, redact_secrets(exc)
            else:
                status, rows, error = result.status, result.rows_inserted, result.error
                logger.info("source {}: status={} rows={}", src.name, status, rows)
            self.store.log_run_end(
                run_id, source=src.name, rows_inserted=rows, status=status, error=error
            )
            self.results[src.name] = {"status": status, "rows_inserted": rows}

    def _needs_retry(self, name: str) -> bool:
        """Transient-failure policy for the end-of-run retry pass.

        - any source that CRASHED (status='error') is worth one more try
          (DNS/connection blips — the 2026-06-09 night-killer)
        - a PRICE source is additionally retried on rate_limited or ok-with-0-
          rows: without prices there is nothing to trade, and a pause is
          exactly the medicine for a per-minute cap / empty response
        - an overlay source that is merely rate_limited (daily quota) or quiet
          (0 rows) is NOT retried — quota won't lift in 90s and quiet is normal
        """
        from sma.ingest.quality import PRICE_SOURCES

        r = self.results.get(name)
        if r is None:
            return False
        if r["status"] == "error":
            return True
        if name in PRICE_SOURCES:
            return r["status"] != "ok" or r["rows_inserted"] == 0
        return False

    def run(
        self,
        asof_date: date,
        *,
        retry_attempts: int = 2,
        retry_delay_s: float = 90.0,
        sleep_fn=time.sleep,
        deadline: datetime | None = None,
        deadline_margin_s: float = 600.0,
        now_fn=None,
    ) -> int:
        from sma.ingest.quality import PRICE_SOURCES

        _now = now_fn or datetime.utcnow
        # Deadline budget (Codex sweep-4 HIGH, 2026-06-11): the 6/10 run took
        # 2h08m (18:32→20:40) — 10 minutes from blocking predict/decide for
        # the night. Past `deadline - margin`, OVERLAY sources are SKIPPED
        # (status='skipped', non-blocking for quality); PRICE sources always
        # run — they are the point of the job. Price-critical work is also
        # ordered FIRST so slow news nights can never starve tradable data.
        #
        # SCHEDULED SAME-DAY RUNS ONLY (2026-09-17 fix): `deadline` is an
        # ABSOLUTE wall-clock cutoff computed from asof_date's OWN scheduled
        # fire time (sma.schedule.deadline: asof's fire_time_et + offset),
        # not a "budget remaining from now". For a historical asof_date (a
        # repair/backfill run against a past trading day), that cutoff is
        # always already in the past by the time anyone runs the command --
        # comparing it against the real current time (_now()) made every
        # historical overlay fetch look "already past the cutoff" and skip
        # unconditionally. This is exactly the bug
        # scripts/backfill_sept2026_outage_fundamentals.py (54e0a7c) had to
        # route around during the Sept 2026 outage repair, because
        # `python -m sma.ingest run --asof-date <past D> --sources ...`
        # never fetched overlays. Mirrors sma.agents.__main__._deadline_reached
        # / sma.autoresearch.__main__._search_deadline_reached's established
        # "SCHEDULED same-day run only" rule: the budget applies only when
        # asof_date is the SAME calendar day as `_now()` (now_fn is always
        # ET-aware in production). Any other asof_date -- historical (a
        # repair/backfill) or, symmetrically, a future one -- runs fully
        # unbudgeted regardless of what `deadline` was computed to be.
        same_day = asof_date == _now().date()
        cutoff = (
            deadline - timedelta(seconds=deadline_margin_s)
            if deadline is not None and same_day
            else None
        )

        run_id = self.store.allocate_run_id()
        logger.info(
            "starting ingest run_id={} asof={} universe_size={}",
            run_id,
            asof_date,
            len(self.universe),
        )
        self.results = {}
        ordered = sorted(
            self.sources, key=lambda s: 0 if s.name in PRICE_SOURCES else 1
        )
        price_sources = [s for s in ordered if s.name in PRICE_SOURCES]
        overlay_sources = [s for s in ordered if s.name not in PRICE_SOURCES]

        # PRICE_SOURCES always run first, fully serially, against the real
        # store -- unchanged from before concurrent fetch. See module
        # docstring for why they're excluded from the parallel-fetch group.
        for src in price_sources:
            self._fetch_one(run_id, asof_date, src, log_start=True)

        # Overlay sources fetch CONCURRENTLY (see _fetch_group_concurrent).
        # The deadline check is evaluated once, up front for the whole
        # batch, rather than per-source as each source's turn came up under
        # the old sequential loop: concurrency only ever SHORTENS the
        # batch's total wall-clock versus running the same sources serially,
        # so a batch judged not-yet-past-cutoff here can only be more
        # likely -- never less -- to actually finish inside the deadline
        # margin than the old per-source check was.
        runnable_overlay = []
        for src in overlay_sources:
            if cutoff is not None and _now() >= cutoff:
                logger.warning(
                    "deadline budget: skipping overlay source {} ({} past cutoff)",
                    src.name,
                    cutoff,
                )
                self.store.log_run_start(run_id, source=src.name)
                self.store.log_run_end(
                    run_id,
                    source=src.name,
                    rows_inserted=0,
                    status="skipped",
                    error="deadline budget: skipped to protect the trading pipeline",
                )
                self.results[src.name] = {"status": "skipped", "rows_inserted": 0}
                continue
            runnable_overlay.append(src)
        self._fetch_group_concurrent(run_id, asof_date, runnable_overlay, log_start=True)

        # End-of-run retry pass: a transient outage early in the run (the
        # 2026-06-09 DNS window killed both price sources at 18:33 while news
        # sources recovered by 18:50 IN THE SAME RUN) should not cost the
        # whole trading night. The first pass itself takes ~20 min, so by the
        # time we get here a blip has often already cleared.
        for attempt in range(1, retry_attempts + 1):
            retryable = [s for s in ordered if self._needs_retry(s.name)]
            if cutoff is not None:
                # Overlays don't burn retry time past the cutoff; price
                # sources may always retry — prices or bust.
                retryable = [
                    s for s in retryable
                    if s.name in PRICE_SOURCES or _now() < cutoff
                ]
            if not retryable:
                break
            logger.warning(
                "ingest retry pass {}/{} for {} in {}s",
                attempt,
                retry_attempts,
                [s.name for s in retryable],
                retry_delay_s,
            )
            sleep_fn(retry_delay_s)
            retry_price = [s for s in retryable if s.name in PRICE_SOURCES]
            retry_overlay = [s for s in retryable if s.name not in PRICE_SOURCES]
            for src in retry_price:
                self._fetch_one(run_id, asof_date, src, log_start=False)
            self._fetch_group_concurrent(run_id, asof_date, retry_overlay, log_start=False)

        logger.info("ingest run_id={} complete", run_id)
        return run_id


@dataclass
class IngestRunResult:
    run_id: int
    quality_report: Any  # QualityReport; typed as Any to avoid circular at module level
    # Raw no_dead_or_frozen_tickers flags (list[DeadOrFrozenFlag]; Any for the
    # same reason). Pure data -- computing this here is read-only and has no
    # notification side effect; the CLI (sma.ingest.__main__) is what calls
    # notify_new_dead_or_frozen_tickers on it, so constructing/testing an
    # IngestRunResult never fires a real notification.
    dead_frozen_flags: Any = field(default_factory=list)


def run(
    *,
    asof: date,
    db_path: Path | str,
    sources: list[Source],
    universe: list[str],
    deadline: datetime | None = None,
    now_fn=None,
    split_threshold: float | None = None,
) -> IngestRunResult:
    """Standalone ingest entry point: open store, run all sources, run quality
    checks, write the sentinel, close the store.

    Callers MUST hold the writer_lock before calling this function. The sentinel
    is written while the lock is still held (per the ordering contract in
    sentinels.py).

    Returns an IngestRunResult with run_id, quality_report, and
    dead_frozen_flags (no_dead_or_frozen_tickers's raw per-ticker flags; the
    CLI, not this function, decides whether to notify on them).
    """
    from sma.ingest.quality import run_quality_checks
    from sma.schedule import get as get_job
    from sma.sentinels import write_sentinel

    started_at = datetime.utcnow()

    run_id: int | None = None
    quality_report = None
    dead_frozen_flags: list = []
    error_exc: Exception | None = None
    runner: IngestRunner | None = None
    store = Store(path=db_path).connect(read_only=False)
    try:
        runner = IngestRunner(store=store, sources=sources, universe=universe)
        run_id = runner.run(asof_date=asof, deadline=deadline, now_fn=now_fn)
        quality_kwargs = {}
        if split_threshold is not None:
            quality_kwargs["split_threshold"] = split_threshold
        quality_report = run_quality_checks(
            store,
            asof_date=asof,
            universe=universe,
            run_id=run_id,
            **quality_kwargs,
        )
        try:
            from sma.ingest.quality import find_dead_or_frozen_tickers

            dead_frozen_flags = find_dead_or_frozen_tickers(
                store, asof_date=asof, universe=universe
            )
        except Exception as flag_exc:
            # Read-only diagnostic, not part of the quality gate -- must not
            # turn a healthy quality_report into a failed/errored ingest run.
            logger.warning(
                "no_dead_or_frozen_tickers flag lookup failed for asof={}: {}",
                asof,
                flag_exc,
            )
            dead_frozen_flags = []
    except Exception as e:
        # The run or the quality checks crashed. We MUST still write a sentinel
        # (fail-closed) — without one, decide's preflight has nothing to read
        # and the pipeline polls/blocks until timeout instead of failing clean.
        error_exc = e
        logger.exception("ingest run/quality failed for asof={}: {}", asof, e)
    finally:
        store.close()

    finished_at = datetime.utcnow()

    ingest_job = get_job("com.sma.ingest.daily")
    waivers = ingest_job.waivers

    if quality_report is not None:
        checks_map = {c.name: ("PASS" if c.passed else "FAIL") for c in quality_report.checks}
        # Only BLOCKING checks gate decide; overlay checks (news/earnings/theses)
        # can fail without stopping trading.
        blocking = quality_report.blocking_failures(frozenset(waivers))
        waived = [c.name for c in quality_report.checks if not c.passed and c.name in waivers]
        quality_passed = len(blocking) == 0
    else:
        # Crash before a usable report → fail closed so decide blocks cleanly.
        checks_map = {}
        blocking = ["ingest_run_error"]
        waived = []
        quality_passed = False

    quality_block: dict = {
        "passed": quality_passed,
        "checks": checks_map,
        "waived": waived,
        "blocking_failures": blocking,
    }
    if error_exc is not None:
        quality_block["error"] = repr(error_exc)

    write_sentinel(
        label="com.sma.ingest.daily",
        asof=asof,
        payload={
            "label": "com.sma.ingest.daily",
            "asof": asof.isoformat(),
            "run_id": run_id,
            "started_at": started_at.isoformat() + "Z",
            "completed_at": finished_at.isoformat() + "Z",
            "sources": (runner.results if runner is not None else {}),
            "quality": quality_block,
        },
    )

    # Sentinel is now durably written; surface the original failure loudly.
    if error_exc is not None:
        raise error_exc

    return IngestRunResult(
        run_id=run_id, quality_report=quality_report, dead_frozen_flags=dead_frozen_flags
    )
