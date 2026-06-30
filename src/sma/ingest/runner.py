"""Daily ingest runner.

Iterates over configured sources, isolates each one's exceptions, and writes
a per-source row to ingest_log. The CLI in __main__.py wires this up to the
store, the universe, and the configured source set.
"""

import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from loguru import logger

from sma.ingest.sources.base import IngestResult, Source
from sma.ingest.store import Store


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
            logger.exception("source {} crashed: {}", src.name, e)
            status, rows, error = "error", 0, str(e)
        self.store.log_run_end(
            run_id,
            source=src.name,
            rows_inserted=rows,
            status=status,
            error=error,
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
        cutoff = (
            deadline - timedelta(seconds=deadline_margin_s)
            if deadline is not None
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
        for src in ordered:
            if (
                cutoff is not None
                and src.name not in PRICE_SOURCES
                and _now() >= cutoff
            ):
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
            self._fetch_one(run_id, asof_date, src, log_start=True)

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
            for src in retryable:
                self._fetch_one(run_id, asof_date, src, log_start=False)

        logger.info("ingest run_id={} complete", run_id)
        return run_id


@dataclass
class IngestRunResult:
    run_id: int
    quality_report: Any  # QualityReport; typed as Any to avoid circular at module level


def run(
    *,
    asof: date,
    db_path: Path | str,
    sources: list[Source],
    universe: list[str],
    deadline: datetime | None = None,
    now_fn=None,
) -> IngestRunResult:
    """Standalone ingest entry point: open store, run all sources, run quality
    checks, write the sentinel, close the store.

    Callers MUST hold the writer_lock before calling this function. The sentinel
    is written while the lock is still held (per the ordering contract in
    sentinels.py).

    Returns an IngestRunResult with run_id and quality_report.
    """
    from sma.ingest.quality import run_quality_checks
    from sma.schedule import get as get_job
    from sma.sentinels import write_sentinel

    started_at = datetime.utcnow()

    run_id: int | None = None
    quality_report = None
    error_exc: Exception | None = None
    runner: IngestRunner | None = None
    store = Store(path=db_path).connect(read_only=False)
    try:
        runner = IngestRunner(store=store, sources=sources, universe=universe)
        run_id = runner.run(asof_date=asof, deadline=deadline, now_fn=now_fn)
        quality_report = run_quality_checks(
            store,
            asof_date=asof,
            universe=universe,
            run_id=run_id,
        )
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

    return IngestRunResult(run_id=run_id, quality_report=quality_report)
