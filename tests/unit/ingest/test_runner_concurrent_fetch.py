"""Concurrent-fetch / serial-insert restructuring (2026-09-02).

Measurement (see log analysis over 2026-08-13..2026-09-01) showed the nightly
18:30 ingest's ~11-13 min wall-clock is ~90% finnhub_news + finnhub_fundamentals,
which SHARE one rate-limit bucket (55 req/min) -- a provider-enforced floor
concurrency cannot shrink. The real, smaller win is overlapping the other
overlay sources' independent network latency with that floor instead of
paying it serially on top.

Design: PRICE_SOURCES (yfinance, alpaca) keep running fully serially, exactly
as before -- this sidesteps yfinance's drift-detection/quarantine logic, which
reads the store mid-fetch and does its OWN nested insert for resynced tickers,
so it does not decompose cleanly into a fetch phase and an insert phase.
Everything NOT in PRICE_SOURCES fetches concurrently in threads (network I/O
only, via a store stand-in that buffers INSERT/UPDATE statements instead of
executing them), then the runner replays each source's buffered statements
serially on the real connection, in fixed `sources`-list order -- DuckDB
connections are not safe for concurrent writes, and this keeps insert order
deterministic regardless of which thread finishes first.
"""

import threading
import time
from datetime import date

from sma.ingest.ratelimit import TokenBucket
from sma.ingest.runner import IngestRunner
from sma.ingest.sources.base import IngestResult
from sma.ingest.store import Store


def _store(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    return Store(path=tmp_path / "t.duckdb").connect(read_only=False)


def _log_rows(store, run_id):
    return dict(
        (r[0], (r[1], r[2]))
        for r in store.conn.execute(
            "SELECT source, status, rows_inserted FROM ingest_log WHERE run_id = ?",
            [run_id],
        ).fetchall()
    )


class _NewsSource:
    """Mimics finnhub_news/alpaca_news/newsapi/edgar: builds rows, then does
    exactly one `store.conn.executemany` INSERT before returning, and reads
    the store never. `delay_s` simulates network latency for ordering tests.
    """

    def __init__(self, name, tickers, delay_s=0.0, run_ids_seen=None):
        self.name = name
        self._tickers = tickers
        self._delay_s = delay_s
        self.calls = 0
        self._run_ids_seen = run_ids_seen if run_ids_seen is not None else []

    def fetch(self, universe, asof, store, run_id):
        self.calls += 1
        if self._delay_s:
            time.sleep(self._delay_s)
        rows = [
            (t, f"http://x/{self.name}/{t}", f"{self.name}:{t}", run_id)
            for t in self._tickers
        ]
        store.conn.executemany(
            "INSERT OR REPLACE INTO _test_news_rows (ticker, url, tag, run_id) "
            "VALUES (?, ?, ?, ?)",
            rows,
        )
        self._run_ids_seen.append(run_id)
        return IngestResult(source=self.name, rows_inserted=len(rows), status="ok", error=None)


class _CrashingSource:
    name = "crashy"

    def fetch(self, universe, asof, store, run_id):
        raise RuntimeError("network exploded")


class _ReaderSource:
    """A misbehaving overlay source that tries to READ from the store mid-fetch
    -- only PRICE_SOURCES are supposed to do this. Must fail loudly rather than
    silently seeing garbage, since a real store read from a parallel-fetch
    worker thread would touch the single shared DuckDB connection unsafely."""

    name = "edgar"  # a real overlay-source name so it lands in the parallel group

    def fetch(self, universe, asof, store, run_id):
        store.conn.execute("SELECT 1")
        return IngestResult(source=self.name, rows_inserted=0, status="ok", error=None)


def _ensure_test_table(store):
    store.conn.execute(
        "CREATE TABLE IF NOT EXISTS _test_news_rows "
        "(ticker VARCHAR, url VARCHAR, tag VARCHAR, run_id BIGINT, "
        "PRIMARY KEY (ticker, tag))"
    )


# ---------------------------------------------------------------------------
# Parity: same fake source logic through the serial (PRICE_SOURCES) path vs
# the parallel (overlay) path must produce identical DB rows and statuses.
# ---------------------------------------------------------------------------


def test_parallel_and_serial_paths_insert_identical_rows(tmp_path):
    tickers = ["AAPL", "MSFT", "GOOG"]

    store_serial = _store(tmp_path / "serial")
    _ensure_test_table(store_serial)
    serial_src = _NewsSource("yfinance", tickers)  # PRICE_SOURCES name -> serial path
    runner_serial = IngestRunner(store=store_serial, sources=[serial_src], universe=tickers)
    run_id_serial = runner_serial.run(asof_date=date(2026, 9, 2), sleep_fn=lambda s: None)

    store_parallel = _store(tmp_path / "parallel")
    _ensure_test_table(store_parallel)
    parallel_src = _NewsSource("edgar", tickers)  # overlay name -> parallel path
    runner_parallel = IngestRunner(store=store_parallel, sources=[parallel_src], universe=tickers)
    run_id_parallel = runner_parallel.run(asof_date=date(2026, 9, 2), sleep_fn=lambda s: None)

    rows_serial = sorted(
        store_serial.conn.execute("SELECT ticker, url FROM _test_news_rows").fetchall()
    )
    rows_parallel = sorted(
        store_parallel.conn.execute("SELECT ticker, url FROM _test_news_rows").fetchall()
    )
    assert [(t, u.replace("yfinance", "SRC")) for t, u in rows_serial] == [
        (t, u.replace("edgar", "SRC")) for t, u in rows_parallel
    ]
    assert runner_serial.results[serial_src.name] == {"status": "ok", "rows_inserted": 3}
    assert runner_parallel.results[parallel_src.name] == {"status": "ok", "rows_inserted": 3}
    assert _log_rows(store_serial, run_id_serial)[serial_src.name] == ("ok", 3)
    assert _log_rows(store_parallel, run_id_parallel)[parallel_src.name] == ("ok", 3)
    store_serial.close()
    store_parallel.close()


def test_multiple_overlay_sources_all_land_via_parallel_path(tmp_path):
    tickers = ["AAPL", "MSFT"]
    store = _store(tmp_path)
    _ensure_test_table(store)
    a = _NewsSource("finnhub_news", tickers)
    b = _NewsSource("alpaca_news", tickers)
    c = _NewsSource("newsapi", tickers)
    runner = IngestRunner(store=store, sources=[a, b, c], universe=tickers)
    run_id = runner.run(asof_date=date(2026, 9, 2), sleep_fn=lambda s: None)

    rows = store.conn.execute(
        "SELECT tag FROM _test_news_rows WHERE run_id = ? ORDER BY tag", [run_id]
    ).fetchall()
    tags = {r[0] for r in rows}
    assert tags == {
        "finnhub_news:AAPL", "finnhub_news:MSFT",
        "alpaca_news:AAPL", "alpaca_news:MSFT",
        "newsapi:AAPL", "newsapi:MSFT",
    }
    logs = _log_rows(store, run_id)
    assert logs["finnhub_news"] == ("ok", 2)
    assert logs["alpaca_news"] == ("ok", 2)
    assert logs["newsapi"] == ("ok", 2)
    store.close()


# ---------------------------------------------------------------------------
# Failure isolation: one overlay source crashing must not take down the
# others fetched concurrently alongside it.
# ---------------------------------------------------------------------------


def test_one_crashing_overlay_source_does_not_kill_others(tmp_path):
    tickers = ["AAPL"]
    store = _store(tmp_path)
    _ensure_test_table(store)
    good_a = _NewsSource("finnhub_news", tickers)
    good_b = _NewsSource("newsapi", tickers)
    bad = _CrashingSource()
    runner = IngestRunner(store=store, sources=[good_a, bad, good_b], universe=tickers)
    run_id = runner.run(asof_date=date(2026, 9, 2), sleep_fn=lambda s: None)

    assert runner.results["finnhub_news"] == {"status": "ok", "rows_inserted": 1}
    assert runner.results["newsapi"] == {"status": "ok", "rows_inserted": 1}
    assert runner.results["crashy"]["status"] == "error"
    assert runner.results["crashy"]["rows_inserted"] == 0

    logs = _log_rows(store, run_id)
    assert logs["finnhub_news"] == ("ok", 1)
    assert logs["newsapi"] == ("ok", 1)
    assert logs["crashy"][0] == "error"
    store.close()


# ---------------------------------------------------------------------------
# The store handed to a parallel-fetch source must refuse reads loudly (only
# PRICE_SOURCES are allowed to read the store mid-fetch, and they stay on the
# serial path entirely).
# ---------------------------------------------------------------------------


def test_overlay_source_reading_the_store_fails_loudly_not_silently(tmp_path):
    tickers = ["AAPL"]
    store = _store(tmp_path)
    _ensure_test_table(store)
    reader = _ReaderSource()
    runner = IngestRunner(store=store, sources=[reader], universe=tickers)
    runner.run(asof_date=date(2026, 9, 2), sleep_fn=lambda s: None)

    # The bad read must have been caught and isolated as this source's own
    # failure (status=error), not silently swallowed and not a runner crash.
    assert runner.results["edgar"]["status"] == "error"
    store.close()


# ---------------------------------------------------------------------------
# Deterministic insert/log order: replay order follows the `sources` list,
# not thread-completion order, even when an earlier-listed source is slower.
# ---------------------------------------------------------------------------


def test_replay_order_is_list_order_not_completion_order(tmp_path):
    tickers = ["AAPL"]
    store = _store(tmp_path)
    _ensure_test_table(store)
    slow_first = _NewsSource("finnhub_news", tickers, delay_s=0.25)
    fast_second = _NewsSource("newsapi", tickers, delay_s=0.0)
    runner = IngestRunner(store=store, sources=[slow_first, fast_second], universe=tickers)
    runner.run(asof_date=date(2026, 9, 2), sleep_fn=lambda s: None)

    # Both must complete and land regardless of who finished fetching first --
    # the replay loop iterates the original list, not a completion queue.
    assert list(runner.results.keys()) == ["finnhub_news", "newsapi"]
    store.close()


def test_price_sources_still_run_fully_serially_before_overlay_group(tmp_path):
    """PRICE_SOURCES (yfinance, alpaca) are excluded from the parallel group
    entirely -- unchanged serial behavior, verified via call ordering."""
    calls = []

    class _Recorder(_NewsSource):
        def fetch(self, universe, asof, store, run_id):
            calls.append(self.name)
            return super().fetch(universe, asof, store, run_id)

    tickers = ["AAPL"]
    store = _store(tmp_path)
    _ensure_test_table(store)
    price = _Recorder("alpaca", tickers)
    overlay = _Recorder("edgar", tickers)
    runner = IngestRunner(store=store, sources=[overlay, price], universe=tickers)
    runner.run(asof_date=date(2026, 9, 2), sleep_fn=lambda s: None)
    assert calls == ["alpaca", "edgar"], "price sources must still run first, fully serially"
    store.close()


# ---------------------------------------------------------------------------
# Rate-limiter sharing stays intact under real concurrent fetch: two overlay
# sources sharing ONE TokenBucket (mirroring finnhub_news + finnhub_fundamentals
# sharing the production finnhub_limiter) must never jointly over-issue tokens.
# ---------------------------------------------------------------------------


class _RateLimitedSource:
    def __init__(self, name, tickers, bucket: TokenBucket, grants: list):
        self.name = name
        self._tickers = tickers
        self._bucket = bucket
        self._grants = grants
        self._lock = threading.Lock()

    def fetch(self, universe, asof, store, run_id):
        rows = []
        for t in self._tickers:
            granted = self._bucket.try_acquire()
            with self._lock:
                self._grants.append(granted)
            if granted:
                rows.append((t, f"http://x/{self.name}/{t}", f"{self.name}:{t}", run_id))
        if rows:
            store.conn.executemany(
                "INSERT OR REPLACE INTO _test_news_rows (ticker, url, tag, run_id) "
                "VALUES (?, ?, ?, ?)",
                rows,
            )
        return IngestResult(source=self.name, rows_inserted=len(rows), status="ok", error=None)


def test_shared_rate_limiter_not_over_issued_across_concurrently_fetched_sources(tmp_path):
    tickers = [f"T{i}" for i in range(20)]
    store = _store(tmp_path)
    _ensure_test_table(store)
    bucket = TokenBucket(capacity=10, refill_rate_per_sec=0.0)  # exactly 10 tokens, ever
    grants: list = []
    a = _RateLimitedSource("finnhub_news", tickers, bucket, grants)
    b = _RateLimitedSource("finnhub_fundamentals", tickers, bucket, grants)
    runner = IngestRunner(store=store, sources=[a, b], universe=tickers)
    runner.run(asof_date=date(2026, 9, 2), sleep_fn=lambda s: None)

    total_granted = sum(1 for g in grants if g)
    assert total_granted == 10, (
        f"shared bucket (capacity=10) granted {total_granted} tokens across two "
        "concurrently-fetched sources -- rate-limiter sharing broke under concurrency"
    )
    store.close()
