"""End-of-run retry pass for transient source failures.

2026-06-09: a ~20-min DNS outage at 18:33 killed BOTH price sources on their
single attempt; news sources that happened to run after 18:50 succeeded in the
SAME run. Quality then (correctly) failed the night and decide refused to
trade. One retry pass at end-of-run would have saved the whole trading day.
"""

from datetime import date

from sma.ingest.runner import IngestRunner
from sma.ingest.sources.base import IngestResult
from sma.ingest.store import Store


class _FlakySource:
    """Fails (or under-delivers) for the first `bad_attempts` fetches."""

    def __init__(self, name, bad_attempts=1, *, bad_mode="crash", ok_rows=5):
        self.name = name
        self.calls = 0
        self._bad_attempts = bad_attempts
        self._bad_mode = bad_mode
        self._ok_rows = ok_rows

    def fetch(self, universe, asof, store, run_id):
        self.calls += 1
        if self.calls <= self._bad_attempts:
            if self._bad_mode == "crash":
                raise ConnectionError("DNS down")
            return IngestResult(
                source=self.name, rows_inserted=0, status=self._bad_mode, error=None
            )
        return IngestResult(
            source=self.name, rows_inserted=self._ok_rows, status="ok", error=None
        )


def _store(tmp_path):
    return Store(path=tmp_path / "t.duckdb").connect(read_only=False)


def _final_log_rows(store, run_id):
    return store.conn.execute(
        "SELECT source, status, rows_inserted FROM ingest_log WHERE run_id = ?",
        [run_id],
    ).fetchall()


def test_errored_source_is_retried_and_recovers(tmp_path):
    src = _FlakySource("yfinance", bad_attempts=1)
    store = _store(tmp_path)
    sleeps = []
    runner = IngestRunner(store=store, sources=[src], universe=["AAPL"])
    run_id = runner.run(
        asof_date=date(2026, 6, 9), retry_delay_s=90.0, sleep_fn=sleeps.append
    )
    assert src.calls == 2
    assert sleeps == [90.0]
    assert runner.results["yfinance"] == {"status": "ok", "rows_inserted": 5}
    # exactly ONE ingest_log row per (run_id, source), reflecting the final attempt
    assert _final_log_rows(store, run_id) == [("yfinance", "ok", 5)]
    store.close()


def test_zero_row_price_source_is_retried_but_zero_row_overlay_is_not(tmp_path):
    price = _FlakySource("alpaca", bad_attempts=1, bad_mode="ok")  # ok but 0 rows
    overlay = _FlakySource("edgar", bad_attempts=99, bad_mode="ok")  # ok but 0 rows
    store = _store(tmp_path)
    runner = IngestRunner(store=store, sources=[price, overlay], universe=["AAPL"])
    runner.run(asof_date=date(2026, 6, 9), sleep_fn=lambda s: None)
    assert price.calls == 2  # retried: a 0-row price day is no tradable data
    assert overlay.calls == 1  # 0-row edgar is a normal quiet day, not retryable
    assert runner.results["alpaca"]["status"] == "ok"
    assert runner.results["alpaca"]["rows_inserted"] == 5
    store.close()


def test_rate_limited_price_source_is_retried(tmp_path):
    src = _FlakySource("yfinance", bad_attempts=1, bad_mode="rate_limited")
    store = _store(tmp_path)
    runner = IngestRunner(store=store, sources=[src], universe=["AAPL"])
    runner.run(asof_date=date(2026, 6, 9), sleep_fn=lambda s: None)
    assert src.calls == 2  # a pause is exactly the medicine for a per-minute cap
    store.close()


def test_rate_limited_overlay_source_is_not_retried(tmp_path):
    src = _FlakySource("newsapi", bad_attempts=1, bad_mode="rate_limited")
    store = _store(tmp_path)
    runner = IngestRunner(store=store, sources=[src], universe=["AAPL"])
    runner.run(asof_date=date(2026, 6, 9), sleep_fn=lambda s: None)
    assert src.calls == 1  # daily news quota won't lift in 90s; don't burn time
    store.close()


def test_retries_exhausted_keeps_error_status(tmp_path):
    src = _FlakySource("yfinance", bad_attempts=99)
    store = _store(tmp_path)
    sleeps = []
    runner = IngestRunner(store=store, sources=[src], universe=["AAPL"])
    run_id = runner.run(
        asof_date=date(2026, 6, 9), retry_attempts=2, sleep_fn=sleeps.append
    )
    assert src.calls == 3  # initial + 2 retries
    assert len(sleeps) == 2
    assert runner.results["yfinance"]["status"] == "error"
    assert _final_log_rows(store, run_id) == [("yfinance", "error", 0)]
    store.close()


def test_no_sleep_when_everything_succeeds(tmp_path):
    src = _FlakySource("yfinance", bad_attempts=0)
    store = _store(tmp_path)
    sleeps = []
    runner = IngestRunner(store=store, sources=[src], universe=["AAPL"])
    runner.run(asof_date=date(2026, 6, 9), sleep_fn=sleeps.append)
    assert src.calls == 1
    assert sleeps == []
    store.close()


class _SlowClock:
    """Fake clock: starts at t0; every now() call returns the same time until
    advanced manually by tests via .t."""

    def __init__(self, t0):
        self.t = t0

    def now(self):
        return self.t


def test_overlay_sources_skipped_past_deadline_but_prices_always_run(tmp_path):
    """Codex sweep-4 HIGH: ingest had no deadline budget — 6/10's run took
    2h08m (18:32→20:40) and finished 10 minutes from costing the trading
    night. Past the cutoff, OVERLAY sources are skipped (status='skipped');
    PRICE sources always run (they are the point of the job)."""
    from datetime import datetime

    price = _FlakySource("yfinance", bad_attempts=0)
    overlay = _FlakySource("edgar", bad_attempts=0)
    store = _store(tmp_path)
    clock = _SlowClock(datetime(2026, 6, 10, 20, 25))  # past cutoff
    runner = IngestRunner(store=store, sources=[price, overlay], universe=["AAPL"])
    run_id = runner.run(
        asof_date=date(2026, 6, 10),
        deadline=datetime(2026, 6, 10, 20, 30),
        deadline_margin_s=600,
        now_fn=clock.now,
        sleep_fn=lambda s: None,
    )
    assert price.calls == 1, "price sources must run regardless of deadline"
    assert overlay.calls == 0, "overlay source must be skipped past the cutoff"
    rows = dict(
        (r[0], r[1]) for r in _final_log_rows(store, run_id)
    )
    assert rows["yfinance"] == "ok"
    assert rows["edgar"] == "skipped"
    assert runner.results["edgar"] == {"status": "skipped", "rows_inserted": 0}
    store.close()


def test_all_sources_run_when_before_cutoff(tmp_path):
    from datetime import datetime

    price = _FlakySource("yfinance", bad_attempts=0)
    overlay = _FlakySource("edgar", bad_attempts=0)
    store = _store(tmp_path)
    clock = _SlowClock(datetime(2026, 6, 10, 18, 35))
    runner = IngestRunner(store=store, sources=[price, overlay], universe=["AAPL"])
    runner.run(
        asof_date=date(2026, 6, 10),
        deadline=datetime(2026, 6, 10, 20, 30),
        now_fn=clock.now,
        sleep_fn=lambda s: None,
    )
    assert price.calls == 1 and overlay.calls == 1
    store.close()


def test_overlay_retry_suppressed_past_deadline_price_retry_allowed(tmp_path):
    from datetime import datetime

    price = _FlakySource("yfinance", bad_attempts=1)   # error once
    overlay = _FlakySource("edgar", bad_attempts=99)   # always errors
    store = _store(tmp_path)
    clock = _SlowClock(datetime(2026, 6, 10, 20, 25))  # past cutoff already
    runner = IngestRunner(store=store, sources=[price, overlay], universe=["AAPL"])
    runner.run(
        asof_date=date(2026, 6, 10),
        deadline=datetime(2026, 6, 10, 20, 30),
        deadline_margin_s=600,
        now_fn=clock.now,
        sleep_fn=lambda s: None,
    )
    assert price.calls == 2, "price retry must happen even past deadline (prices or bust)"
    assert overlay.calls == 0, "overlay was skipped pre-deadline and must not retry either"
    store.close()


def test_price_sources_ordered_first(tmp_path):
    """Price-critical work happens before slow overlay fetches so a slow news
    night can never starve the tradable data."""
    calls = []

    class _Recorder(_FlakySource):
        def fetch(self, universe, asof, store, run_id):
            calls.append(self.name)
            return super().fetch(universe, asof, store, run_id)

    overlay = _Recorder("edgar", bad_attempts=0)
    price = _Recorder("alpaca", bad_attempts=0)
    store = _store(tmp_path)
    runner = IngestRunner(store=store, sources=[overlay, price], universe=["AAPL"])
    runner.run(asof_date=date(2026, 6, 10), sleep_fn=lambda s: None)
    assert calls == ["alpaca", "edgar"], "price sources must run first"
    store.close()


def test_historical_asof_ignores_stale_deadline_and_runs_overlays(tmp_path):
    """2026-09-17 bug (found during the Sept 2026 outage repair): `deadline` is
    an ABSOLUTE wall-clock cutoff computed from asof_date's OWN scheduled fire
    time (sma.schedule.deadline), not a "budget remaining from now". For a
    historical asof_date (a repair/backfill run against a past trading day),
    that cutoff is always already in the past by the time anyone runs the
    command -- comparing it against the real current time made
    `python -m sma.ingest run --asof-date <past D> --sources edgar` (etc.)
    skip EVERY overlay source unconditionally, exactly what
    scripts/backfill_sept2026_outage_fundamentals.py (54e0a7c) had to route
    around. The deadline budget must only apply to a SCHEDULED SAME-DAY run:
    asof_date == now_fn()'s date. Mirrors sma.agents.__main__._deadline_reached
    / sma.autoresearch.__main__._search_deadline_reached's established rule.
    """
    from datetime import datetime

    price = _FlakySource("yfinance", bad_attempts=0)
    overlay = _FlakySource("edgar", bad_attempts=0)
    store = _store(tmp_path)
    # `now` is 2026-09-17 (today); asof_date is a historical trading day
    # whose own scheduled cutoff (2026-06-10 20:30) is naturally long past by
    # 2026-09-17 -- exactly the shape a real repair/backfill run has.
    clock = _SlowClock(datetime(2026, 9, 17, 12, 0))
    runner = IngestRunner(store=store, sources=[price, overlay], universe=["AAPL"])
    run_id = runner.run(
        asof_date=date(2026, 6, 10),
        deadline=datetime(2026, 6, 10, 20, 30),
        deadline_margin_s=600,
        now_fn=clock.now,
        sleep_fn=lambda s: None,
    )
    assert price.calls == 1
    assert overlay.calls == 1, "a historical asof must run overlays despite a stale deadline"
    rows = dict((r[0], r[1]) for r in _final_log_rows(store, run_id))
    assert rows["yfinance"] == "ok"
    assert rows["edgar"] == "ok"
    store.close()


def test_future_asof_also_unbudgeted(tmp_path):
    """Symmetric with the historical case: an asof_date that is not TODAY
    (even a future one) is not a scheduled same-day run either, so it must
    not be budgeted -- mirrors _search_deadline_reached's
    test_not_reached_for_future_asof."""
    from datetime import datetime

    overlay = _FlakySource("edgar", bad_attempts=0)
    store = _store(tmp_path)
    clock = _SlowClock(datetime(2026, 6, 10, 12, 0))
    runner = IngestRunner(store=store, sources=[overlay], universe=["AAPL"])
    runner.run(
        asof_date=date(2026, 6, 17),  # a week ahead of "now"
        deadline=datetime(2026, 6, 17, 18, 40),  # already past relative to "now" is irrelevant
        deadline_margin_s=600,
        now_fn=clock.now,
        sleep_fn=lambda s: None,
    )
    assert overlay.calls == 1


def test_same_day_asof_still_enforces_the_deadline_budget(tmp_path):
    """Regression guard for the fix above: a genuinely SAME-DAY scheduled run
    (asof_date == now_fn()'s date) must keep skipping overlays past the
    cutoff exactly as before -- the historical-asof exemption must not
    accidentally swallow the real budget."""
    from datetime import datetime

    price = _FlakySource("yfinance", bad_attempts=0)
    overlay = _FlakySource("edgar", bad_attempts=0)
    store = _store(tmp_path)
    clock = _SlowClock(datetime(2026, 6, 10, 20, 25))  # same day, past cutoff
    runner = IngestRunner(store=store, sources=[price, overlay], universe=["AAPL"])
    runner.run(
        asof_date=date(2026, 6, 10),
        deadline=datetime(2026, 6, 10, 20, 30),
        deadline_margin_s=600,
        now_fn=clock.now,
        sleep_fn=lambda s: None,
    )
    assert price.calls == 1
    assert overlay.calls == 0, "same-day scheduled run must still honor the budget"
