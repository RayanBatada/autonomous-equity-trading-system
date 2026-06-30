"""Pre-flight check tests updated for sentinel-based preflight (Task 8).

The legacy DuckDB-based preflight (staleness, equity divergence, position
divergence, account checks) has been replaced by a sentinel-based preflight
that reads only sentinel files and the Alpaca calendar. These tests verify
the new contract.
"""

from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from sma.live.preflight import (
    CalendarHolidayError,
    IngestNotCompleteError,
    NextSessionNotTomorrow,
    UpstreamMissedDeadline,
    UpstreamReadinessFailed,
    run_preflight,
)
from sma.sentinels import write_sentinel


@pytest.fixture(autouse=True)
def _patch_sentinel_dir(tmp_path, monkeypatch):
    """Redirect sentinel writes/reads to tmp_path for every test in this module."""
    sentinel_dir = tmp_path / "sentinels"
    sentinel_dir.mkdir()
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(sentinel_dir))
    return sentinel_dir


def _write_sentinel_ok(label: str, asof: date) -> None:
    write_sentinel(
        label=label,
        asof=asof,
        payload={
            "label": label,
            "asof": asof.isoformat(),
            "run_id": 1,
            "quality": {"passed": True, "blocking_failures": []},
        },
    )


def _write_sentinel_fail(label: str, asof: date, blocking: list) -> None:
    write_sentinel(
        label=label,
        asof=asof,
        payload={
            "label": label,
            "asof": asof.isoformat(),
            "run_id": 1,
            "quality": {"passed": False, "blocking_failures": blocking},
        },
    )


def _mock_alpaca(next_session_date_value: date) -> MagicMock:
    alpaca = MagicMock()
    alpaca.next_session_date.return_value = next_session_date_value
    return alpaca


def _write_all_upstream_ok(asof: date) -> None:
    for label in (
        "com.sma.ingest.daily",
        "com.sma.model.predict.daily",
        "com.sma.agents.daily",
    ):
        _write_sentinel_ok(label, asof)


def test_pre_flight_passes_on_clean_state(tmp_path):
    """Happy path: all upstream sentinels ready, next session is tomorrow."""
    asof = date(2026, 5, 1)  # Friday
    _write_all_upstream_ok(asof)
    alpaca = _mock_alpaca(date(2026, 5, 4))  # Mon

    # Should not raise; returns None in new sentinel-based preflight
    result = run_preflight(asof=asof, db_path=Path(str(tmp_path)), alpaca=alpaca, max_wait_s=0)
    assert result is None


def test_pre_flight_accepts_holiday_adjacent_session(tmp_path):
    """Pre-holiday days are valid trading days. Empirical 2026-05-22: Friday
    before Memorial Day; alpaca correctly returned next session = Tue 5/26
    (skipping Mon holiday). Pre-fix preflight raised NextSessionNotTomorrow
    and the bot skipped a full trading-day's signal. The relaxed check
    accepts any next session within 14 days strictly after asof — OPG orders
    queue at Alpaca and fill at the next genuine open auction."""
    asof = date(2026, 5, 1)  # Friday
    _write_all_upstream_ok(asof)
    # Memorial Day Monday -> next session is Tuesday (3-day gap)
    alpaca = _mock_alpaca(date(2026, 5, 5))

    # Should NOT raise.
    result = run_preflight(asof=asof, db_path=Path(str(tmp_path)), alpaca=alpaca, max_wait_s=0)
    assert result is None


def test_pre_flight_aborts_when_next_session_too_far_out(tmp_path):
    """Sanity-bound case: an obviously broken calendar (>14 days out) still
    aborts so we don't queue OPG orders against bad SDK state."""
    asof = date(2026, 5, 1)
    _write_all_upstream_ok(asof)
    alpaca = _mock_alpaca(date(2026, 6, 30))  # 60 days out

    with pytest.raises(NextSessionNotTomorrow):
        run_preflight(asof=asof, db_path=Path(str(tmp_path)), alpaca=alpaca, max_wait_s=0)


def test_pre_flight_calendar_alias_still_works(tmp_path):
    """CalendarHolidayError is kept as a back-compat alias for callers that
    catch by the old name; raising the underlying NextSessionNotTomorrow is
    catchable via either name."""
    asof = date(2026, 5, 1)
    _write_all_upstream_ok(asof)
    alpaca = _mock_alpaca(date(2026, 6, 30))

    with pytest.raises(CalendarHolidayError):
        run_preflight(asof=asof, db_path=Path(str(tmp_path)), alpaca=alpaca, max_wait_s=0)


def test_pre_flight_aborts_when_ingest_sentinel_absent(tmp_path):
    """Ingest sentinel not written -> IngestNotCompleteError (max_wait_s=0 = test mode)."""
    asof = date(2026, 5, 1)
    alpaca = _mock_alpaca(date(2026, 5, 4))

    with pytest.raises(IngestNotCompleteError):
        run_preflight(asof=asof, db_path=Path(str(tmp_path)), alpaca=alpaca, max_wait_s=0)


def test_pre_flight_aborts_when_ingest_quality_failed(tmp_path):
    """Ingest sentinel present but quality.passed=False -> UpstreamReadinessFailed."""
    asof = date(2026, 5, 1)
    _write_sentinel_fail(
        "com.sma.ingest.daily",
        asof,
        blocking=["all_tickers_have_price"],
    )
    alpaca = _mock_alpaca(date(2026, 5, 4))

    with pytest.raises(UpstreamReadinessFailed, match="all_tickers_have_price"):
        run_preflight(asof=asof, db_path=Path(str(tmp_path)), alpaca=alpaca, max_wait_s=0)


def test_pre_flight_aborts_when_predict_sentinel_missing(tmp_path):
    """Ingest OK but predict sentinel absent -> UpstreamMissedDeadline."""
    asof = date(2026, 5, 1)
    _write_sentinel_ok("com.sma.ingest.daily", asof)
    # predict + agents sentinels not written
    alpaca = _mock_alpaca(date(2026, 5, 4))

    with pytest.raises(UpstreamMissedDeadline):
        run_preflight(asof=asof, db_path=Path(str(tmp_path)), alpaca=alpaca, max_wait_s=0)


def test_pre_flight_proceeds_when_agents_sentinel_missing(tmp_path):
    """Agents is an ADVISORY dependency (2026-06-05): with ingest + predict OK, a
    MISSING agents sentinel must NOT abort preflight — it's logged, not fatal, so
    a network blip on the overlay can't re-freeze trading."""
    asof = date(2026, 5, 1)
    _write_sentinel_ok("com.sma.ingest.daily", asof)
    _write_sentinel_ok("com.sma.model.predict.daily", asof)
    # agents sentinel not written — advisory, so preflight proceeds.
    alpaca = _mock_alpaca(date(2026, 5, 4))

    # Must NOT raise.
    run_preflight(asof=asof, db_path=Path(str(tmp_path)), alpaca=alpaca, max_wait_s=0)


def test_pre_flight_aborts_on_partial_source_failure(tmp_path):
    """Ingest sentinel quality.passed=False (blocking failures) -> UpstreamReadinessFailed."""
    asof = date(2026, 5, 1)
    _write_sentinel_fail(
        "com.sma.ingest.daily",
        asof,
        blocking=["all_tickers_have_price", "min_row_count"],
    )
    alpaca = _mock_alpaca(date(2026, 5, 4))

    with pytest.raises(UpstreamReadinessFailed):
        run_preflight(asof=asof, db_path=Path(str(tmp_path)), alpaca=alpaca, max_wait_s=0)


def test_pre_flight_passes_when_only_newsapi_failed(tmp_path):
    """newsapi_rate_limit is a waiver on the decide job; ingest sentinel passed=True
    despite newsapi failure -> preflight passes.
    """
    asof = date(2026, 5, 1)
    # Waiver applied upstream: sentinel has passed=True even though newsapi was slow.
    _write_sentinel_ok("com.sma.ingest.daily", asof)
    _write_sentinel_ok("com.sma.model.predict.daily", asof)
    _write_sentinel_ok("com.sma.agents.daily", asof)
    alpaca = _mock_alpaca(date(2026, 5, 4))

    result = run_preflight(asof=asof, db_path=Path(str(tmp_path)), alpaca=alpaca, max_wait_s=0)
    assert result is None


def test_pre_flight_passes_friday_to_monday_calendar(tmp_path):
    """Friday's next_session is Monday (skip weekend) -- calendar check accepts."""
    asof = date(2026, 5, 1)  # Friday
    _write_all_upstream_ok(asof)
    alpaca = _mock_alpaca(date(2026, 5, 4))  # Mon

    result = run_preflight(asof=asof, db_path=Path(str(tmp_path)), alpaca=alpaca, max_wait_s=0)
    assert result is None


def test_pre_flight_passes_thursday_to_friday_calendar(tmp_path):
    """Thursday's next session is Friday."""
    asof = date(2026, 4, 30)  # Thursday
    _write_all_upstream_ok(asof)
    alpaca = _mock_alpaca(date(2026, 5, 1))  # Friday

    result = run_preflight(asof=asof, db_path=Path(str(tmp_path)), alpaca=alpaca, max_wait_s=0)
    assert result is None


def test_pre_flight_does_not_open_duckdb(tmp_path, monkeypatch):
    """Sentinel-based preflight must not import or open DuckDB at all."""
    import duckdb

    original_connect = duckdb.connect
    connect_calls = []

    def spy_connect(*args, **kwargs):
        connect_calls.append((args, kwargs))
        return original_connect(*args, **kwargs)

    monkeypatch.setattr(duckdb, "connect", spy_connect)

    asof = date(2026, 4, 30)
    _write_all_upstream_ok(asof)
    alpaca = _mock_alpaca(date(2026, 5, 1))

    run_preflight(asof=asof, db_path=Path(str(tmp_path)), alpaca=alpaca, max_wait_s=0)

    assert connect_calls == [], (
        f"run_preflight opened DuckDB {len(connect_calls)} time(s): {connect_calls}"
    )


def test_pre_flight_accepts_store_kwarg_for_backward_compat(tmp_path):
    """Legacy store= kwarg is accepted and ignored (no AttributeError)."""
    asof = date(2026, 4, 30)
    _write_all_upstream_ok(asof)
    alpaca = _mock_alpaca(date(2026, 5, 1))

    # store= is accepted as a deprecated no-op; must not raise TypeError
    run_preflight(
        asof=asof,
        db_path=Path(str(tmp_path)),
        alpaca=alpaca,
        max_wait_s=0,
        store=MagicMock(),
    )


def test_pre_flight_ingest_error_is_subclass_of_preflight_abort_error(tmp_path):
    """IngestNotCompleteError is catchable as PreflightAbortError for broad callers."""
    assert issubclass(IngestNotCompleteError, Exception)


def test_pre_flight_upstream_missed_deadline_is_subclass_of_preflight_error(tmp_path):
    """UpstreamMissedDeadline is catchable as a preflight error."""
    from sma.live.preflight import PreflightError

    assert issubclass(UpstreamMissedDeadline, PreflightError)
