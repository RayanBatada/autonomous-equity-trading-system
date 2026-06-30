"""Tests for the ingest market-holiday short-circuit.

Without this, Mon 5/25 Memorial Day (and every market-closed weekday) runs
the full ingest pipeline, fails the all_tickers_have_price quality check,
and writes a noisy `quality.passed=False` sentinel. The skip path writes a
clean `quality.passed=True, holiday_skipped=True` sentinel so downstream
preflight can distinguish "ingest broke" from "market closed".
"""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

from sma.ingest.__main__ import (
    _is_us_market_holiday_weekday,
    _write_holiday_skipped_sentinel,
)


def _stub_settings(*, has_secrets: bool = True):
    s = MagicMock()
    s.secrets.alpaca_api_key = "key" if has_secrets else ""
    s.secrets.alpaca_api_secret = "secret" if has_secrets else ""
    return s


def test_weekend_is_not_treated_as_holiday():
    """The ingest plist doesn't fire on Sat/Sun anyway, but make this
    explicit so a future cron rewrite can't accidentally skip a weekend
    backfill."""
    saturday = date(2026, 5, 23)
    sunday = date(2026, 5, 24)
    assert _is_us_market_holiday_weekday(saturday, _stub_settings()) is False
    assert _is_us_market_holiday_weekday(sunday, _stub_settings()) is False


def test_calendar_lookup_failure_falls_through_to_run():
    """If Alpaca is unreachable, default to RUNNING ingest. False-skipping
    a real trading day costs more than running on a holiday (the latter
    just wastes one cycle; the former permanently loses that day's data)."""
    monday_memorial_day = date(2026, 5, 25)
    settings = _stub_settings(has_secrets=False)
    assert _is_us_market_holiday_weekday(monday_memorial_day, settings) is False


def test_market_holiday_weekday_returns_true(monkeypatch):
    """Memorial Day Mon 5/25 — alpaca returns empty sessions → skip ingest."""
    from sma.live import alpaca_client as _ac

    fake = MagicMock()
    fake.sessions_between.return_value = []  # holiday → no session
    monkeypatch.setattr(_ac.AlpacaClient, "paper_from_env", classmethod(lambda cls, **kw: fake))

    memorial_day = date(2026, 5, 25)
    assert _is_us_market_holiday_weekday(memorial_day, _stub_settings()) is True


def test_normal_weekday_returns_false(monkeypatch):
    """A regular Tuesday — alpaca returns the date itself → don't skip."""
    from sma.live import alpaca_client as _ac

    tue = date(2026, 5, 26)
    fake = MagicMock()
    fake.sessions_between.return_value = [tue]  # a trading session
    monkeypatch.setattr(_ac.AlpacaClient, "paper_from_env", classmethod(lambda cls, **kw: fake))

    assert _is_us_market_holiday_weekday(tue, _stub_settings()) is False


def test_holiday_skipped_sentinel_marks_quality_passed(monkeypatch, tmp_path):
    """Sentinel must set quality.passed=True so decide preflight doesn't
    block on a stale quality failure."""
    import sma.sentinels as _sentinels
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path / "sentinels"))

    memorial_day = date(2026, 5, 25)
    _write_holiday_skipped_sentinel(asof=memorial_day)

    sentinel = _sentinels.read_sentinel(label="com.sma.ingest.daily", asof=memorial_day)
    assert sentinel is not None
    assert sentinel["quality"]["passed"] is True
    assert sentinel["quality"]["blocking_failures"] == []
    assert sentinel["holiday_skipped"] is True
    # asof + completed_at always present so monitoring + dashboard can parse
    assert sentinel["asof"] == memorial_day.isoformat()
    assert sentinel["completed_at"].endswith("Z")
