"""Data-freshness banner for the Pipeline tab.

The per-job pipeline status is *today-scoped*: each job reports PENDING until
its scheduled fire time. That means a DB frozen for days (the 6/1-6/3 freeze)
shows benign gray PENDING every morning until the day's fire time — a
multi-day freeze is indistinguishable from a normal pre-market morning.

These tests pin a freshness check that is INDEPENDENT of today's schedule:
compare the latest data we actually have against the most recent *completed*
NYSE session and count how many trading sessions we're behind.
"""

from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

from dashboard.tabs.pipeline import (
    _data_staleness_trading_days,
    _most_recent_completed_trading_day,
)

ET = ZoneInfo("America/New_York")


def _et(y: int, m: int, d: int, hh: int, mm: int = 0) -> datetime:
    return datetime(y, m, d, hh, mm, tzinfo=ET)


# ── _most_recent_completed_trading_day ───────────────────────────────────


def test_most_recent_completed_is_today_after_close():
    # Thursday 2026-06-04 17:00 ET — cash session has closed.
    assert _most_recent_completed_trading_day(_et(2026, 6, 4, 17)) == date(2026, 6, 4)


def test_most_recent_completed_is_prior_day_before_close():
    # Thursday 2026-06-04 10:00 ET — today's session not yet closed.
    assert _most_recent_completed_trading_day(_et(2026, 6, 4, 10)) == date(2026, 6, 3)


def test_most_recent_completed_skips_weekend():
    # Saturday 2026-06-06 — most recent completed session is Friday 6/5.
    assert _most_recent_completed_trading_day(_et(2026, 6, 6, 12)) == date(2026, 6, 5)


def test_most_recent_completed_skips_holiday():
    # Memorial Day Monday 2026-05-25 (holiday) — falls back to Friday 5/22.
    assert _most_recent_completed_trading_day(_et(2026, 5, 25, 12)) == date(2026, 5, 22)


# ── _data_staleness_trading_days ─────────────────────────────────────────


def test_fresh_when_data_through_last_completed_session():
    # After Thursday's close with Thursday's data → 0 sessions behind.
    assert _data_staleness_trading_days(date(2026, 6, 4), _et(2026, 6, 4, 17)) == 0


def test_fresh_before_close_with_prior_session_data():
    # Thursday morning, today not expected yet; we have Wednesday's data → 0.
    assert _data_staleness_trading_days(date(2026, 6, 3), _et(2026, 6, 4, 10)) == 0


def test_one_session_behind_before_today_close():
    # Thursday morning, latest is Tuesday → Wednesday's session is missing → 1.
    assert _data_staleness_trading_days(date(2026, 6, 2), _et(2026, 6, 4, 10)) == 1


def test_three_session_freeze_flagged():
    # The freeze shape: latest Monday 6/1, now Thursday 6/4 after close.
    # Missing sessions: Tue 6/2, Wed 6/3, Thu 6/4 → 3.
    assert _data_staleness_trading_days(date(2026, 6, 1), _et(2026, 6, 4, 19)) == 3


def test_weekend_with_friday_data_is_fresh():
    # Saturday with Friday's data → 0 (no session to be behind on).
    assert _data_staleness_trading_days(date(2026, 6, 5), _et(2026, 6, 6, 12)) == 0


def test_weekend_stale_counts_only_trading_sessions():
    # Saturday, latest Wednesday 6/3 → missing Thu 6/4 + Fri 6/5 = 2.
    assert _data_staleness_trading_days(date(2026, 6, 3), _et(2026, 6, 6, 12)) == 2


def test_holiday_not_counted_as_a_missing_session():
    # Tuesday 5/26 after close, latest Friday 5/22. Monday 5/25 is Memorial
    # Day (no session), so only 5/26 is missing → 1, not 2.
    assert _data_staleness_trading_days(date(2026, 5, 22), _et(2026, 5, 26, 17)) == 1


def test_no_data_at_all_alarms():
    # An empty DB must alarm, not read as fresh.
    assert _data_staleness_trading_days(None, _et(2026, 6, 4, 17)) > 0
