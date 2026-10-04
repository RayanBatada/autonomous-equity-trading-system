"""Unit tests for autoresearch's trading-day deadline guard.

2026-08-24 audit: on Mon 2026-08-17, autoresearch (fires 07:00 ET) finished at
09:22 — two minutes past the 09:20 pre-market boundary that the live stop-loss
sweep needs the writer lock free by. That day's 09:25 stop-loss sweep never
logged at all (see live.stop-loss.*.log — zero entries for 2026-08-17,
confirming it was squeezed out). 1ee39b1 (2026-08-17, landed AFTER that
morning's run) cut autoresearch's feature-build cost ~2.2x; 2026-08-24's run
finished in 56 minutes (07:00-07:56), a comfortable margin — but that's one
data point against a search whose runtime has ranged 56min-7h across recent
Mondays (config count / data volume dependent), so the guard is cheap
insurance mirroring agents.__main__._deadline_reached's same-day-only shape.
"""
from datetime import date, datetime
from zoneinfo import ZoneInfo

from sma.autoresearch.__main__ import (
    _AUTORESEARCH_TRADING_DAY_CUTOFF_HOUR_MINUTE_ET,
    _search_deadline_reached,
)

ET = ZoneInfo("America/New_York")


def test_cutoff_constant_is_08_45():
    assert _AUTORESEARCH_TRADING_DAY_CUTOFF_HOUR_MINUTE_ET == (8, 45)


def test_not_reached_before_cutoff_on_same_day():
    now_et = datetime(2026, 8, 24, 8, 44, 59, tzinfo=ET)
    assert _search_deadline_reached(now_et=now_et, asof=date(2026, 8, 24)) is False


def test_reached_at_cutoff_on_same_day():
    now_et = datetime(2026, 8, 24, 8, 45, 0, tzinfo=ET)
    assert _search_deadline_reached(now_et=now_et, asof=date(2026, 8, 24)) is True


def test_reached_after_cutoff_on_same_day():
    now_et = datetime(2026, 8, 24, 9, 22, 0, tzinfo=ET)
    assert _search_deadline_reached(now_et=now_et, asof=date(2026, 8, 24)) is True


def test_not_reached_for_historical_asof_even_past_cutoff_time_of_day():
    # A manual/backtest run against an old asof must stay unbudgeted — mirrors
    # agents._deadline_reached's "SCHEDULED same-day run only" rule.
    now_et = datetime(2026, 8, 24, 10, 0, 0, tzinfo=ET)
    assert _search_deadline_reached(now_et=now_et, asof=date(2026, 6, 10)) is False


def test_not_reached_for_future_asof():
    now_et = datetime(2026, 8, 24, 10, 0, 0, tzinfo=ET)
    assert _search_deadline_reached(now_et=now_et, asof=date(2026, 8, 31)) is False
