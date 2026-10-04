"""_pre_close_skip_reason: a same-day account_snapshots row must not be
written before its session has actually closed (or on a non-session day).

2026-08-20 live bug: the Mac booted at 05:23 and a launchd catch-up ran
reconcile at ~05:36, before 2026-08-20's session had even OPENED. The write
still went through -- get_account() at that hour prices off Wednesday
night's after-hours mark, not anything about "today" -- so the DB's 8/20
"close" was actually $121,930.99 (Wed-night marks) against a live $121,534,
and stood as the day's row until the next post-close reconcile.
"""

from datetime import date, datetime
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import pytest

from sma.live.reconcile import _pre_close_skip_reason

ET = ZoneInfo("America/New_York")
DAY = date(2026, 8, 20)  # a regular Thursday session: 09:30-16:00 ET


def _alpaca_with_window(window):
    a = MagicMock()
    a.session_window.return_value = window
    return a


def test_pre_open_run_skips():
    """05:36 catch-up, session hasn't opened (let alone closed) -- must skip."""
    alpaca = _alpaca_with_window(
        (datetime(2026, 8, 20, 9, 30, tzinfo=ET), datetime(2026, 8, 20, 16, 0, tzinfo=ET))
    )
    reason = _pre_close_skip_reason(
        snapshot_date=DAY, now=datetime(2026, 8, 20, 5, 36, tzinfo=ET), alpaca=alpaca,
    )
    assert reason is not None
    assert "has not closed yet" in reason


def test_mid_session_run_skips():
    """Session open but not yet closed -- also must skip, not just pre-open."""
    alpaca = _alpaca_with_window(
        (datetime(2026, 8, 20, 9, 30, tzinfo=ET), datetime(2026, 8, 20, 16, 0, tzinfo=ET))
    )
    reason = _pre_close_skip_reason(
        snapshot_date=DAY, now=datetime(2026, 8, 20, 11, 0, tzinfo=ET), alpaca=alpaca,
    )
    assert reason is not None


def test_post_close_run_writes():
    """The normal 16:30 ET case: session has closed -- safe to write."""
    alpaca = _alpaca_with_window(
        (datetime(2026, 8, 20, 9, 30, tzinfo=ET), datetime(2026, 8, 20, 16, 0, tzinfo=ET))
    )
    reason = _pre_close_skip_reason(
        snapshot_date=DAY, now=datetime(2026, 8, 20, 16, 30, tzinfo=ET), alpaca=alpaca,
    )
    assert reason is None


def test_exact_close_boundary_writes():
    """now == close is safe to write (not strictly after)."""
    alpaca = _alpaca_with_window(
        (datetime(2026, 8, 20, 9, 30, tzinfo=ET), datetime(2026, 8, 20, 16, 0, tzinfo=ET))
    )
    reason = _pre_close_skip_reason(
        snapshot_date=DAY, now=datetime(2026, 8, 20, 16, 0, tzinfo=ET), alpaca=alpaca,
    )
    assert reason is None


def test_non_session_day_skips():
    """session_window returns None (weekend/holiday) -- must skip, never write
    a snapshot for a day that isn't a trading session at all."""
    alpaca = _alpaca_with_window(None)
    reason = _pre_close_skip_reason(
        snapshot_date=date(2026, 8, 22), now=datetime(2026, 8, 22, 20, 0, tzinfo=ET),
        alpaca=alpaca,
    )
    assert reason is not None
    assert "not an NYSE trading session" in reason


def test_half_day_before_1300_close_skips():
    """2026-11-27 (Thanksgiving half-day): session closes 13:00, not 16:00.
    A run at 12:00 must still skip -- the close hasn't happened yet."""
    half_day = date(2026, 11, 27)
    alpaca = _alpaca_with_window(
        (datetime(2026, 11, 27, 9, 30, tzinfo=ET), datetime(2026, 11, 27, 13, 0, tzinfo=ET))
    )
    reason = _pre_close_skip_reason(
        snapshot_date=half_day, now=datetime(2026, 11, 27, 12, 0, tzinfo=ET), alpaca=alpaca,
    )
    assert reason is not None


def test_half_day_after_1300_close_writes():
    """Same half-day, run at 14:00 -- past the REAL 13:00 close -- must write.
    A hardcoded 16:00 assumption would wrongly skip this."""
    half_day = date(2026, 11, 27)
    alpaca = _alpaca_with_window(
        (datetime(2026, 11, 27, 9, 30, tzinfo=ET), datetime(2026, 11, 27, 13, 0, tzinfo=ET))
    )
    reason = _pre_close_skip_reason(
        snapshot_date=half_day, now=datetime(2026, 11, 27, 14, 0, tzinfo=ET), alpaca=alpaca,
    )
    assert reason is None


def test_calendar_failure_before_1600_fallback_skips():
    """session_window raising (calendar unreachable) falls back to treating
    16:00 ET as the close -- a 14:00 run must still skip."""
    alpaca = MagicMock()
    alpaca.session_window.side_effect = RuntimeError("alpaca calendar unreachable")
    reason = _pre_close_skip_reason(
        snapshot_date=DAY, now=datetime(2026, 8, 20, 14, 0, tzinfo=ET), alpaca=alpaca,
    )
    assert reason is not None


def test_calendar_failure_after_1600_fallback_writes():
    """Same calendar failure, but at 16:30 -- past the fallback 16:00 close --
    must write, preserving the normal 16:30 reconcile even when the calendar
    call inside the guard itself is down."""
    alpaca = MagicMock()
    alpaca.session_window.side_effect = RuntimeError("alpaca calendar unreachable")
    reason = _pre_close_skip_reason(
        snapshot_date=DAY, now=datetime(2026, 8, 20, 16, 30, tzinfo=ET), alpaca=alpaca,
    )
    assert reason is None


def test_skip_reason_names_the_date():
    alpaca = _alpaca_with_window(None)
    reason = _pre_close_skip_reason(
        snapshot_date=date(2026, 8, 22), now=datetime(2026, 8, 22, 20, 0, tzinfo=ET),
        alpaca=alpaca,
    )
    assert "2026-08-22" in reason


@pytest.mark.parametrize("hour", [0, 1, 5, 9])
def test_various_pre_open_hours_all_skip(hour):
    """A launchd catch-up could fire at any dark-machine wake hour -- all of
    them, not just 05:36, must skip."""
    alpaca = _alpaca_with_window(
        (datetime(2026, 8, 20, 9, 30, tzinfo=ET), datetime(2026, 8, 20, 16, 0, tzinfo=ET))
    )
    reason = _pre_close_skip_reason(
        snapshot_date=DAY, now=datetime(2026, 8, 20, hour, 0, tzinfo=ET), alpaca=alpaca,
    )
    assert reason is not None
