"""Account snapshots must not store after-hours marks.

get_account().equity is a LIVE mark: after ~16:00 ET it prices the book off
after-hours quotes rather than the close. 2026-07-29 it was off by thousands.
2026-08-12 the reboot fired reconcile at 20:16 as a launchd catch-up and it
stored equity 102,516.39 for 8/12 against a true 16:00 close of 102,343.20
(0.17% high) -- and every downstream drawdown/catastrophic-loss check reads
that row.

After 16:10 ET the snapshot sources equity from Alpaca's portfolio-history
close instead, and records equity_source either way.
"""

from datetime import date, datetime
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import pytest

from sma.live.reconcile import _snapshot_equity

ET = ZoneInfo("America/New_York")
DAY = date(2026, 8, 12)

# The real 8/12 numbers: cash is mark-independent, so equity - cash is the
# whole of the after-hours error.
ACCOUNT = {
    "equity": 102_516.39,
    "cash": 7_720.23,
    "buying_power": 282_973.07,
    "long_market_value": 94_796.16,
}
TRUE_CLOSE = 102_343.20


def _alpaca(close: tuple[float, str] | None = None, *, raises: bool = False):
    a = MagicMock()
    if raises:
        a.session_close_equity.side_effect = RuntimeError("history 503")
    else:
        a.session_close_equity.return_value = close
    return a


def test_before_cutoff_uses_the_live_read():
    """Inside the session the live read IS the session mark, so take it and
    don't spend an API call on portfolio-history."""
    alpaca = _alpaca()
    equity, lmv, source = _snapshot_equity(
        snapshot_date=DAY, account=ACCOUNT, alpaca=alpaca,
        now=datetime(2026, 8, 12, 15, 45, tzinfo=ET),
    )

    assert equity == ACCOUNT["equity"]
    assert lmv == ACCOUNT["long_market_value"]
    assert source == "get_account"
    alpaca.session_close_equity.assert_not_called()


def test_after_cutoff_prefers_the_portfolio_history_close():
    """The 2026-08-12 case: a 20:16 catch-up reconcile."""
    alpaca = _alpaca((TRUE_CLOSE, "portfolio_history_1min_close"))
    equity, lmv, source = _snapshot_equity(
        snapshot_date=DAY, account=ACCOUNT, alpaca=alpaca,
        now=datetime(2026, 8, 12, 20, 16, tzinfo=ET),
    )

    assert equity == TRUE_CLOSE
    assert source == "portfolio_history_1min_close"
    # equity == cash + long_market_value must still hold on the stored row.
    assert lmv == pytest.approx(TRUE_CLOSE - ACCOUNT["cash"], abs=0.01)
    assert equity + 0 == pytest.approx(ACCOUNT["cash"] + lmv, abs=0.01)


def test_the_normal_1630_reconcile_uses_history_too():
    """16:30 is past the cutoff, so the scheduled run takes the same path --
    this is the fix's steady state, not just a recovery path."""
    alpaca = _alpaca((TRUE_CLOSE, "portfolio_history_daily"))
    equity, _, source = _snapshot_equity(
        snapshot_date=DAY, account=ACCOUNT, alpaca=alpaca,
        now=datetime(2026, 8, 12, 16, 30, tzinfo=ET),
    )

    assert equity == TRUE_CLOSE
    assert source == "portfolio_history_daily"


def test_falls_back_and_labels_when_history_has_no_close_yet():
    """16:10-18:00ish (and empirically later): Alpaca has not published the
    session. Keep the live read but LABEL it so the row is explainable."""
    alpaca = _alpaca(None)
    equity, lmv, source = _snapshot_equity(
        snapshot_date=DAY, account=ACCOUNT, alpaca=alpaca,
        now=datetime(2026, 8, 12, 17, 0, tzinfo=ET),
    )

    assert equity == ACCOUNT["equity"]
    assert lmv == ACCOUNT["long_market_value"]
    assert source == "get_account_after_hours"


def test_history_error_does_not_lose_the_snapshot():
    """A portfolio-history 503 must degrade to the live read, not abort the
    snapshot -- a missing row suppresses catastrophic-loss detection entirely."""
    alpaca = _alpaca(raises=True)
    equity, _, source = _snapshot_equity(
        snapshot_date=DAY, account=ACCOUNT, alpaca=alpaca,
        now=datetime(2026, 8, 12, 20, 16, tzinfo=ET),
    )

    assert equity == ACCOUNT["equity"]
    assert source == "get_account_after_hours"


def test_backdated_snapshot_is_not_rewritten_from_todays_history():
    """Snapshotting a PAST date (a drain of an old batch) must not pull today's
    portfolio-history; the guard is snapshot_date == today."""
    alpaca = _alpaca((TRUE_CLOSE, "portfolio_history_daily"))
    equity, _, source = _snapshot_equity(
        snapshot_date=date(2026, 8, 11), account=ACCOUNT, alpaca=alpaca,
        now=datetime(2026, 8, 12, 20, 16, tzinfo=ET),
    )

    assert equity == ACCOUNT["equity"]
    assert source == "get_account"
    alpaca.session_close_equity.assert_not_called()


def test_cutoff_boundary():
    """16:09 is live, 16:10 is history."""
    alpaca = _alpaca((TRUE_CLOSE, "portfolio_history_daily"))
    _, _, before = _snapshot_equity(
        snapshot_date=DAY, account=ACCOUNT, alpaca=alpaca,
        now=datetime(2026, 8, 12, 16, 9, tzinfo=ET),
    )
    _, _, after = _snapshot_equity(
        snapshot_date=DAY, account=ACCOUNT, alpaca=alpaca,
        now=datetime(2026, 8, 12, 16, 10, tzinfo=ET),
    )

    assert before == "get_account"
    assert after == "portfolio_history_daily"
