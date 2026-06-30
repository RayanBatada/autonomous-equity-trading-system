"""Unit tests for the earnings blackout helper."""

from datetime import date

from sma.backtest.earnings_blackout import in_earnings_blackout


def test_in_blackout_when_within_3_days_before_earnings():
    """asof 2 days before earnings => blackout fires."""
    earnings = {"AAA": [date(2025, 7, 17)]}
    assert in_earnings_blackout("AAA", date(2025, 7, 15), earnings, blackout_days=3) is True


def test_not_in_blackout_when_well_before_earnings():
    """asof 16 days before earnings => no blackout."""
    earnings = {"AAA": [date(2025, 7, 17)]}
    assert in_earnings_blackout("AAA", date(2025, 7, 1), earnings, blackout_days=3) is False


def test_not_in_blackout_after_earnings():
    """asof 3 days AFTER earnings => no blackout (blackout is only before)."""
    earnings = {"AAA": [date(2025, 7, 17)]}
    assert in_earnings_blackout("AAA", date(2025, 7, 20), earnings, blackout_days=3) is False


def test_not_in_blackout_when_no_earnings_for_ticker():
    """Ticker with no entry in earnings dict => no blackout."""
    earnings = {"AAA": [date(2025, 7, 17)]}
    assert in_earnings_blackout("BBB", date(2025, 7, 15), earnings, blackout_days=3) is False


def test_in_blackout_on_earnings_date_itself():
    """asof == earnings date is also within the blackout window."""
    earnings = {"AAA": [date(2025, 7, 17)]}
    assert in_earnings_blackout("AAA", date(2025, 7, 17), earnings, blackout_days=3) is True


def test_in_blackout_exactly_at_boundary():
    """asof exactly blackout_days before earnings is on the edge and should be blocked."""
    earnings = {"AAA": [date(2025, 7, 17)]}
    # 3 days before: 2025-07-14. asof + 3 == 2025-07-17 == earnings => True
    assert in_earnings_blackout("AAA", date(2025, 7, 14), earnings, blackout_days=3) is True
    # 4 days before: 2025-07-13. asof + 3 == 2025-07-16 < earnings => False
    assert in_earnings_blackout("AAA", date(2025, 7, 13), earnings, blackout_days=3) is False
