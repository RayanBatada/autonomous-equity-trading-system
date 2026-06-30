"""earnings_blackout unit tests (was tested via integration in tests/integration/...)."""

from datetime import date

from sma.risk.earnings_blackout import in_earnings_blackout


def test_blackout_blocks_3_days_before_earnings():
    earnings = {"AAPL": [date(2026, 1, 25)]}
    # 3 days before
    assert in_earnings_blackout("AAPL", date(2026, 1, 22), earnings) is True
    # 1 day before
    assert in_earnings_blackout("AAPL", date(2026, 1, 24), earnings) is True
    # earnings day itself
    assert in_earnings_blackout("AAPL", date(2026, 1, 25), earnings) is True


def test_blackout_does_not_block_4_days_before():
    earnings = {"AAPL": [date(2026, 1, 25)]}
    assert in_earnings_blackout("AAPL", date(2026, 1, 21), earnings) is False


def test_blackout_does_not_block_after_earnings():
    earnings = {"AAPL": [date(2026, 1, 25)]}
    assert in_earnings_blackout("AAPL", date(2026, 1, 26), earnings) is False


def test_blackout_returns_false_for_missing_ticker():
    earnings = {"MSFT": [date(2026, 1, 25)]}
    assert in_earnings_blackout("AAPL", date(2026, 1, 22), earnings) is False


def test_blackout_handles_multiple_upcoming_earnings():
    earnings = {"AAPL": [date(2026, 1, 25), date(2026, 4, 25)]}
    assert in_earnings_blackout("AAPL", date(2026, 4, 23), earnings) is True
    assert in_earnings_blackout("AAPL", date(2026, 1, 23), earnings) is True
    # Between the two — neither in window
    assert in_earnings_blackout("AAPL", date(2026, 3, 1), earnings) is False
