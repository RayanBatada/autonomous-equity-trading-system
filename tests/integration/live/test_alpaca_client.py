"""AlpacaClient wrapper tests using MagicMock(spec=TradingClient). No real network."""

from datetime import date, datetime, timedelta
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import pytest
from alpaca.common.exceptions import APIError
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import QueryOrderStatus

from sma.live.alpaca_client import (
    NEXT_SESSION_LOOKAHEAD_DAYS,
    AlpacaClient,
)


def _mock_account(
    equity=100_000.0, cash=50_000.0, buying_power=100_000.0,
    long_market_value=50_000.0, trading_blocked=False, account_blocked=False,
):
    a = MagicMock()
    a.equity = str(equity)
    a.cash = str(cash)
    a.buying_power = str(buying_power)
    a.long_market_value = str(long_market_value)
    a.trading_blocked = trading_blocked
    a.account_blocked = account_blocked
    return a


def test_get_account_returns_normalized_dict():
    mock_tc = MagicMock(spec=TradingClient)
    mock_tc.get_account.return_value = _mock_account()
    client = AlpacaClient(trading_client=mock_tc)

    acct = client.get_account()
    assert acct["equity"] == 100_000.0
    assert acct["cash"] == 50_000.0
    assert acct["buying_power"] == 100_000.0
    assert acct["long_market_value"] == 50_000.0
    assert acct["trading_blocked"] is False
    assert acct["account_blocked"] is False


def test_get_account_validates_trading_blocked_flag():
    mock_tc = MagicMock(spec=TradingClient)
    mock_tc.get_account.return_value = _mock_account(trading_blocked=True)
    client = AlpacaClient(trading_client=mock_tc)
    acct = client.get_account()
    assert acct["trading_blocked"] is True


def test_get_account_validates_account_blocked_flag():
    mock_tc = MagicMock(spec=TradingClient)
    mock_tc.get_account.return_value = _mock_account(account_blocked=True)
    client = AlpacaClient(trading_client=mock_tc)
    acct = client.get_account()
    assert acct["account_blocked"] is True


def test_get_positions_handles_empty_account():
    mock_tc = MagicMock(spec=TradingClient)
    mock_tc.get_all_positions.return_value = []
    client = AlpacaClient(trading_client=mock_tc)
    assert client.get_positions() == {}


def test_get_positions_returns_ticker_to_shares_map():
    mock_tc = MagicMock(spec=TradingClient)
    pos1 = MagicMock(symbol="AAPL", qty="25", avg_entry_price="200.0")
    pos2 = MagicMock(symbol="MSFT", qty="10", avg_entry_price="400.0")
    mock_tc.get_all_positions.return_value = [pos1, pos2]
    client = AlpacaClient(trading_client=mock_tc)
    positions = client.get_positions()
    assert positions == {
        "AAPL": {"shares": 25, "cost_basis": 200.0},
        "MSFT": {"shares": 10, "cost_basis": 400.0},
    }


def test_submit_day_opg_order_calls_alpaca_with_correct_params():
    mock_tc = MagicMock(spec=TradingClient)
    returned = MagicMock()
    returned.id = "ord-123"
    mock_tc.submit_order.return_value = returned
    client = AlpacaClient(trading_client=mock_tc)

    order_id = client.submit_day_opg_buy("AAPL", shares=10)
    assert order_id == "ord-123"
    mock_tc.submit_order.assert_called_once()
    req = mock_tc.submit_order.call_args[0][0]
    assert req.symbol == "AAPL"
    assert int(req.qty) == 10
    # OrderSide enum check via .value or repr — just confirm BUY direction
    assert "BUY" in str(req.side).upper()


def test_submit_market_sell_calls_alpaca_with_correct_params():
    mock_tc = MagicMock(spec=TradingClient)
    returned = MagicMock()
    returned.id = "ord-456"
    mock_tc.submit_order.return_value = returned
    client = AlpacaClient(trading_client=mock_tc)

    order_id = client.submit_market_sell("AAPL", shares=5)
    assert order_id == "ord-456"
    mock_tc.submit_order.assert_called_once()
    req = mock_tc.submit_order.call_args[0][0]
    assert "SELL" in str(req.side).upper()


def test_get_calendar_returns_next_session_date():
    mock_tc = MagicMock(spec=TradingClient)
    cal_entry = MagicMock(date=date(2026, 5, 4))
    mock_tc.get_calendar.return_value = [cal_entry]
    client = AlpacaClient(trading_client=mock_tc)

    next_date = client.next_session_date(today=date(2026, 5, 1))
    assert next_date == date(2026, 5, 4)


def test_next_session_date_queries_forward_window_not_today():
    """Regression for codex Finding 1: the request must span (today, today+10],
    not just today, otherwise Alpaca returns today's own session entry.
    """
    mock_tc = MagicMock(spec=TradingClient)
    cal_entry = MagicMock(date=date(2026, 5, 4))
    mock_tc.get_calendar.return_value = [cal_entry]
    client = AlpacaClient(trading_client=mock_tc)

    today = date(2026, 5, 1)
    client.next_session_date(today=today)

    mock_tc.get_calendar.assert_called_once()
    req = mock_tc.get_calendar.call_args.kwargs["filters"]
    assert req.start == today + timedelta(days=1)
    assert req.end == today + timedelta(days=NEXT_SESSION_LOOKAHEAD_DAYS)
    assert req.start != today, "must NOT include today, or Alpaca returns today's session"


def test_get_calendar_raises_on_empty_response():
    mock_tc = MagicMock(spec=TradingClient)
    mock_tc.get_calendar.return_value = []
    client = AlpacaClient(trading_client=mock_tc)

    with pytest.raises(RuntimeError, match="empty calendar"):
        client.next_session_date(today=date(2026, 5, 1))


def test_sessions_between_returns_dates_for_requested_range():
    """sessions_between sends a calendar request with the inclusive range and
    returns the date attribute of each entry. Tests assert the request shape
    so a wrapper bug (e.g., off-by-one bounds) gets caught.
    """
    mock_tc = MagicMock(spec=TradingClient)
    e1 = MagicMock(date=date(2026, 5, 11))
    e2 = MagicMock(date=date(2026, 5, 12))
    mock_tc.get_calendar.return_value = [e1, e2]
    client = AlpacaClient(trading_client=mock_tc)

    start = date(2026, 5, 11)
    end = date(2026, 5, 13)
    result = client.sessions_between(start=start, end=end)

    assert result == [date(2026, 5, 11), date(2026, 5, 12)]
    mock_tc.get_calendar.assert_called_once()
    req = mock_tc.get_calendar.call_args.kwargs["filters"]
    assert req.start == start
    assert req.end == end


def test_sessions_between_returns_empty_when_start_after_end():
    """No API call when range is inverted — defensive guard."""
    mock_tc = MagicMock(spec=TradingClient)
    client = AlpacaClient(trading_client=mock_tc)

    result = client.sessions_between(
        start=date(2026, 5, 15), end=date(2026, 5, 14),
    )
    assert result == []
    mock_tc.get_calendar.assert_not_called()


def test_get_orders_for_date_uses_status_all_and_full_day_window():
    """Regression for codex Finding 2: GetOrdersRequest must request status=ALL
    (defaults to OPEN otherwise) and use a full ET-midnight-to-midnight window
    (not a zero-width same-date range).
    """
    mock_tc = MagicMock(spec=TradingClient)
    mock_tc.get_orders.return_value = []
    client = AlpacaClient(trading_client=mock_tc)

    target = date(2026, 5, 1)
    client.get_orders_for_date(target)

    mock_tc.get_orders.assert_called_once()
    req = mock_tc.get_orders.call_args.kwargs["filter"]
    assert req.status == QueryOrderStatus.ALL, (
        "must explicitly request status=ALL or alpaca-py defaults to OPEN, "
        "filtering out filled/canceled orders the reconcile code expects to see"
    )
    # Window must be 24h, ET-midnight-to-midnight, both bounds tz-aware.
    et = ZoneInfo("America/New_York")
    expected_after = datetime.combine(target, datetime.min.time(), tzinfo=et)
    expected_until = expected_after + timedelta(days=1)
    assert req.after == expected_after
    assert req.until == expected_until
    assert (req.until - req.after) == timedelta(days=1), (
        "zero-width or shorter window returns no orders for the day"
    )


def test_apierror_propagates_with_context():
    mock_tc = MagicMock(spec=TradingClient)
    mock_tc.submit_order.side_effect = APIError("insufficient buying power")
    client = AlpacaClient(trading_client=mock_tc)

    with pytest.raises(APIError, match="insufficient buying power"):
        client.submit_day_opg_buy("AAPL", shares=10)
