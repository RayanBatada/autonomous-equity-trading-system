"""AlpacaClient wrapper tests using MagicMock(spec=TradingClient). No real network."""

from datetime import date, datetime, timedelta
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pytest
import requests
from alpaca.common.exceptions import APIError
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import QueryOrderStatus

from sma.live.alpaca_client import (
    NEXT_SESSION_LOOKAHEAD_DAYS,
    AlpacaClient,
)


def _mock_account(
    equity=100_000.0,
    cash=50_000.0,
    buying_power=100_000.0,
    long_market_value=50_000.0,
    trading_blocked=False,
    account_blocked=False,
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
        start=date(2026, 5, 15),
        end=date(2026, 5, 14),
    )
    assert result == []
    mock_tc.get_calendar.assert_not_called()


def test_session_window_localizes_naive_calendar_datetimes_to_et():
    """Alpaca returns Calendar.open/close as NAIVE datetimes carrying the ET
    wall clock (probed live 2026-08-16). They must be LOCALIZED to ET, never
    read as UTC — a UTC reading would put the open at 05:30 ET."""
    mock_tc = MagicMock(spec=TradingClient)
    day = date(2026, 8, 17)
    mock_tc.get_calendar.return_value = [
        MagicMock(date=day, open=datetime(2026, 8, 17, 9, 30), close=datetime(2026, 8, 17, 16, 0))
    ]
    client = AlpacaClient(trading_client=mock_tc)

    et = ZoneInfo("America/New_York")
    assert client.session_window(day=day) == (
        datetime(2026, 8, 17, 9, 30, tzinfo=et),
        datetime(2026, 8, 17, 16, 0, tzinfo=et),
    )
    req = mock_tc.get_calendar.call_args.kwargs["filters"]
    assert req.start == day
    assert req.end == day


def test_session_window_reports_the_real_half_day_close():
    """2026-11-27 closes at 13:00 ET, not 16:00 — the whole point of asking
    the calendar instead of hardcoding regular hours."""
    mock_tc = MagicMock(spec=TradingClient)
    day = date(2026, 11, 27)
    mock_tc.get_calendar.return_value = [
        MagicMock(date=day, open=datetime(2026, 11, 27, 9, 30), close=datetime(2026, 11, 27, 13, 0))
    ]
    client = AlpacaClient(trading_client=mock_tc)

    session_open, session_close = client.session_window(day=day)
    assert (session_open.hour, session_open.minute) == (9, 30)
    assert (session_close.hour, session_close.minute) == (13, 0)


def test_session_window_accepts_time_typed_calendar_fields():
    """Older alpaca-py typed Calendar.open/close as datetime.time. Both shapes
    must resolve to the same ET-aware datetime on `day`."""
    from datetime import time as time_cls

    mock_tc = MagicMock(spec=TradingClient)
    day = date(2026, 8, 17)
    mock_tc.get_calendar.return_value = [
        MagicMock(date=day, open=time_cls(9, 30), close=time_cls(16, 0))
    ]
    client = AlpacaClient(trading_client=mock_tc)

    et = ZoneInfo("America/New_York")
    assert client.session_window(day=day) == (
        datetime(2026, 8, 17, 9, 30, tzinfo=et),
        datetime(2026, 8, 17, 16, 0, tzinfo=et),
    )


def test_session_window_returns_none_on_a_holiday():
    """An empty calendar for the day means it is not an NYSE session."""
    mock_tc = MagicMock(spec=TradingClient)
    mock_tc.get_calendar.return_value = []
    client = AlpacaClient(trading_client=mock_tc)

    assert client.session_window(day=date(2026, 7, 3)) is None


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


# --- session_close_equity (2026-08-12) -------------------------------------
# get_account().equity is a LIVE mark; after ~16:00 ET it prices the book off
# after-hours quotes. These cover the two portfolio-history sources the
# snapshot path prefers instead, and the "not published yet" hole between them.


def _ts(y, m, d, hh, mm=0):
    """Epoch seconds for an ET wall-clock instant (Alpaca returns epochs)."""
    return int(datetime(y, m, d, hh, mm, tzinfo=ZoneInfo("America/New_York")).timestamp())


def test_session_close_equity_prefers_the_published_daily_bar():
    mock_tc = MagicMock(spec=TradingClient)
    # Alpaca stamps the 1D bar at end-of-day; 8/11 20:00 ET is the 8/11 session.
    mock_tc.get_portfolio_history.return_value = MagicMock(
        timestamp=[_ts(2026, 8, 11, 20), _ts(2026, 8, 12, 20)],
        equity=[101923.87, 102343.20],
    )
    client = AlpacaClient(trading_client=mock_tc)

    assert client.session_close_equity(day=date(2026, 8, 12)) == (
        102343.20,
        "portfolio_history_daily",
    )
    assert mock_tc.get_portfolio_history.call_count == 1  # no minute fallback


def _calendar(day, *, close_hour=16, close_minute=0):
    return [
        MagicMock(
            date=day,
            open=datetime(day.year, day.month, day.day, 9, 30),
            close=datetime(day.year, day.month, day.day, close_hour, close_minute),
        )
    ]


def test_session_close_equity_falls_back_to_the_1600_minute_point():
    """The common evening case: the 1D bar for today is not published yet."""
    mock_tc = MagicMock(spec=TradingClient)
    daily = MagicMock(timestamp=[_ts(2026, 8, 11, 20)], equity=[101923.87])
    minutes = MagicMock(
        timestamp=[_ts(2026, 8, 12, 15, 59), _ts(2026, 8, 12, 16, 0)],
        equity=[102381.37, 102343.20],
    )
    mock_tc.get_portfolio_history.side_effect = [daily, minutes]
    mock_tc.get_calendar.return_value = _calendar(date(2026, 8, 12))
    client = AlpacaClient(trading_client=mock_tc)

    assert client.session_close_equity(day=date(2026, 8, 12)) == (
        102343.20,
        "portfolio_history_1min_close",
    )


def test_session_close_equity_uses_the_half_day_1300_close():
    """2026-11-27 closes at 13:00 ET. Demanding an exact (16, 0) stamp returned
    None on every half-day, sending the snapshot back to the polluted
    get_account read. The LAST in-session point is the close."""
    mock_tc = MagicMock(spec=TradingClient)
    day = date(2026, 11, 27)
    daily = MagicMock(timestamp=[_ts(2026, 11, 25, 20)], equity=[101923.87])
    minutes = MagicMock(
        timestamp=[_ts(2026, 11, 27, 12, 59), _ts(2026, 11, 27, 13, 0)],
        equity=[104880.11, 104903.55],
    )
    mock_tc.get_portfolio_history.side_effect = [daily, minutes]
    mock_tc.get_calendar.return_value = _calendar(day, close_hour=13)
    client = AlpacaClient(trading_client=mock_tc)

    assert client.session_close_equity(day=day) == (
        104903.55,
        "portfolio_history_1min_close",
    )


def test_session_close_equity_survives_a_missing_final_bar():
    """A session whose 16:00 point never arrived: use the last one that did,
    rather than returning None and falling back to the after-hours mark."""
    mock_tc = MagicMock(spec=TradingClient)
    daily = MagicMock(timestamp=[_ts(2026, 8, 11, 20)], equity=[101923.87])
    minutes = MagicMock(
        timestamp=[_ts(2026, 8, 12, 15, 57), _ts(2026, 8, 12, 15, 58)],
        equity=[102300.00, 102340.00],
    )
    mock_tc.get_portfolio_history.side_effect = [daily, minutes]
    mock_tc.get_calendar.return_value = _calendar(date(2026, 8, 12))
    client = AlpacaClient(trading_client=mock_tc)

    assert client.session_close_equity(day=date(2026, 8, 12)) == (
        102340.00,
        "portfolio_history_1min_close",
    )


def test_session_close_equity_ignores_points_outside_regular_hours():
    """An after-hours point must never be mistaken for the close — that is the
    exact mark this whole method exists to avoid."""
    mock_tc = MagicMock(spec=TradingClient)
    daily = MagicMock(timestamp=[_ts(2026, 8, 11, 20)], equity=[101923.87])
    minutes = MagicMock(
        timestamp=[_ts(2026, 8, 12, 9, 15), _ts(2026, 8, 12, 16, 0), _ts(2026, 8, 12, 18, 30)],
        equity=[101000.00, 102343.20, 102516.39],
    )
    mock_tc.get_portfolio_history.side_effect = [daily, minutes]
    mock_tc.get_calendar.return_value = _calendar(date(2026, 8, 12))
    client = AlpacaClient(trading_client=mock_tc)

    assert client.session_close_equity(day=date(2026, 8, 12)) == (
        102343.20,
        "portfolio_history_1min_close",
    )


def test_session_close_equity_returns_none_when_the_session_is_unpublished():
    """Neither source has the day: the caller must fall back to get_account and
    label the row, not invent a close."""
    mock_tc = MagicMock(spec=TradingClient)
    daily = MagicMock(timestamp=[_ts(2026, 8, 11, 20)], equity=[101923.87])
    minutes = MagicMock(timestamp=[], equity=[])
    mock_tc.get_portfolio_history.side_effect = [daily, minutes]
    mock_tc.get_calendar.return_value = _calendar(date(2026, 8, 12))
    client = AlpacaClient(trading_client=mock_tc)

    assert client.session_close_equity(day=date(2026, 8, 12)) is None


def test_session_close_equity_returns_none_on_a_non_session_day():
    """The calendar says `day` was not a trading session: there is no close to
    read, so do not scan minute points for one."""
    mock_tc = MagicMock(spec=TradingClient)
    daily = MagicMock(timestamp=[_ts(2026, 8, 11, 20)], equity=[101923.87])
    mock_tc.get_portfolio_history.side_effect = [daily]
    mock_tc.get_calendar.return_value = []
    client = AlpacaClient(trading_client=mock_tc)

    assert client.session_close_equity(day=date(2026, 7, 3)) is None


def test_session_close_equity_falls_back_to_regular_hours_on_calendar_failure():
    """A calendar blip must not lose the close: assume standard 09:30-16:00."""
    mock_tc = MagicMock(spec=TradingClient)
    daily = MagicMock(timestamp=[_ts(2026, 8, 11, 20)], equity=[101923.87])
    minutes = MagicMock(
        timestamp=[_ts(2026, 8, 12, 16, 0), _ts(2026, 8, 12, 19, 0)],
        equity=[102343.20, 102516.39],
    )
    mock_tc.get_portfolio_history.side_effect = [daily, minutes]
    mock_tc.get_calendar.side_effect = RuntimeError("calendar 503")
    client = AlpacaClient(trading_client=mock_tc)

    assert client.session_close_equity(day=date(2026, 8, 12)) == (
        102343.20,
        "portfolio_history_1min_close",
    )


# --- transient-network retry (2026-08-24) -----------------------------------
# reconcile crashed hard on a transient DNS blip hitting GET /v2/account
# (roving laptop between networks); get_account had zero retry protection.
# Every read-only AlpacaClient method now retries requests.exceptions.
# ConnectionError/Timeout up to 3x with ~2/4/8s backoff -- alpaca-py raises
# those UNWRAPPED (never turned into APIError) when the request never reached
# Alpaca's servers at all, so blind retry is safe. submit_* methods are
# deliberately excluded: a connection error there can occur AFTER the order
# was already sent, so blind retry risks a double-submit.


def test_get_account_retries_on_connection_error_then_succeeds():
    mock_tc = MagicMock(spec=TradingClient)
    mock_tc.get_account.side_effect = [
        requests.exceptions.ConnectionError("dns blip"),
        _mock_account(),
    ]
    client = AlpacaClient(trading_client=mock_tc)

    with patch("sma.live.alpaca_client._SLEEP", lambda _s: None):
        acct = client.get_account()

    assert acct["equity"] == 100_000.0
    assert mock_tc.get_account.call_count == 2


def test_get_account_retries_on_timeout_then_succeeds():
    mock_tc = MagicMock(spec=TradingClient)
    mock_tc.get_account.side_effect = [
        requests.exceptions.Timeout("read timeout"),
        _mock_account(),
    ]
    client = AlpacaClient(trading_client=mock_tc)

    with patch("sma.live.alpaca_client._SLEEP", lambda _s: None):
        acct = client.get_account()

    assert acct["equity"] == 100_000.0
    assert mock_tc.get_account.call_count == 2


def test_get_account_raises_after_three_connection_errors():
    mock_tc = MagicMock(spec=TradingClient)
    mock_tc.get_account.side_effect = requests.exceptions.ConnectionError("dns down")
    client = AlpacaClient(trading_client=mock_tc)

    with (
        patch("sma.live.alpaca_client._SLEEP", lambda _s: None),
        pytest.raises(requests.exceptions.ConnectionError, match="dns down"),
    ):
        client.get_account()

    assert mock_tc.get_account.call_count == 3


def test_get_account_retry_uses_2_4_8s_backoff():
    mock_tc = MagicMock(spec=TradingClient)
    mock_tc.get_account.side_effect = requests.exceptions.ConnectionError("dns down")
    client = AlpacaClient(trading_client=mock_tc)

    sleeps = []
    with (
        patch("sma.live.alpaca_client._SLEEP", sleeps.append),
        pytest.raises(requests.exceptions.ConnectionError),
    ):
        client.get_account()

    # 3 attempts total, but only 2 sleeps -- none after the final failed attempt.
    assert sleeps == [2.0, 4.0]


def test_get_account_does_not_retry_on_apierror():
    mock_tc = MagicMock(spec=TradingClient)
    mock_tc.get_account.side_effect = APIError("insufficient buying power")
    client = AlpacaClient(trading_client=mock_tc)

    sleeps = []
    with (
        patch("sma.live.alpaca_client._SLEEP", sleeps.append),
        pytest.raises(APIError, match="insufficient buying power"),
    ):
        client.get_account()

    assert mock_tc.get_account.call_count == 1
    assert sleeps == []


def test_get_positions_retries_on_connection_error_then_succeeds():
    """The retry lives at the method boundary, not per-callsite -- proves it
    generalizes past get_account (the method that actually crashed live)."""
    mock_tc = MagicMock(spec=TradingClient)
    pos = MagicMock(symbol="AAPL", qty="25", avg_entry_price="200.0")
    mock_tc.get_all_positions.side_effect = [
        requests.exceptions.ConnectionError("dns blip"),
        [pos],
    ]
    client = AlpacaClient(trading_client=mock_tc)

    with patch("sma.live.alpaca_client._SLEEP", lambda _s: None):
        positions = client.get_positions()

    assert positions == {"AAPL": {"shares": 25, "cost_basis": 200.0}}
    assert mock_tc.get_all_positions.call_count == 2


def test_submit_day_opg_buy_does_not_retry_on_connection_error():
    """Order submission stays single-attempt even for a transient network
    error, not just for APIError (test_apierror_propagates_with_context
    covers that side) -- a connection error here may have occurred AFTER the
    order was already sent, so blind retry risks a double-submit."""
    mock_tc = MagicMock(spec=TradingClient)
    mock_tc.submit_order.side_effect = requests.exceptions.ConnectionError("dns blip")
    client = AlpacaClient(trading_client=mock_tc)

    with pytest.raises(requests.exceptions.ConnectionError):
        client.submit_day_opg_buy("AAPL", shares=10)

    assert mock_tc.submit_order.call_count == 1
