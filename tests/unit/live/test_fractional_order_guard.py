"""Alpaca accepts a fractional qty ONLY on time_in_force=DAY, and alpaca-py does
not enforce it — a fractional GTC/OPG request constructs fine locally and is
rejected at the broker, i.e. at 18:35 with the whole batch."""

from unittest.mock import MagicMock

import pytest
from alpaca.trading.enums import TimeInForce

from sma.live.alpaca_client import AlpacaClient, _assert_fractional_is_day


def _client():
    tc = MagicMock()
    tc.submit_order.return_value = MagicMock(id="oid-1")
    return AlpacaClient(trading_client=tc, paper=True)


@pytest.mark.parametrize("tif", [TimeInForce.OPG, TimeInForce.GTC, TimeInForce.IOC,
                                 TimeInForce.FOK, TimeInForce.CLS])
def test_fractional_rejected_on_every_non_day_tif(tif):
    with pytest.raises(ValueError, match="time_in_force=DAY"):
        _assert_fractional_is_day(2.5, tif, "AAPL")


def test_fractional_allowed_on_day():
    _assert_fractional_is_day(2.5, TimeInForce.DAY, "AAPL")


@pytest.mark.parametrize("tif", [TimeInForce.OPG, TimeInForce.GTC, TimeInForce.DAY])
def test_whole_shares_allowed_on_any_tif(tif):
    """Whole-share behaviour is untouched — this guard only ever sees a
    fraction."""
    _assert_fractional_is_day(175, tif, "AAPL")
    _assert_fractional_is_day(175.0, tif, "AAPL")


def test_opg_buy_refuses_a_fractional_quantity():
    with pytest.raises(ValueError, match="time_in_force=DAY"):
        _client().submit_day_opg_buy("AAPL", 2.5)


def test_day_market_buy_accepts_a_fractional_quantity():
    c = _client()
    assert c.submit_day_market_buy("AAPL", 2.5) == "oid-1"
    req = c.tc.submit_order.call_args[0][0]
    assert req.qty == 2.5
    assert req.time_in_force == TimeInForce.DAY


def test_day_sell_accepts_a_fractional_quantity():
    c = _client()
    c.submit_day_sell("AAPL", 0.709973641)
    assert c.tc.submit_order.call_args[0][0].qty == 0.709973641


def test_whole_share_submission_is_unchanged():
    c = _client()
    c.submit_day_market_buy("AAPL", 175)
    req = c.tc.submit_order.call_args[0][0]
    assert req.qty == 175
    assert req.time_in_force == TimeInForce.DAY


def test_get_positions_preserves_a_fractional_quantity():
    tc = MagicMock()
    tc.get_all_positions.return_value = [
        MagicMock(symbol="AAPL", qty="2.5", avg_entry_price="100.0"),
        MagicMock(symbol="MSFT", qty="175", avg_entry_price="50.0"),
    ]
    got = AlpacaClient(trading_client=tc, paper=True).get_positions()
    assert got["AAPL"]["shares"] == 2.5
    assert got["MSFT"]["shares"] == 175
    assert isinstance(got["MSFT"]["shares"], int)   # unchanged for whole shares


def test_close_position_liquidates_without_computing_a_quantity():
    """The no-dust exit path: DELETE /v2/positions/{symbol} with no qty."""
    tc = MagicMock()
    tc.close_position.return_value = MagicMock(id="close-1")
    assert AlpacaClient(trading_client=tc, paper=True).close_position("AAPL") == "close-1"
    tc.close_position.assert_called_once_with("AAPL")


def test_paper_and_live_clients_report_their_endpoint():
    assert AlpacaClient(trading_client=MagicMock(), paper=True).paper is True
    assert AlpacaClient(trading_client=MagicMock(), paper=False).paper is False
