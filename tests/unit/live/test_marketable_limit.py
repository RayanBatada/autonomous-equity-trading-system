"""Marketable-limit pricing, submission and the cancel-and-replace sweep.
All broker/data calls are MagicMocks; nothing touches the network."""

from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import pytest
from alpaca.trading.enums import OrderSide, TimeInForce

from sma.live.alpaca_client import (
    AlpacaClient,
    QuoteUnavailableError,
    RefPrice,
    WorkingOrder,
    marketable_limit_price,
)

ET = ZoneInfo("America/New_York")


def _client(bid=100.0, ask=100.10, trade=100.05):
    tc = MagicMock()
    data = MagicMock()
    data.get_stock_latest_quote.side_effect = lambda req: {
        req.symbol_or_symbols: SimpleNamespace(bid_price=bid, ask_price=ask, bid_size=1,
                                               ask_size=1, timestamp=None)}
    data.get_stock_latest_trade.side_effect = lambda req: {
        req.symbol_or_symbols: SimpleNamespace(price=trade)}
    tc.submit_order.side_effect = lambda req: SimpleNamespace(id=f"id-{req.symbol}-{req.qty}")
    return AlpacaClient(trading_client=tc, data_client=data), tc


def test_price_buy_rounds_up_sell_rounds_down():
    ref = RefPrice(bid=100.0, ask=100.10, source="quote")
    assert marketable_limit_price("BUY", ref, 10) == pytest.approx(100.21)   # 100.2001 -> up
    assert marketable_limit_price("SELL", ref, 10) == pytest.approx(99.90)
    penny = RefPrice(bid=0.5, ask=0.5, source="trade")
    assert marketable_limit_price("BUY", penny, 10) == pytest.approx(0.5005)
    with pytest.raises(ValueError):
        marketable_limit_price("HOLD", ref, 10)


def test_submit_buy_uses_ask_plus_offset_day_tif():
    c, tc = _client()
    res = c.submit_marketable_limit("AAPL", "buy", 5, offset_bps=10, client_order_id="x")
    req = tc.submit_order.call_args[0][0]
    assert req.side == OrderSide.BUY
    assert req.time_in_force == TimeInForce.DAY
    assert float(req.limit_price) == pytest.approx(100.21)
    assert req.qty == 5 and req.client_order_id == "x"
    assert res.limit_price == pytest.approx(100.21) and res.ref.source == "quote"


def test_notional_converts_at_limit_whole_shares_by_default():
    c, tc = _client()
    res = c.submit_marketable_limit("AAPL", "SELL", 1000.0, offset_bps=10, is_notional=True)
    assert res.qty == 10  # 1000 / 99.90 = 10.01 -> 10 whole shares
    res = c.submit_marketable_limit("AAPL", "BUY", 1000.0, offset_bps=10, is_notional=True,
                                    fractional=True, fractional_precision=4)
    assert res.qty == pytest.approx(9.9790)


def test_zero_qty_refused_and_fractional_non_day_refused():
    c, tc = _client()
    with pytest.raises(ValueError):
        c.submit_marketable_limit("AAPL", "BUY", 50.0, offset_bps=10, is_notional=True)
    with pytest.raises(ValueError):
        c.submit_marketable_limit("AAPL", "BUY", 1.5, offset_bps=10, tif=TimeInForce.GTC)
    tc.submit_order.assert_not_called()


def test_wide_or_empty_quote_falls_back_to_last_trade():
    c, _ = _client(bid=90.0, ask=110.0, trade=101.0)
    ref = c.reference_price("XLRE", max_spread_bps=50)
    assert (ref.bid, ref.ask, ref.source) == (101.0, 101.0, "trade")
    c, _ = _client(bid=0.0, ask=0.0, trade=0.0)
    with pytest.raises(QuoteUnavailableError):
        c.reference_price("XLRE")


def test_dash_ticker_translated_for_alpaca():
    c, tc = _client()
    c.submit_marketable_limit("BRK-B", "BUY", 1, offset_bps=10)
    assert tc.submit_order.call_args[0][0].symbol == "BRK.B"


# ---- sweep -----------------------------------------------------------------


class _Clock:
    def __init__(self):
        self.t = datetime(2026, 9, 28, 10, 35, tzinfo=ET)

    def now(self):
        return self.t

    def sleep(self, s):
        self.t += timedelta(seconds=s)


def _order(status, filled):
    return SimpleNamespace(status=status, filled_qty=filled)


def _sweep_client(states):
    """states: order_id -> list of (status, filled) returned on successive reads;
    the last entry repeats. cancel flips an open order to canceled."""
    c, tc = _client()
    reads = {k: list(v) for k, v in states.items()}
    canceled = set()

    def get(oid):
        seq = reads.get(oid, [("new", 0)])
        st, f = seq.pop(0) if len(seq) > 1 else seq[0]
        if oid in canceled and st not in ("filled",):
            st = "canceled"
        return _order(st, f)

    tc.get_order_by_id.side_effect = get
    tc.cancel_order_by_id.side_effect = lambda oid: canceled.add(oid)
    return c, tc


def _kw(clock, **over):
    kw = dict(session="midday", deadline=clock.t + timedelta(minutes=25), after_minutes=5,
              offset_bps=10, reprice_once=True, fallback="market",
              now_fn=clock.now, sleep_fn=clock.sleep, poll_s=30)
    kw.update(over)
    return kw


def test_sweep_all_filled_before_first_round_touches_nothing():
    clock = _Clock()
    c, tc = _sweep_client({"a": [("filled", 10)]})
    w = WorkingOrder(order_id="a", ticker="AAPL", side="BUY", qty=10)
    out = c.sweep_unfilled([w], **_kw(clock))
    assert [o.order_id for o in out] == ["a"] and w.done
    tc.cancel_order_by_id.assert_not_called()
    tc.submit_order.assert_not_called()


def test_sweep_reprices_remainder_once_then_market():
    clock = _Clock()
    c, tc = _sweep_client({"a": [("partially_filled", 4)]})
    replaced = []
    w = WorkingOrder(order_id="a", ticker="AAPL", side="BUY", qty=10)
    out = c.sweep_unfilled(
        [w], **_kw(clock), on_replace=lambda old, new, d: replaced.append((new.stage, new.qty)),
        coid_fn=lambda old, stage: f"coid-{stage}")
    # original cancelled after 5 min, remainder 6 repriced, still open -> market 6
    assert replaced == [("reprice", 6), ("market", 6)]
    reqs = [call[0][0] for call in tc.submit_order.call_args_list]
    assert reqs[0].limit_price is not None and reqs[0].client_order_id == "coid-reprice"
    assert getattr(reqs[1], "limit_price", None) is None
    assert reqs[1].client_order_id == "coid-market"
    assert [o.stage for o in out] == ["limit", "reprice", "market"]
    assert clock.t <= datetime(2026, 9, 28, 11, 0, tzinfo=ET)


def test_sweep_leave_does_not_cancel_the_repriced_limit():
    clock = _Clock()
    c, tc = _sweep_client({"a": [("new", 0)]})
    w = WorkingOrder(order_id="a", ticker="AAPL", side="SELL", qty=3)
    out = c.sweep_unfilled([w], **_kw(clock, fallback="leave"))
    assert [o.stage for o in out] == ["limit", "reprice"]
    assert tc.cancel_order_by_id.call_count == 1  # only the original


def test_sweep_no_reprice_goes_straight_to_market():
    clock = _Clock()
    c, tc = _sweep_client({"a": [("new", 0)]})
    w = WorkingOrder(order_id="a", ticker="AAPL", side="SELL", qty=3)
    out = c.sweep_unfilled([w], **_kw(clock, reprice_once=False))
    assert [o.stage for o in out] == ["limit", "market"]


def test_sweep_respects_deadline_before_after_minutes():
    clock = _Clock()
    start = clock.t
    c, tc = _sweep_client({"a": [("new", 0)]})
    w = WorkingOrder(order_id="a", ticker="AAPL", side="BUY", qty=3)
    c.sweep_unfilled([w], **_kw(clock, deadline=start + timedelta(minutes=2), fallback="leave"))
    assert clock.t <= start + timedelta(minutes=2, seconds=30)


def test_sweep_never_replaces_when_cancel_does_not_settle():
    clock = _Clock()
    c, tc = _sweep_client({"a": [("new", 0)]})
    tc.cancel_order_by_id.side_effect = lambda oid: None  # cancel acknowledged, never settles
    tc.get_order_by_id.side_effect = lambda oid: _order("pending_cancel", 0)
    w = WorkingOrder(order_id="a", ticker="AAPL", side="BUY", qty=3)
    out = c.sweep_unfilled([w], **_kw(clock))
    assert len(out) == 1
    tc.submit_order.assert_not_called()


def test_sweep_fill_during_cancel_is_not_replaced():
    clock = _Clock()
    c, tc = _sweep_client({"a": [("new", 0)]})
    tc.cancel_order_by_id.side_effect = RuntimeError("order already filled")
    reads = iter([("new", 0)] * 12 + [("filled", 3)] * 50)
    tc.get_order_by_id.side_effect = lambda oid: _order(*next(reads))
    w = WorkingOrder(order_id="a", ticker="AAPL", side="BUY", qty=3)
    out = c.sweep_unfilled([w], **_kw(clock))
    assert len(out) == 1 and w.filled_qty == 3
    tc.submit_order.assert_not_called()
