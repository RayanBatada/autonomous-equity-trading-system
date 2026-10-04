"""Tests for live.stop_loss.stop_loss_sweep."""

import uuid
from datetime import date, timedelta
from unittest.mock import MagicMock

from alpaca.common.exceptions import APIError
from alpaca.trading.client import TradingClient

from sma.ingest.store import Store
from sma.live.alpaca_client import AlpacaClient
from sma.live.stop_loss import stop_loss_sweep
from sma.risk.rails import RiskRails


def _seed_decide_order(
    store, *, asof: date, ticker: str, side: str = "BUY",
    alpaca_order_id: str = "decide-1", status: str = "submitted",
):
    """A still-open decide order (as it sits at 09:25, before that batch's 16:30
    reconcile) for the same name the sweep is about to exit."""
    rid = store.allocate_run_id()
    iid = str(uuid.uuid4())
    store.conn.execute("""
        INSERT INTO intended_orders
        (intended_order_id, asof_date, ticker, side, target_shares,
         target_weight, last_price, source, alpaca_order_id, status, run_id)
        VALUES (?, ?, ?, ?, 10, NULL, 200.0, 'decide', ?, ?, ?)
    """, [iid, asof, ticker, side, alpaca_order_id, status, rid])
    return iid


def _seed_prices(store, ticker: str, last_close: float, asof: date):
    rid = store.allocate_run_id()
    for offset in range(3):
        d = asof - timedelta(days=2 - offset)
        store.conn.execute(
            "INSERT INTO prices (ticker, date, open, high, low, close, "
            " adj_close, volume, source, run_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, 1000000, 'yfinance', ?)",
            [ticker, d, last_close, last_close, last_close,
             last_close, last_close, rid],
        )


def _seed_price_on(store, ticker: str, close: float, price_date: date):
    rid = store.allocate_run_id()
    store.conn.execute(
        "INSERT INTO prices (ticker, date, open, high, low, close, "
        " adj_close, volume, source, run_id) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, 1000000, 'yfinance', ?)",
        [ticker, price_date, close, close, close, close, close, rid],
    )


def _seed_price_with_adj(store, ticker: str, close: float, adj_close: float, price_date: date):
    """Like `_seed_price_on` but with a raw close that differs from adj_close
    (a dividend-paying name where adj_close has been discounted below the raw
    close)."""
    rid = store.allocate_run_id()
    store.conn.execute(
        "INSERT INTO prices (ticker, date, open, high, low, close, "
        " adj_close, volume, source, run_id) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, 1000000, 'yfinance', ?)",
        [ticker, price_date, close, close, close, close, adj_close, rid],
    )


def _alpaca_with_positions(positions_list):
    """positions_list: [(ticker, shares, cost_basis), ...]"""
    tc = MagicMock(spec=TradingClient)
    pos_objs = []
    for ticker, shares, cost in positions_list:
        p = MagicMock()
        p.symbol = ticker
        p.qty = str(shares)
        p.avg_entry_price = str(cost)
        pos_objs.append(p)
    tc.get_all_positions.return_value = pos_objs

    submitted = []

    def submit_order(req):
        oid = f"ord-{len(submitted)}"
        submitted.append(oid)
        ret = MagicMock()
        ret.id = oid
        return ret

    tc.submit_order.side_effect = submit_order
    return AlpacaClient(trading_client=tc), tc


def test_stop_loss_disabled_returns_immediately(tmp_path):
    """rails.stop_loss_pct=0 → zero rows written, zero Alpaca calls."""
    store = Store(path=str(tmp_path / "test.duckdb")).connect()
    alpaca, tc = _alpaca_with_positions([("AAPL", 25, 200.0)])

    result = stop_loss_sweep(
        asof=date(2026, 5, 1), store=store, alpaca=alpaca,
        rails=RiskRails(stop_loss_pct=0.0),
    )
    assert result.triggered == 0
    assert result.submitted == 0
    tc.submit_order.assert_not_called()
    rows = store.conn.execute("SELECT COUNT(*) FROM intended_orders").fetchone()
    assert rows[0] == 0


def test_stop_loss_enabled_triggers_on_8pct_drop(tmp_path):
    store = Store(path=str(tmp_path / "test.duckdb")).connect()
    asof = date(2026, 5, 1)
    _seed_prices(store, "AAPL", last_close=180.0, asof=asof)   # cost 200, now 180 = -10%
    alpaca, tc = _alpaca_with_positions([("AAPL", 25, 200.0)])

    result = stop_loss_sweep(
        asof=asof, store=store, alpaca=alpaca,
        rails=RiskRails(stop_loss_pct=0.08),
    )
    assert result.triggered == 1
    assert result.submitted == 1
    rows = store.conn.execute(
        "SELECT ticker, side, target_shares, last_price, source, status "
        "FROM intended_orders"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0] == ("AAPL", "SELL", 25, 180.0, "stop-loss", "submitted")


def test_stop_loss_uses_raw_close_not_adj_close_for_the_basis_compare(tmp_path):
    """cost_basis is Alpaca's RAW (unadjusted) avg_entry_price. The fixed
    stop-loss compare must use the raw close too, or a dividend-discounted
    adj_close that sits below the raw close mis-fires the stop even though
    the real raw-dollar loss hasn't crossed the threshold.

    cost_basis=100, raw close=93 (-7%, under the 8% stop) but adj_close=91
    (-9% vs the same raw cost_basis) — the old adj_close-based compare would
    wrongly fire; the raw-close-based compare correctly does not."""
    store = Store(path=str(tmp_path / "test.duckdb")).connect()
    asof = date(2026, 5, 1)
    _seed_price_with_adj(store, "KO", close=93.0, adj_close=91.0, price_date=asof)
    alpaca, tc = _alpaca_with_positions([("KO", 25, 100.0)])

    result = stop_loss_sweep(
        asof=asof, store=store, alpaca=alpaca,
        rails=RiskRails(stop_loss_pct=0.08),
    )
    assert result.triggered == 0
    assert result.submitted == 0
    tc.submit_order.assert_not_called()


def test_stop_loss_does_not_trigger_when_position_above_threshold(tmp_path):
    store = Store(path=str(tmp_path / "test.duckdb")).connect()
    asof = date(2026, 5, 1)
    _seed_prices(store, "AAPL", last_close=185.0, asof=asof)   # cost 200, now 185 = -7.5%
    alpaca, tc = _alpaca_with_positions([("AAPL", 25, 200.0)])

    result = stop_loss_sweep(
        asof=asof, store=store, alpaca=alpaca,
        rails=RiskRails(stop_loss_pct=0.08),
    )
    assert result.triggered == 0
    assert result.submitted == 0
    tc.submit_order.assert_not_called()


def test_stop_loss_per_position_isolation_on_alpaca_error(tmp_path):
    """One Alpaca error doesn't kill the batch; others still try."""
    store = Store(path=str(tmp_path / "test.duckdb")).connect()
    asof = date(2026, 5, 1)
    _seed_prices(store, "AAPL", last_close=180.0, asof=asof)
    _seed_prices(store, "MSFT", last_close=350.0, asof=asof)
    alpaca, tc = _alpaca_with_positions([
        ("AAPL", 25, 200.0),   # 10% loss
        ("MSFT", 10, 400.0),   # 12.5% loss
    ])

    call_count = [0]

    def maybe_fail(req):
        call_count[0] += 1
        if call_count[0] == 1:
            raise APIError("test failure")
        ret = MagicMock()
        ret.id = f"ord-{call_count[0]}"
        return ret

    tc.submit_order.side_effect = maybe_fail

    result = stop_loss_sweep(
        asof=asof, store=store, alpaca=alpaca,
        rails=RiskRails(stop_loss_pct=0.08),
    )
    assert result.triggered == 2
    assert result.submitted == 1
    assert result.failed == 1


def test_stop_loss_no_positions_returns_zero(tmp_path):
    store = Store(path=str(tmp_path / "test.duckdb")).connect()
    alpaca, tc = _alpaca_with_positions([])

    result = stop_loss_sweep(
        asof=date(2026, 5, 1), store=store, alpaca=alpaca,
        rails=RiskRails(stop_loss_pct=0.08),
    )
    assert result.triggered == 0


def test_stop_loss_skips_ticker_without_recent_price(tmp_path):
    """Position held but no recent prices in DB → log warning, skip."""
    store = Store(path=str(tmp_path / "test.duckdb")).connect()
    alpaca, tc = _alpaca_with_positions([("AAPL", 25, 200.0)])

    result = stop_loss_sweep(
        asof=date(2026, 5, 1), store=store, alpaca=alpaca,
        rails=RiskRails(stop_loss_pct=0.08),
    )
    assert result.triggered == 0
    assert result.submitted == 0


def test_stop_loss_uses_friday_close_on_tuesday_after_monday_holiday(tmp_path):
    """A Friday close is 4 calendar days old after a Monday holiday; still check."""
    store = Store(path=str(tmp_path / "test.duckdb")).connect()
    asof = date(2026, 5, 26)  # Tuesday after Memorial Day.
    _seed_price_on(store, "AAPL", close=180.0, price_date=date(2026, 5, 22))
    alpaca, tc = _alpaca_with_positions([("AAPL", 25, 200.0)])

    result = stop_loss_sweep(
        asof=asof, store=store, alpaca=alpaca,
        rails=RiskRails(stop_loss_pct=0.08),
    )

    assert result.triggered == 1
    assert result.submitted == 1
    tc.submit_order.assert_called_once()


def test_stop_loss_warns_but_checks_price_older_than_five_calendar_days(tmp_path, caplog):
    store = Store(path=str(tmp_path / "test.duckdb")).connect()
    asof = date(2026, 5, 8)
    _seed_price_on(store, "AAPL", close=180.0, price_date=date(2026, 5, 2))
    alpaca, tc = _alpaca_with_positions([("AAPL", 25, 200.0)])

    result = stop_loss_sweep(
        asof=asof, store=store, alpaca=alpaca,
        rails=RiskRails(stop_loss_pct=0.08),
    )

    assert result.triggered == 1
    assert result.submitted == 1
    assert "price-exit price for AAPL is stale" in caplog.text


# --- price exits: take-profit + trailing-stop (2026-07-01) -------------------
def test_all_price_exits_disabled_returns_immediately(tmp_path):
    """All three knobs at 0 → NOOP (no positions fetched, no submits)."""
    store = Store(path=str(tmp_path / "test.duckdb")).connect()
    alpaca, tc = _alpaca_with_positions([("AAPL", 25, 200.0)])
    result = stop_loss_sweep(
        asof=date(2026, 5, 1), store=store, alpaca=alpaca,
        rails=RiskRails(stop_loss_pct=0.0, trailing_stop_pct=0.0, take_profit_pct=0.0),
    )
    assert (result.triggered, result.submitted) == (0, 0)
    tc.submit_order.assert_not_called()


def test_take_profit_fires_on_gain(tmp_path):
    store = Store(path=str(tmp_path / "test.duckdb")).connect()
    asof = date(2026, 5, 1)
    _seed_prices(store, "AAPL", last_close=260.0, asof=asof)  # cost 200 → +30%
    alpaca, tc = _alpaca_with_positions([("AAPL", 25, 200.0)])
    result = stop_loss_sweep(
        asof=asof, store=store, alpaca=alpaca,
        rails=RiskRails(stop_loss_pct=0.0, take_profit_pct=0.20),
    )
    assert result.triggered == 1
    assert result.submitted == 1
    rows = store.conn.execute(
        "SELECT ticker, side, target_shares FROM intended_orders"
    ).fetchall()
    assert rows == [("AAPL", "SELL", 25)]


def test_take_profit_uses_raw_close_not_adj_close_for_the_basis_compare(tmp_path):
    """cost_basis is Alpaca's RAW (unadjusted) avg_entry_price. The take-profit
    compare must use the raw close too, or a dividend-discounted adj_close
    that sits below the raw close can mask a real take-profit trigger.

    cost_basis=100, raw close=121 (+21%, above the 20% take-profit threshold)
    but adj_close=119 (+19%, below threshold) — the old adj_close-based
    compare would wrongly NOT fire; the raw-close-based compare correctly
    fires."""
    store = Store(path=str(tmp_path / "test.duckdb")).connect()
    asof = date(2026, 5, 1)
    _seed_price_with_adj(store, "KO", close=121.0, adj_close=119.0, price_date=asof)
    alpaca, tc = _alpaca_with_positions([("KO", 25, 100.0)])

    result = stop_loss_sweep(
        asof=asof, store=store, alpaca=alpaca,
        rails=RiskRails(stop_loss_pct=0.0, take_profit_pct=0.20),
    )
    assert result.triggered == 1
    assert result.submitted == 1


def test_take_profit_does_not_fire_below_threshold(tmp_path):
    store = Store(path=str(tmp_path / "test.duckdb")).connect()
    asof = date(2026, 5, 1)
    _seed_prices(store, "AAPL", last_close=215.0, asof=asof)  # +7.5% < +20%
    alpaca, tc = _alpaca_with_positions([("AAPL", 25, 200.0)])
    result = stop_loss_sweep(
        asof=asof, store=store, alpaca=alpaca,
        rails=RiskRails(stop_loss_pct=0.0, take_profit_pct=0.20),
    )
    assert result.triggered == 0
    tc.submit_order.assert_not_called()


def test_trailing_stop_peak_uses_raw_close_not_adj_close_for_the_basis_compare(tmp_path):
    """The trailing-stop peak floor is the RAW cost_basis (see
    _peak_prices_from_store). Before this fix, the peak's max-since-entry was
    computed from adj_close while current_price was also adj_close — mostly
    consistent, EXCEPT when the raw floor wins (adj_close has never reached
    the raw cost_basis since entry) while the raw close series HAS reached
    it: the peak then mixes a raw floor against an adj_close high-water
    mark, versus an adj_close current price — a mixed-basis compare that can
    mis-trigger relative to the true raw-dollar move.

    cost_basis=200 (raw). Since entry: adj_close peaks at 195 (never clears
    the 200 floor, so the OLD adj-based peak = 200, the raw floor) while raw
    close peaks at 210 (clears the floor, so the FIXED raw-based peak = 210).
    Today: close=193, adj_close=178.
      OLD (peak=200 vs adj current=178): drop = 11.0% -> mis-triggers.
      FIXED (peak=210 vs raw current=193): drop = 8.1% -> correctly does not.
    """
    store = Store(path=str(tmp_path / "test.duckdb")).connect()
    asof = date(2026, 5, 8)
    rid = store.allocate_run_id()
    store.conn.execute(
        "INSERT INTO paper_fills (alpaca_order_id, asof_date, ticker, side, "
        " filled_shares, fill_price, status, submitted_at, filled_at, run_id) "
        "VALUES (?, DATE '2026-05-01', 'KO', 'BUY', 25, 200.0, 'filled', "
        " TIMESTAMP '2026-05-01 15:00:00', TIMESTAMP '2026-05-01 15:00:00', ?)",
        ["ord-buy-1", rid],
    )
    for d, close, adj in [
        (date(2026, 5, 1), 200.0, 190.0),
        (date(2026, 5, 5), 210.0, 195.0),
        (date(2026, 5, 8), 193.0, 178.0),
    ]:
        _seed_price_with_adj(store, "KO", close=close, adj_close=adj, price_date=d)
    alpaca, tc = _alpaca_with_positions([("KO", 25, 200.0)])

    result = stop_loss_sweep(
        asof=asof, store=store, alpaca=alpaca,
        rails=RiskRails(stop_loss_pct=0.0, trailing_stop_pct=0.10),
    )
    assert result.triggered == 0
    tc.submit_order.assert_not_called()


def test_trailing_stop_fires_when_below_peak(tmp_path):
    """Position ran to a 300 peak (seeded across the price history) then fell to
    250 (−16.7%) → trailing stop (10%) fires. Peak is recomputed from stored
    closes since the most-recent BUY fill."""
    store = Store(path=str(tmp_path / "test.duckdb")).connect()
    asof = date(2026, 5, 8)
    rid = store.allocate_run_id()
    # Entry BUY fill on 2026-05-01 (so _latest_buy_dates finds an entry date).
    store.conn.execute(
        "INSERT INTO paper_fills (alpaca_order_id, asof_date, ticker, side, "
        " filled_shares, fill_price, status, submitted_at, filled_at, run_id) "
        "VALUES (?, DATE '2026-05-01', 'AAPL', 'BUY', 25, 200.0, 'filled', "
        " TIMESTAMP '2026-05-01 15:00:00', TIMESTAMP '2026-05-01 15:00:00', ?)",
        ["ord-buy-1", rid],
    )
    # Price history since entry: peak 300 on 5/5, current 250 on 5/8.
    for d, px in [
        (date(2026, 5, 1), 200.0), (date(2026, 5, 5), 300.0), (date(2026, 5, 8), 250.0),
    ]:
        store.conn.execute(
            "INSERT INTO prices (ticker, date, open, high, low, close, adj_close, "
            " volume, source, run_id) VALUES ('AAPL', ?, ?, ?, ?, ?, ?, 1000000, 'yfinance', ?)",
            [d, px, px, px, px, px, rid],
        )
    alpaca, tc = _alpaca_with_positions([("AAPL", 25, 200.0)])
    result = stop_loss_sweep(
        asof=asof, store=store, alpaca=alpaca,
        rails=RiskRails(stop_loss_pct=0.0, trailing_stop_pct=0.10),
    )
    assert result.triggered == 1
    assert result.submitted == 1


def test_trailing_stop_does_not_fire_near_peak(tmp_path):
    """Same setup but current price 290 is only 3.3% off the 300 peak → no fire."""
    store = Store(path=str(tmp_path / "test.duckdb")).connect()
    asof = date(2026, 5, 8)
    rid = store.allocate_run_id()
    store.conn.execute(
        "INSERT INTO paper_fills (alpaca_order_id, asof_date, ticker, side, "
        " filled_shares, fill_price, status, submitted_at, filled_at, run_id) "
        "VALUES (?, DATE '2026-05-01', 'AAPL', 'BUY', 25, 200.0, 'filled', "
        " TIMESTAMP '2026-05-01 15:00:00', TIMESTAMP '2026-05-01 15:00:00', ?)",
        ["ord-buy-1", rid],
    )
    for d, px in [
        (date(2026, 5, 1), 200.0), (date(2026, 5, 5), 300.0), (date(2026, 5, 8), 290.0),
    ]:
        store.conn.execute(
            "INSERT INTO prices (ticker, date, open, high, low, close, adj_close, "
            " volume, source, run_id) VALUES ('AAPL', ?, ?, ?, ?, ?, ?, 1000000, 'yfinance', ?)",
            [d, px, px, px, px, px, rid],
        )
    alpaca, tc = _alpaca_with_positions([("AAPL", 25, 200.0)])
    result = stop_loss_sweep(
        asof=asof, store=store, alpaca=alpaca,
        rails=RiskRails(stop_loss_pct=0.0, trailing_stop_pct=0.10),
    )
    assert result.triggered == 0
    tc.submit_order.assert_not_called()


def test_trailing_stop_fires_after_top_up_using_full_holding_peak(tmp_path):
    """Bug 2: after averaging up, the trailing peak must span the WHOLE current
    holding (earliest buy), matching the simulator — not restart at the top-up
    buy. Entry 5/1, peak close 300 on 5/5 (BEFORE the top-up), top-up 5/6,
    current 250 on 5/8: 250 is >16% below the 300 peak → fires. With the OLD
    most-recent-buy window the peak would collapse to ~260 (post-top-up) and the
    stop would MISS the exit the sim fires — the live-vs-sim drift being fixed."""
    store = Store(path=str(tmp_path / "test.duckdb")).connect()
    asof = date(2026, 5, 8)
    rid = store.allocate_run_id()
    # Entry BUY on 5/1, then a top-up BUY on 5/6 (averaging up into a winner).
    for oid, d, sh, px in [
        ("ord-buy-1", "2026-05-01", 25, 200.0),
        ("ord-buy-2", "2026-05-06", 10, 260.0),
    ]:
        store.conn.execute(
            "INSERT INTO paper_fills (alpaca_order_id, asof_date, ticker, side, "
            " filled_shares, fill_price, status, submitted_at, filled_at, run_id) "
            f"VALUES (?, DATE '{d}', 'AAPL', 'BUY', ?, ?, 'filled', "
            f" TIMESTAMP '{d} 15:00:00', TIMESTAMP '{d} 15:00:00', ?)",
            [oid, sh, px, rid],
        )
    for d, px in [
        (date(2026, 5, 1), 200.0), (date(2026, 5, 5), 300.0),  # peak BEFORE the top-up
        (date(2026, 5, 6), 260.0), (date(2026, 5, 8), 250.0),
    ]:
        store.conn.execute(
            "INSERT INTO prices (ticker, date, open, high, low, close, adj_close, "
            " volume, source, run_id) VALUES ('AAPL', ?, ?, ?, ?, ?, ?, 1000000, 'yfinance', ?)",
            [d, px, px, px, px, px, rid],
        )
    alpaca, tc = _alpaca_with_positions([("AAPL", 35, 217.14)])
    result = stop_loss_sweep(
        asof=asof, store=store, alpaca=alpaca,
        rails=RiskRails(stop_loss_pct=0.0, trailing_stop_pct=0.10),
    )
    assert result.triggered == 1
    assert result.submitted == 1


def test_sweep_cancels_pending_decide_orders_for_stopped_ticker(tmp_path):
    """When a stop fires for X, the sweep cancels last night's still-open decide
    order(s) for X — the pending OPG BUY would otherwise re-open the position at
    the same 09:30 open the sweep just sold into (a no-op round-trip paying double
    slippage; the sim prevents it via _exited_today). Canceling also prevents a
    pending decide SELL from double-submitting against the sweep's full-exit sell
    (oversell). The canceled row is marked so reconcile's buy-miss detector won't
    read a deliberate cancel as a missed execution (#9/#10, review 2026-07-04)."""
    store = Store(path=str(tmp_path / "test.duckdb")).connect()
    asof = date(2026, 5, 1)
    _seed_prices(store, "AAPL", last_close=180.0, asof=asof)  # cost 200 → -10% stop
    # last night's decide BUY for AAPL, still open at 09:25 (D-1 batch unreconciled)
    _seed_decide_order(store, asof=date(2026, 4, 30), ticker="AAPL", side="BUY",
                       alpaca_order_id="decide-buy-1")
    alpaca, tc = _alpaca_with_positions([("AAPL", 25, 200.0)])

    result = stop_loss_sweep(
        asof=asof, store=store, alpaca=alpaca, rails=RiskRails(stop_loss_pct=0.08),
    )
    assert result.triggered == 1
    assert result.submitted == 1
    tc.cancel_order_by_id.assert_called_once_with("decide-buy-1")
    status = store.conn.execute(
        "SELECT status FROM intended_orders WHERE alpaca_order_id='decide-buy-1'"
    ).fetchone()[0]
    assert status == "canceled_by_stop"


def test_sweep_fails_closed_when_pending_decide_cancel_cannot_be_confirmed(tmp_path):
    """If canceling a still-open decide order raises AND the order is still live
    (transient 503, not terminal), the sweep must NOT submit its full-exit stop
    sell — a live opposing decide order + the stop sell would oversell/short a
    long-only book. Fail closed: skip the sell, count failed, page (review
    2026-07-04, adversarial)."""
    store = Store(path=str(tmp_path / "test.duckdb")).connect()
    asof = date(2026, 5, 1)
    _seed_prices(store, "AAPL", last_close=180.0, asof=asof)  # -10% stop
    _seed_decide_order(store, asof=date(2026, 4, 30), ticker="AAPL", side="SELL",
                       alpaca_order_id="decide-sell-1")
    alpaca, tc = _alpaca_with_positions([("AAPL", 25, 200.0)])
    tc.cancel_order_by_id.side_effect = RuntimeError("503 transient")
    live = MagicMock()
    live.status = "accepted"  # re-fetch: order is still LIVE
    tc.get_order_by_id.return_value = live

    result = stop_loss_sweep(
        asof=asof, store=store, alpaca=alpaca, rails=RiskRails(stop_loss_pct=0.08),
    )
    assert result.triggered == 1
    assert result.submitted == 0       # did NOT sell (fail closed)
    assert result.failed == 1
    tc.submit_order.assert_not_called()  # no stop sell crossed the same open


def test_sweep_proceeds_when_pending_decide_cancel_confirms_terminal(tmp_path):
    """A cancel that raises but whose order is confirmed already-terminal (no
    fill) is benign — the opposing order is gone, so the sweep still exits."""
    store = Store(path=str(tmp_path / "test.duckdb")).connect()
    asof = date(2026, 5, 1)
    _seed_prices(store, "AAPL", last_close=180.0, asof=asof)
    _seed_decide_order(store, asof=date(2026, 4, 30), ticker="AAPL", side="BUY",
                       alpaca_order_id="decide-buy-1")
    alpaca, tc = _alpaca_with_positions([("AAPL", 25, 200.0)])
    tc.cancel_order_by_id.side_effect = RuntimeError("order already canceled")
    term = MagicMock()
    term.status = "canceled"  # re-fetch: confirmed terminal...
    term.filled_qty = "0"     # ...with no fill → benign
    tc.get_order_by_id.return_value = term

    result = stop_loss_sweep(
        asof=asof, store=store, alpaca=alpaca, rails=RiskRails(stop_loss_pct=0.08),
    )
    assert result.triggered == 1
    assert result.submitted == 1  # proceeded to exit


def test_sweep_recovers_and_cancels_null_id_crash_after_accept_decide_order(tmp_path):
    """A crash-after-accept decide order (status='submitted', alpaca_order_id=NULL,
    yet LIVE at the broker) must NOT be silently ignored: the query used to filter
    alpaca_order_id IS NOT NULL, so the sweep saw nothing, sold, and the live buy
    re-opened the exited position at the open with no page. The sweep must recover
    the broker id via the deterministic coid and cancel it (re-review 2026-07-04)."""
    from sma.live.orders import client_order_id

    store = Store(path=str(tmp_path / "test.duckdb")).connect()
    asof = date(2026, 5, 1)
    _seed_prices(store, "AAPL", last_close=180.0, asof=asof)  # -10% stop
    _seed_decide_order(store, asof=date(2026, 4, 30), ticker="AAPL", side="BUY",
                       alpaca_order_id=None)  # crash-after-accept: id never recorded
    alpaca, tc = _alpaca_with_positions([("AAPL", 25, 200.0)])
    coid = client_order_id(date(2026, 4, 30), "AAPL", "BUY")
    live = MagicMock()
    live.id = "recovered-1"
    live.status = "accepted"  # broker holds it, still live
    tc.get_order_by_client_id.side_effect = lambda c: live if c == coid else None

    result = stop_loss_sweep(
        asof=asof, store=store, alpaca=alpaca, rails=RiskRails(stop_loss_pct=0.08),
    )
    assert result.triggered == 1
    assert result.submitted == 1  # recovered + canceled → safe to exit
    tc.cancel_order_by_id.assert_called_once_with("recovered-1")
    row = store.conn.execute(
        "SELECT status, alpaca_order_id FROM intended_orders WHERE source='decide'"
    ).fetchone()
    assert row == ("canceled_by_stop", "recovered-1")  # marked + id backfilled


def test_order_confirmed_gone_rejects_a_partially_filled_cancel(tmp_path):
    """_order_confirmed_gone must honor its 'WITHOUT a new fill' contract: a
    canceled order that partially filled moved the book, so it is NOT safe to
    ignore (re-review 2026-07-04, LOW)."""
    from sma.live.stop_loss import _order_confirmed_gone

    _, tc = _alpaca_with_positions([])
    o = MagicMock()
    o.status = "canceled"
    o.filled_qty = "3"  # partial fill before cancel
    tc.get_order_by_id.return_value = o
    assert _order_confirmed_gone(AlpacaClient(trading_client=tc), "x") is False

    o.filled_qty = "0"
    assert _order_confirmed_gone(AlpacaClient(trading_client=tc), "x") is True


def test_sweep_does_not_cancel_when_no_stop_fires(tmp_path):
    """A pending decide BUY is left untouched when the position does NOT trip a
    stop — cancellation only happens for names the sweep actually exits."""
    store = Store(path=str(tmp_path / "test.duckdb")).connect()
    asof = date(2026, 5, 1)
    _seed_prices(store, "AAPL", last_close=195.0, asof=asof)  # -2.5%, no stop
    _seed_decide_order(store, asof=date(2026, 4, 30), ticker="AAPL", side="BUY",
                       alpaca_order_id="decide-buy-1")
    alpaca, tc = _alpaca_with_positions([("AAPL", 25, 200.0)])

    result = stop_loss_sweep(
        asof=asof, store=store, alpaca=alpaca, rails=RiskRails(stop_loss_pct=0.08),
    )
    assert result.triggered == 0
    tc.cancel_order_by_id.assert_not_called()
    status = store.conn.execute(
        "SELECT status FROM intended_orders WHERE alpaca_order_id='decide-buy-1'"
    ).fetchone()[0]
    assert status == "submitted"


def test_trailing_peak_resets_after_full_close_and_reopen(tmp_path):
    """Complement to the top-up case: a full close then a re-buy starts a NEW
    holding, so the peak window begins at the REOPEN (not the original entry).
    Entry 5/1 → SELL-to-flat 5/3 → re-buy 5/6; the pre-close 300 peak on 5/2 is
    NOT in the current holding's window, so the current 250 (only ~3.8% below the
    post-reopen 260 high) does not fire."""
    store = Store(path=str(tmp_path / "test.duckdb")).connect()
    asof = date(2026, 5, 8)
    rid = store.allocate_run_id()
    for oid, d, side, sh, px in [
        ("ord-buy-1", "2026-05-01", "BUY", 25, 200.0),
        ("ord-sell-1", "2026-05-03", "SELL", 25, 300.0),  # fully closed → flat
        ("ord-buy-2", "2026-05-06", "BUY", 10, 260.0),    # reopen
    ]:
        store.conn.execute(
            "INSERT INTO paper_fills (alpaca_order_id, asof_date, ticker, side, "
            " filled_shares, fill_price, status, submitted_at, filled_at, run_id) "
            f"VALUES (?, DATE '{d}', 'AAPL', ?, ?, ?, 'filled', "
            f" TIMESTAMP '{d} 15:00:00', TIMESTAMP '{d} 15:00:00', ?)",
            [oid, side, sh, px, rid],
        )
    for d, px in [
        (date(2026, 5, 1), 200.0), (date(2026, 5, 2), 300.0),  # peak in the OLD holding
        (date(2026, 5, 6), 260.0), (date(2026, 5, 8), 250.0),
    ]:
        store.conn.execute(
            "INSERT INTO prices (ticker, date, open, high, low, close, adj_close, "
            " volume, source, run_id) VALUES ('AAPL', ?, ?, ?, ?, ?, ?, 1000000, 'yfinance', ?)",
            [d, px, px, px, px, px, rid],
        )
    alpaca, tc = _alpaca_with_positions([("AAPL", 10, 260.0)])
    result = stop_loss_sweep(
        asof=asof, store=store, alpaca=alpaca,
        rails=RiskRails(stop_loss_pct=0.0, trailing_stop_pct=0.10),
    )
    assert result.triggered == 0
    tc.submit_order.assert_not_called()
