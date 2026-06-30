"""Tests for live.stop_loss.stop_loss_sweep."""

from datetime import date, timedelta
from unittest.mock import MagicMock

from alpaca.common.exceptions import APIError
from alpaca.trading.client import TradingClient

from sma.ingest.store import Store
from sma.live.alpaca_client import AlpacaClient
from sma.live.stop_loss import stop_loss_sweep
from sma.risk.rails import RiskRails


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
    assert "stop-loss price for AAPL is stale" in caplog.text
