"""_submit_with_audit_trail must be idempotent across a crash between Alpaca
accepting an order and the DB recording its id.

The window: Alpaca accepts the order (returns an id) → the process dies BEFORE
the `UPDATE ... SET alpaca_order_id` → the row is left with alpaca_order_id=NULL.
The old null-id path DELETEd + resubmitted → a DUPLICATE live order. The fix uses
a deterministic client_order_id and recovers the existing broker order instead of
resubmitting (2026-06-05 audit, CRITICAL).
"""

import uuid
from datetime import date
from unittest.mock import MagicMock

from sma.ingest.store import Store
from sma.live.decide import _submit_with_audit_trail
from sma.live.orders import Order
from sma.live.stop_loss import _submit_stop_sell


def _store(tmp_path) -> Store:
    s = Store(path=str(tmp_path / "t.duckdb")).connect()
    s.allocate_run_id()
    return s


def _order(ticker="AAPL", side="SELL", shares=10) -> Order:
    return Order(ticker=ticker, side=side, shares=shares, type="DAY", last_price=100.0)


def _insert_null_id_row(store, asof, order, run_id=1):
    """Simulate a prior attempt that crashed after Alpaca accepted but before the
    id was written: a row exists with alpaca_order_id = NULL."""
    store.conn.execute(
        "INSERT INTO intended_orders (intended_order_id, asof_date, ticker, side, "
        "target_shares, target_weight, last_price, source, status, run_id) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, 'decide', 'submitted', ?)",
        [str(uuid.uuid4()), asof, order.ticker, order.side, order.shares, 0.05,
         order.last_price, run_id],
    )


def _insert_stop_loss_null_id_row(store, asof, ticker="AAPL", shares=10, run_id=1):
    store.conn.execute(
        "INSERT INTO intended_orders (intended_order_id, asof_date, ticker, side, "
        "target_shares, target_weight, last_price, source, status, run_id) "
        "VALUES (?, ?, ?, 'SELL', ?, NULL, 100.0, 'stop-loss', 'submitted', ?)",
        [str(uuid.uuid4()), asof, ticker, shares, run_id],
    )


def _aid(store, asof, ticker):
    return store.conn.execute(
        "SELECT alpaca_order_id FROM intended_orders "
        "WHERE asof_date=? AND ticker=? AND source='decide'",
        [asof, ticker],
    ).fetchone()[0]


def _stop_loss_aid(store, asof, ticker):
    return store.conn.execute(
        "SELECT alpaca_order_id FROM intended_orders "
        "WHERE asof_date=? AND ticker=? AND source='stop-loss'",
        [asof, ticker],
    ).fetchone()[0]


def test_fresh_sell_submit_passes_deterministic_client_order_id(tmp_path):
    store = _store(tmp_path)
    asof = date(2026, 6, 5)
    order = _order(ticker="TSLA", side="SELL", shares=2)
    alpaca = MagicMock()
    alpaca.submit_day_sell.return_value = "AID-FRESH"

    result = _submit_with_audit_trail(
        order, asof=asof, store=store, alpaca=alpaca, run_id=1, decisions=[]
    )

    assert result == "submitted"
    _, kwargs = alpaca.submit_day_sell.call_args
    assert kwargs.get("client_order_id") == "sma-2026-06-05-TSLA-SELL"
    assert _aid(store, asof, "TSLA") == "AID-FRESH"


def test_recovers_order_by_client_order_id_instead_of_double_submitting(tmp_path):
    store = _store(tmp_path)
    asof = date(2026, 6, 5)
    order = _order()
    _insert_null_id_row(store, asof, order)  # crash-after-accept

    alpaca = MagicMock()
    alpaca.get_order_by_client_order_id.return_value = ("ALPACA-RECOVERED", "accepted")
    alpaca.submit_day_sell.side_effect = AssertionError("must NOT resubmit")
    alpaca.submit_day_opg_buy.side_effect = AssertionError("must NOT resubmit")
    alpaca.submit_day_market_buy.side_effect = AssertionError("must NOT resubmit")

    result = _submit_with_audit_trail(
        order, asof=asof, store=store, alpaca=alpaca, run_id=2, decisions=[]
    )

    assert result == "submitted"
    alpaca.get_order_by_client_order_id.assert_called_once()
    assert _aid(store, asof, "AAPL") == "ALPACA-RECOVERED"


def test_null_id_resubmits_when_no_order_found_at_broker(tmp_path):
    """If the prior attempt truly failed before Alpaca accepted (broker confirms
    no order), resubmitting is safe — and uses the deterministic id."""
    store = _store(tmp_path)
    asof = date(2026, 6, 5)
    order = _order(ticker="MSFT")
    _insert_null_id_row(store, asof, order)

    alpaca = MagicMock()
    alpaca.get_order_by_client_order_id.return_value = None
    alpaca.submit_day_sell.return_value = "AID-NEW"

    result = _submit_with_audit_trail(
        order, asof=asof, store=store, alpaca=alpaca, run_id=2, decisions=[]
    )

    assert result == "submitted"
    alpaca.submit_day_sell.assert_called_once()
    _, kwargs = alpaca.submit_day_sell.call_args
    assert kwargs.get("client_order_id") == "sma-2026-06-05-MSFT-SELL"
    assert _aid(store, asof, "MSFT") == "AID-NEW"


def test_terminal_prior_order_is_not_resubmitted(tmp_path):
    """If the prior order under this coid is dead (rejected/canceled), do NOT
    resubmit today — the coid is spent and a fresh id would break idempotency."""
    store = _store(tmp_path)
    asof = date(2026, 6, 5)
    order = _order(ticker="NFLX")
    _insert_null_id_row(store, asof, order)

    alpaca = MagicMock()
    alpaca.get_order_by_client_order_id.return_value = ("OLD-REJECTED", "rejected")
    alpaca.submit_day_sell.side_effect = AssertionError("must NOT resubmit a dead coid")

    result = _submit_with_audit_trail(
        order, asof=asof, store=store, alpaca=alpaca, run_id=2, decisions=[]
    )

    assert result == "failed"
    assert _aid(store, asof, "NFLX") is None  # not adopted, not resubmitted


def test_recovery_lookup_error_fails_closed_no_resubmit(tmp_path):
    """A lookup error must NOT be treated as 'no order' — fail closed (no
    resubmit) so a transient API error can't double-submit a live order."""
    store = _store(tmp_path)
    asof = date(2026, 6, 5)
    order = _order(ticker="AMZN")
    _insert_null_id_row(store, asof, order)

    alpaca = MagicMock()
    alpaca.get_order_by_client_order_id.side_effect = RuntimeError("alpaca 503")
    alpaca.submit_day_sell.side_effect = AssertionError("must NOT resubmit on lookup error")

    result = _submit_with_audit_trail(
        order, asof=asof, store=store, alpaca=alpaca, run_id=2, decisions=[]
    )

    assert result == "failed"
    assert _aid(store, asof, "AMZN") is None


def test_buy_uses_day_market_not_opg(tmp_path, monkeypatch):
    """Buys must use a DAY MARKET order, never OPG. Opening-auction-only (OPG)
    orders barely fill in Alpaca's paper engine — the 2026-06-08 batch EXPIRED
    with ~0 fills, so the bot sold but couldn't buy (stuck at 82% cash)."""
    import sma.live.decide as dec

    # Force the (old) OPG-window branch to be 'open' so the old code would pick OPG.
    monkeypatch.setattr(dec, "_opg_window_open", lambda *a, **k: True)
    store = _store(tmp_path)
    asof = date(2026, 6, 5)
    order = _order(ticker="NVDA", side="BUY", shares=3)
    alpaca = MagicMock()
    alpaca.submit_day_market_buy.return_value = "AID-MKT"
    alpaca.submit_day_opg_buy.return_value = "AID-OPG"

    result = _submit_with_audit_trail(
        order, asof=asof, store=store, alpaca=alpaca, run_id=1, decisions=[]
    )

    assert result == "submitted"
    alpaca.submit_day_opg_buy.assert_not_called()
    alpaca.submit_day_market_buy.assert_called_once()
    assert _aid(store, asof, "NVDA") == "AID-MKT"


def test_stop_loss_fresh_submit_passes_distinct_deterministic_client_order_id(tmp_path):
    store = _store(tmp_path)
    asof = date(2026, 6, 5)
    alpaca = MagicMock()
    alpaca.submit_market_sell.return_value = "AID-STOP"

    assert _submit_stop_sell(
        ticker="TSLA",
        shares=2,
        last_price=90.0,
        asof=asof,
        store=store,
        alpaca=alpaca,
        run_id=1,
        reason="triggered",
    )

    _, kwargs = alpaca.submit_market_sell.call_args
    assert kwargs.get("client_order_id") == "sma-2026-06-05-TSLA-stop-loss-SELL"
    assert _stop_loss_aid(store, asof, "TSLA") == "AID-STOP"


def test_stop_loss_recovers_null_id_row_by_client_order_id_no_resubmit(tmp_path):
    store = _store(tmp_path)
    asof = date(2026, 6, 5)
    _insert_stop_loss_null_id_row(store, asof, ticker="AAPL")
    alpaca = MagicMock()
    alpaca.get_order_by_client_order_id.return_value = ("AID-RECOVERED", "accepted")
    alpaca.submit_market_sell.side_effect = AssertionError("must NOT resubmit")

    assert _submit_stop_sell(
        ticker="AAPL",
        shares=10,
        last_price=100.0,
        asof=asof,
        store=store,
        alpaca=alpaca,
        run_id=2,
        reason="triggered",
    )

    alpaca.get_order_by_client_order_id.assert_called_once_with(
        "sma-2026-06-05-AAPL-stop-loss-SELL"
    )
    assert _stop_loss_aid(store, asof, "AAPL") == "AID-RECOVERED"


def test_stop_loss_existing_submitted_row_skips_without_broker_lookup(tmp_path):
    store = _store(tmp_path)
    asof = date(2026, 6, 5)
    store.conn.execute(
        "INSERT INTO intended_orders (intended_order_id, asof_date, ticker, side, "
        "target_shares, target_weight, last_price, source, status, alpaca_order_id, run_id) "
        "VALUES (?, ?, 'MSFT', 'SELL', 5, NULL, 80.0, 'stop-loss', 'submitted', 'AID-OLD', 1)",
        [str(uuid.uuid4()), asof],
    )
    alpaca = MagicMock()
    alpaca.submit_market_sell.side_effect = AssertionError("must NOT resubmit")

    assert _submit_stop_sell(
        ticker="MSFT",
        shares=5,
        last_price=80.0,
        asof=asof,
        store=store,
        alpaca=alpaca,
        run_id=2,
        reason="triggered",
    )

    alpaca.get_order_by_client_order_id.assert_not_called()
    assert _stop_loss_aid(store, asof, "MSFT") == "AID-OLD"


def test_stop_loss_recovery_lookup_error_fails_closed_no_resubmit(tmp_path):
    store = _store(tmp_path)
    asof = date(2026, 6, 5)
    _insert_stop_loss_null_id_row(store, asof, ticker="AMZN")
    alpaca = MagicMock()
    alpaca.get_order_by_client_order_id.side_effect = RuntimeError("alpaca 503")
    alpaca.submit_market_sell.side_effect = AssertionError("must NOT resubmit")

    assert not _submit_stop_sell(
        ticker="AMZN",
        shares=10,
        last_price=100.0,
        asof=asof,
        store=store,
        alpaca=alpaca,
        run_id=2,
        reason="triggered",
    )

    row = store.conn.execute(
        "SELECT status, alpaca_order_id, error FROM intended_orders "
        "WHERE asof_date=? AND ticker='AMZN' AND source='stop-loss'",
        [asof],
    ).fetchone()
    assert row[0] == "recovery_failed"
    assert row[1] is None
    assert "recovery lookup failed" in row[2]
