"""Schema migration v4 adds intended_orders, paper_fills, account_snapshots."""

import uuid
from datetime import date as date_cls
from datetime import datetime

import pytest

from sma.ingest.store import Store


def test_v4_creates_intended_orders_with_required_columns(tmp_path):
    db = tmp_path / "test.duckdb"
    store = Store(path=str(db)).connect()
    cols = {row[0] for row in store.conn.execute("DESCRIBE intended_orders").fetchall()}
    expected = {
        "intended_order_id", "asof_date", "ticker", "side",
        "target_shares", "target_weight", "last_price",
        "source", "alpaca_order_id", "status", "error",
        "run_id", "created_at",
    }
    assert expected.issubset(cols), f"missing: {expected - cols}"


def test_v4_creates_paper_fills_with_commission_and_fees(tmp_path):
    db = tmp_path / "test.duckdb"
    store = Store(path=str(db)).connect()
    cols = {row[0] for row in store.conn.execute("DESCRIBE paper_fills").fetchall()}
    assert "commission" in cols
    assert "fees" in cols
    assert "fill_price" in cols
    assert "filled_shares" in cols


def test_v4_creates_account_snapshots_with_long_market_value(tmp_path):
    db = tmp_path / "test.duckdb"
    store = Store(path=str(db)).connect()
    cols = {row[0] for row in store.conn.execute("DESCRIBE account_snapshots").fetchall()}
    assert "long_market_value" in cols
    assert "equity" in cols
    assert "cash" in cols
    assert "buying_power" in cols


def test_v4_intended_orders_unique_constraint_dedupes(tmp_path):
    """UNIQUE (asof_date, ticker, source) prevents two rows for same intent on same day."""
    db = tmp_path / "test.duckdb"
    store = Store(path=str(db)).connect()
    rid = store.allocate_run_id()
    iid = str(uuid.uuid4())
    store.conn.execute(
        "INSERT INTO intended_orders "
        "(intended_order_id, asof_date, ticker, side, target_shares, target_weight, "
        " last_price, source, status, run_id) "
        "VALUES (?, ?, ?, 'BUY', 10, 0.05, 100.0, 'decide', 'submitted', ?)",
        [iid, date_cls(2026, 5, 1), "AAPL", rid],
    )
    with pytest.raises(Exception):
        store.conn.execute(
            "INSERT INTO intended_orders "
            "(intended_order_id, asof_date, ticker, side, target_shares, target_weight, "
            " last_price, source, status, run_id) "
            "VALUES (?, ?, ?, 'BUY', 20, 0.10, 100.0, 'decide', 'submitted', ?)",
            [str(uuid.uuid4()), date_cls(2026, 5, 1), "AAPL", rid],
        )


def test_v4_intended_orders_persists_last_price(tmp_path):
    db = tmp_path / "test.duckdb"
    store = Store(path=str(db)).connect()
    rid = store.allocate_run_id()
    iid = str(uuid.uuid4())
    store.conn.execute(
        "INSERT INTO intended_orders "
        "(intended_order_id, asof_date, ticker, side, target_shares, target_weight, "
        " last_price, source, status, run_id) "
        "VALUES (?, ?, ?, 'BUY', 10, 0.05, 237.42, 'decide', 'submitted', ?)",
        [iid, date_cls(2026, 5, 1), "AAPL", rid],
    )
    row = store.conn.execute(
        "SELECT last_price FROM intended_orders WHERE intended_order_id = ?", [iid],
    ).fetchone()
    assert row[0] == 237.42


def test_v4_paper_fills_persists_commission_and_fees(tmp_path):
    db = tmp_path / "test.duckdb"
    store = Store(path=str(db)).connect()
    rid = store.allocate_run_id()
    store.conn.execute(
        "INSERT INTO paper_fills "
        "(alpaca_order_id, asof_date, ticker, side, filled_shares, fill_price, "
        " commission, fees, status, submitted_at, run_id) "
        "VALUES (?, ?, ?, 'BUY', 10, 200.0, 0.0, 0.05, 'filled', ?, ?)",
        ["ord-1", date_cls(2026, 5, 1), "AAPL", datetime(2026, 5, 1, 9, 30), rid],
    )
    row = store.conn.execute(
        "SELECT commission, fees FROM paper_fills WHERE alpaca_order_id = ?", ["ord-1"],
    ).fetchone()
    assert row == (0.0, 0.05)


def test_v4_account_snapshots_persists_long_market_value(tmp_path):
    db = tmp_path / "test.duckdb"
    store = Store(path=str(db)).connect()
    rid = store.allocate_run_id()
    store.conn.execute(
        "INSERT INTO account_snapshots "
        "(asof_date, equity, cash, buying_power, long_market_value, position_count, run_id) "
        "VALUES (?, 100000.0, 50000.0, 100000.0, 50000.0, 5, ?)",
        [date_cls(2026, 5, 1), rid],
    )
    row = store.conn.execute(
        "SELECT long_market_value FROM account_snapshots WHERE asof_date = ?",
        [date_cls(2026, 5, 1)],
    ).fetchone()
    assert row[0] == 50000.0


def test_v4_account_snapshots_idempotent_per_day(tmp_path):
    """asof_date PK prevents two snapshots for the same day."""
    db = tmp_path / "test.duckdb"
    store = Store(path=str(db)).connect()
    rid = store.allocate_run_id()
    store.conn.execute(
        "INSERT INTO account_snapshots "
        "(asof_date, equity, cash, buying_power, long_market_value, position_count, run_id) "
        "VALUES (?, 100000.0, 50000.0, 100000.0, 50000.0, 5, ?)",
        [date_cls(2026, 5, 1), rid],
    )
    with pytest.raises(Exception):
        store.conn.execute(
            "INSERT INTO account_snapshots "
            "(asof_date, equity, cash, buying_power, long_market_value, position_count, run_id) "
            "VALUES (?, 99000.0, 49000.0, 99000.0, 50000.0, 5, ?)",
            [date_cls(2026, 5, 1), rid],
        )


def test_v4_migration_idempotent(tmp_path):
    """Running migrations twice on the same DB doesn't error."""
    db = tmp_path / "test.duckdb"
    Store(path=str(db)).connect()
    # Second connect re-runs migrations
    Store(path=str(db)).connect()
