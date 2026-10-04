"""Migration 9 widens the two share-quantity columns to DOUBLE so a fractional
quantity survives the round trip. Lossless for every existing row."""

from datetime import date, datetime

import pytest

from sma.ingest.store import MIGRATIONS, Store, current_schema_version


def test_migration_nine_exists_and_is_last():
    versions = [v for v, _ in MIGRATIONS]
    assert versions == sorted(versions), "migrations must stay in ascending order"
    assert current_schema_version() >= 9


def test_share_columns_are_double(tmp_path, monkeypatch):
    store = Store(path=":memory:")
    import duckdb
    store.conn = duckdb.connect(":memory:")
    store._apply_migrations()
    types = {
        r[0]: r[1] for r in store.conn.execute("DESCRIBE intended_orders").fetchall()
    }
    assert types["target_shares"] == "DOUBLE"
    types = {
        r[0]: r[1] for r in store.conn.execute("DESCRIBE paper_fills").fetchall()
    }
    assert types["filled_shares"] == "DOUBLE"


def _fresh_store():
    import duckdb

    from sma.ingest.store import Store
    s = Store(path=":memory:")
    s.conn = duckdb.connect(":memory:")
    s._apply_migrations()
    return s


def test_fractional_quantity_round_trips_through_intended_orders():
    s = _fresh_store()
    s.conn.execute(
        "INSERT INTO intended_orders (intended_order_id, asof_date, ticker, side, "
        "target_shares, target_weight, last_price, source, status, run_id) "
        "VALUES (gen_random_uuid(), ?, 'AAPL', 'BUY', ?, 0.1, 100.0, 'decide', 'submitted', 1)",
        [date(2026, 8, 27), 2.5],
    )
    got = s.conn.execute("SELECT target_shares FROM intended_orders").fetchone()[0]
    assert got == pytest.approx(2.5)


def test_fractional_quantity_round_trips_through_paper_fills():
    s = _fresh_store()
    s.conn.execute(
        "INSERT INTO paper_fills (alpaca_order_id, asof_date, ticker, side, "
        "filled_shares, fill_price, status, submitted_at, run_id) "
        "VALUES ('o1', ?, 'AAPL', 'BUY', ?, 100.0, 'filled', ?, 1)",
        [date(2026, 8, 27), 0.709973641, datetime(2026, 8, 27, 18, 35)],
    )
    got = s.conn.execute("SELECT filled_shares FROM paper_fills").fetchone()[0]
    assert got == pytest.approx(0.709973641)


def test_indexes_survive_the_alter():
    s = _fresh_store()
    idx = {r[0] for r in s.conn.execute("SELECT index_name FROM duckdb_indexes()").fetchall()}
    assert "idx_intended_orders_asof" in idx
    assert "idx_paper_fills_asof" in idx
