"""Migration 11: prices_intraday + session/order_type audit columns."""

import uuid
from datetime import date, datetime

from sma.ingest.store import Store


def _store(tmp_path):
    return Store(path=str(tmp_path / "m.duckdb")).connect()


def test_prices_intraday_upsert_is_idempotent(tmp_path):
    s = _store(tmp_path)
    row = ["SPY", datetime(2026, 9, 25, 13, 30), 1.0, 2.0, 0.5, 1.5, 100, "alpaca_iex", 1]
    sql = ("INSERT OR REPLACE INTO prices_intraday "
           "(ticker, ts, open, high, low, close, volume, source, run_id) "
           "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)")
    s.conn.execute(sql, row)
    s.conn.execute(sql, row[:-1] + [2])
    assert s.conn.execute("SELECT COUNT(*), MAX(run_id) FROM prices_intraday").fetchone() == (1, 2)


def test_session_columns_nullable_and_upsert_still_works(tmp_path):
    s = _store(tmp_path)
    iid = str(uuid.uuid4())
    s.conn.execute(
        "INSERT INTO intended_orders (intended_order_id, asof_date, ticker, side, "
        "target_shares, source, status, run_id) VALUES (?, ?, 'AAPL', 'BUY', 1, "
        "'decide', 'submitted', 1)", [iid, date(2026, 9, 25)])
    assert s.conn.execute(
        "SELECT session, order_type, limit_price FROM intended_orders").fetchone() == (
        None, None, None)
    ins = ("INSERT INTO paper_fills (alpaca_order_id, intended_order_id, asof_date, ticker, "
           "side, filled_shares, fill_price, status, submitted_at, run_id, session, order_type) "
           "VALUES ('o1', ?, ?, 'AAPL', 'BUY', ?, 10, 'filled', ?, 1, 'midday', 'limit') "
           "ON CONFLICT (alpaca_order_id) DO UPDATE SET filled_shares = EXCLUDED.filled_shares")
    s.conn.execute(ins, [iid, date(2026, 9, 25), 1, datetime(2026, 9, 25, 10, 35)])
    s.conn.execute(ins, [iid, date(2026, 9, 25), 2, datetime(2026, 9, 25, 10, 35)])
    assert s.conn.execute(
        "SELECT filled_shares, session, order_type FROM paper_fills").fetchone() == (
        2, "midday", "limit")
