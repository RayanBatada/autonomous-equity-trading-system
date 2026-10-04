

import pytest

from sma.ingest.store import Store, current_schema_version


def test_store_applies_all_migrations_on_fresh_db(tmp_path):
    db_path = tmp_path / "sma.duckdb"
    s = Store(path=db_path)
    s.connect()
    assert s.schema_version() == current_schema_version()
    # All eight tables exist
    tables = {r[0] for r in s.conn.execute(
        "SELECT table_name FROM duckdb_tables() WHERE schema_name = 'main'"
    ).fetchall()}
    expected = {"prices", "fundamentals", "news", "sentiment", "filings",
                "earnings", "market_index", "ingest_log", "_schema_version"}
    assert expected.issubset(tables)


def test_migrations_are_idempotent(tmp_path):
    db_path = tmp_path / "sma.duckdb"
    Store(path=db_path).connect().close()
    s2 = Store(path=db_path)
    s2.connect()
    assert s2.schema_version() == current_schema_version()
    s2.close()


def test_allocate_run_id_is_monotonic(tmp_path):
    db_path = tmp_path / "sma.duckdb"
    s = Store(path=db_path)
    s.connect()
    a = s.allocate_run_id()
    b = s.allocate_run_id()
    c = s.allocate_run_id()
    assert a < b < c


def test_log_run_writes_ingest_log_row(tmp_path):
    db_path = tmp_path / "sma.duckdb"
    s = Store(path=db_path)
    s.connect()
    rid = s.allocate_run_id()
    s.log_run_start(rid, source="yfinance")
    s.log_run_end(rid, source="yfinance", rows_inserted=120, status="ok", error=None)
    rows = s.conn.execute(
        "SELECT run_id, source, rows_inserted, status, error FROM ingest_log "
        "WHERE run_id = ? AND source = ?",
        [rid, "yfinance"],
    ).fetchall()
    assert rows == [(rid, "yfinance", 120, "ok", None)]


def test_in_memory_store_works_for_tests(tmp_path):
    s = Store(path=":memory:")
    s.connect()
    assert s.schema_version() == current_schema_version()


def test_predictions_table_exists_after_init(tmp_path):
    """Schema migration 2 creates the predictions table."""
    db_path = tmp_path / "test.duckdb"
    store = Store(path=str(db_path))
    store.connect()
    result = store.conn.execute(
        "SELECT COUNT(*) FROM information_schema.tables "
        "WHERE table_name = 'predictions'"
    ).fetchone()
    assert result[0] == 1, "predictions table should exist after init"
    store.close()


def test_schema_version_is_current_after_init(tmp_path):
    """After all migrations applied, schema version matches current_schema_version()."""
    db_path = tmp_path / "test.duckdb"
    store = Store(path=str(db_path))
    store.connect()
    v = store.conn.execute(
        "SELECT MAX(version) FROM _schema_version"
    ).fetchone()[0]
    assert v == current_schema_version()
    store.close()


def test_migration_v3_creates_theses_table():
    s = Store(":memory:").connect()
    cols = {r[1] for r in s.conn.execute("PRAGMA table_info('theses')").fetchall()}
    expected = {
        "ticker", "asof_date", "run_id",
        "news_summary", "key_developments", "notable_filings",
        "bull_case", "bear_case", "asymmetric_risks", "catalyst_window",
        "conviction", "score", "flags", "action_hint", "reasoning",
        "created_at",
    }
    assert expected.issubset(cols), f"missing: {expected - cols}"
    s.close()


def test_migration_v3_creates_agent_calls_table():
    s = Store(":memory:").connect()
    cols = {r[1] for r in s.conn.execute("PRAGMA table_info('agent_calls')").fetchall()}
    expected = {
        "call_id", "run_id", "ticker", "asof_date",
        "agent_role", "model_id",
        "input_tokens", "output_tokens", "cache_read_tokens",
        "est_cost_usd", "latency_ms", "status", "error", "created_at",
    }
    assert expected.issubset(cols), f"missing: {expected - cols}"
    s.close()


def test_migration_v3_idempotent(tmp_path):
    """Re-connecting to a real file should not crash on duplicate-table errors."""
    path = str(tmp_path / "v3_idempotent.duckdb")
    Store(path).connect().close()
    Store(path).connect().close()  # second connect should be safe


def test_migration_v8_account_snapshots_non_equity_columns_are_nullable(tmp_path):
    """2026-08-20: a gap-filled snapshot (backfill_missing_snapshots) knows
    only the official equity close for a day reconcile never ran -- cash,
    buying_power, long_market_value, and position_count are genuinely
    unknowable in retrospect and must be storable as NULL. equity itself
    stays NOT NULL."""
    s = Store(":memory:").connect()
    rid = s.allocate_run_id()
    s.conn.execute(
        """
        INSERT INTO account_snapshots
        (asof_date, equity, cash, buying_power, long_market_value,
         position_count, total_unrealized_pnl, run_id, equity_source)
        VALUES ('2026-08-19', 121_600.00, NULL, NULL, NULL, NULL, NULL, ?,
                'portfolio_history_daily_close')
        """,
        [rid],
    )
    row = s.conn.execute(
        "SELECT equity, cash, buying_power, long_market_value, position_count "
        "FROM account_snapshots WHERE asof_date = '2026-08-19'"
    ).fetchone()
    assert row == (121_600.00, None, None, None, None)
    s.close()


def test_migration_v8_equity_still_required(tmp_path):
    """equity was never relaxed -- a NULL equity row must still be rejected."""
    s = Store(":memory:").connect()
    rid = s.allocate_run_id()

    with pytest.raises(Exception):
        s.conn.execute(
            """
            INSERT INTO account_snapshots
            (asof_date, equity, cash, buying_power, long_market_value,
             position_count, total_unrealized_pnl, run_id, equity_source)
            VALUES ('2026-08-19', NULL, NULL, NULL, NULL, NULL, NULL, ?, NULL)
            """,
            [rid],
        )
    s.close()
