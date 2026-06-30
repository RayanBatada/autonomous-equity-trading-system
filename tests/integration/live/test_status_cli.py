"""Tests for `python -m sma.live status`."""

from datetime import date, datetime

from click.testing import CliRunner

from sma.ingest.store import Store
from sma.live.__main__ import cli


def _make_store_with_data(tmp_path):
    """Tmp store with one decide row, one stop-loss row, one snapshot."""
    db = tmp_path / "test.duckdb"
    store = Store(path=str(db)).connect()
    rid = store.allocate_run_id()

    # decide intended order
    import uuid
    store.conn.execute("""
        INSERT INTO intended_orders
        (intended_order_id, asof_date, ticker, side, target_shares, target_weight,
         last_price, source, status, run_id, created_at)
        VALUES (?, ?, 'AAPL', 'BUY', 25, 0.05, 200.0, 'decide', 'submitted', ?, ?)
    """, [str(uuid.uuid4()), date(2026, 5, 1), rid,
          datetime(2026, 5, 1, 18, 35, 12)])

    # stop-loss intended order
    store.conn.execute("""
        INSERT INTO intended_orders
        (intended_order_id, asof_date, ticker, side, target_shares, target_weight,
         last_price, source, status, run_id, created_at)
        VALUES (?, ?, 'MSFT', 'SELL', 10, NULL, 360.0, 'stop-loss', 'submitted', ?, ?)
    """, [str(uuid.uuid4()), date(2026, 5, 1), rid,
          datetime(2026, 5, 1, 9, 25, 8)])

    # account snapshot
    store.conn.execute("""
        INSERT INTO account_snapshots
        (asof_date, equity, cash, buying_power, long_market_value,
         position_count, run_id, created_at)
        VALUES (?, 100123.45, 50000.0, 100000.0, 50123.45, 17, ?, ?)
    """, [date(2026, 5, 1), rid, datetime(2026, 5, 1, 16, 30, 14)])

    # Close so the CLI can open its own connection.
    store.conn.close()
    return db


def test_status_prints_last_decide_fire(tmp_path):
    db = _make_store_with_data(tmp_path)
    runner = CliRunner()
    result = runner.invoke(cli, ["status", "--db", str(db)])
    assert result.exit_code == 0, result.output
    assert "Last decide fire:" in result.output
    assert "intended orders" in result.output


def test_status_prints_last_stop_loss_fire(tmp_path):
    db = _make_store_with_data(tmp_path)
    runner = CliRunner()
    result = runner.invoke(cli, ["status", "--db", str(db)])
    assert result.exit_code == 0
    assert "Last stop-loss fire:" in result.output


def test_status_prints_last_reconcile_fire(tmp_path):
    """Reconcile is summarized via account_snapshots latest write."""
    db = _make_store_with_data(tmp_path)
    runner = CliRunner()
    result = runner.invoke(cli, ["status", "--db", str(db)])
    assert result.exit_code == 0
    assert "Last reconcile fire:" in result.output
    assert "100,123.45" in result.output or "100123" in result.output


def test_status_handles_empty_db_cleanly(tmp_path):
    """Cold-start: no intended_orders, no account_snapshots → 'never'."""
    db = tmp_path / "empty.duckdb"
    Store(path=str(db)).connect()   # migrations only
    runner = CliRunner()
    result = runner.invoke(cli, ["status", "--db", str(db)])
    assert result.exit_code == 0, result.output
    assert "never" in result.output


def test_status_uses_iso8601_with_timezone(tmp_path):
    db = _make_store_with_data(tmp_path)
    runner = CliRunner()
    result = runner.invoke(cli, ["status", "--db", str(db)])
    # ISO-8601 with TZ format: "2026-05-01T18:35:12-04:00" or similar
    import re
    pattern = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}")
    assert pattern.search(result.output), (
        f"expected ISO-8601 with TZ in output:\n{result.output}"
    )


def test_decide_subcommand_help_includes_dry_run_and_canary():
    runner = CliRunner()
    result = runner.invoke(cli, ["decide", "--help"])
    assert result.exit_code == 0
    assert "--dry-run" in result.output
    assert "--canary" in result.output


def test_stop_loss_subcommand_registered():
    runner = CliRunner()
    result = runner.invoke(cli, ["stop-loss-sweep", "--help"])
    assert result.exit_code == 0


def test_reconcile_subcommand_registered():
    runner = CliRunner()
    result = runner.invoke(cli, ["reconcile", "--help"])
    assert result.exit_code == 0


def test_top_level_help_lists_all_subcommands():
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "decide" in result.output
    assert "stop-loss-sweep" in result.output
    assert "reconcile" in result.output
    assert "status" in result.output
