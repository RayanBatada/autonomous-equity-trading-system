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


def test_status_reports_latest_equity_not_the_all_time_high(tmp_path):
    """In a DRAWDOWN, status must report the LATEST snapshot's equity.

    Regression (found 2026-07-29): the summary was built from
    `MAX(asof_date), MAX(created_at), MAX(equity)` — three independent
    aggregates — so it paired the newest date with the highest equity EVER
    recorded. Live, `status` claimed `asof=2026-07-28, equity=$111,202.45`
    (the 7/7 peak) while the account was actually at $96,585.61. A health
    check that always prints the high-water mark hides the exact condition
    you most need it to surface. Every pre-existing test inserted only ONE
    snapshot, where MAX(equity) coincidentally equals the latest, so the bug
    was invisible.
    """
    import uuid

    db = tmp_path / "drawdown.duckdb"
    store = Store(path=str(db)).connect()
    rid = store.allocate_run_id()
    store.conn.execute("""
        INSERT INTO intended_orders
        (intended_order_id, asof_date, ticker, side, target_shares, target_weight,
         last_price, source, status, run_id, created_at)
        VALUES (?, ?, 'AAPL', 'BUY', 25, 0.05, 200.0, 'decide', 'submitted', ?, ?)
    """, [str(uuid.uuid4()), date(2026, 5, 1), rid,
          datetime(2026, 5, 1, 18, 35, 12)])
    # Peak first, then a lower CURRENT snapshot.
    store.conn.execute("""
        INSERT INTO account_snapshots
        (asof_date, equity, cash, buying_power, long_market_value,
         position_count, run_id, created_at)
        VALUES (?, 111202.45, 50000.0, 100000.0, 61202.45, 11, ?, ?)
    """, [date(2026, 5, 1), rid, datetime(2026, 5, 1, 16, 30, 14)])
    store.conn.execute("""
        INSERT INTO account_snapshots
        (asof_date, equity, cash, buying_power, long_market_value,
         position_count, run_id, created_at)
        VALUES (?, 96585.61, 8998.45, 20000.0, 87587.16, 11, ?, ?)
    """, [date(2026, 5, 2), rid, datetime(2026, 5, 2, 16, 30, 14)])
    store.conn.close()

    result = CliRunner().invoke(cli, ["status", "--db", str(db)])
    assert result.exit_code == 0, result.output
    assert "96,585.61" in result.output, (
        f"status must report the LATEST equity, not the peak:\n{result.output}"
    )
    assert "111,202.45" not in result.output


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


def _seed_spy_prices(store, *dates):
    for d in dates:
        store.conn.execute(
            "INSERT INTO prices VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ["SPY", d, 450.0, 450.0, 450.0, 450.0, 450.0, 1_000_000, "yfinance", 1],
        )


def test_status_shows_staleness_warning_when_snapshot_is_two_sessions_behind(tmp_path):
    """The 2026-08-19 outage shape: the last snapshot is 2026-08-18, and SPY
    prices exist for 8/19 AND 8/20 (today) -- a full session was skipped."""
    db = tmp_path / "stale.duckdb"
    store = Store(path=str(db)).connect()
    rid = store.allocate_run_id()
    store.conn.execute("""
        INSERT INTO account_snapshots
        (asof_date, equity, cash, buying_power, long_market_value,
         position_count, run_id, created_at)
        VALUES ('2026-08-18', 102700.00, 7178.0, 100000.0, 95522.0, 11, ?, ?)
    """, [rid, datetime(2026, 8, 18, 16, 30, 14)])
    _seed_spy_prices(store, date(2026, 8, 19), date(2026, 8, 20))
    store.conn.close()

    result = CliRunner().invoke(cli, ["status", "--db", str(db)])

    assert result.exit_code == 0, result.output
    assert "WARNING" in result.output
    assert "2 sessions old" in result.output
    assert "2026-08-18" in result.output
    assert "live account may differ" in result.output


def test_status_no_staleness_warning_when_snapshot_is_current(tmp_path):
    db = tmp_path / "fresh.duckdb"
    store = Store(path=str(db)).connect()
    rid = store.allocate_run_id()
    store.conn.execute("""
        INSERT INTO account_snapshots
        (asof_date, equity, cash, buying_power, long_market_value,
         position_count, run_id, created_at)
        VALUES ('2026-08-20', 121600.00, 5000.0, 100000.0, 116600.0, 11, ?, ?)
    """, [rid, datetime(2026, 8, 20, 16, 30, 14)])
    _seed_spy_prices(store, date(2026, 8, 20))
    store.conn.close()

    result = CliRunner().invoke(cli, ["status", "--db", str(db)])

    assert result.exit_code == 0, result.output
    assert "WARNING" not in result.output


def test_status_no_staleness_warning_at_exactly_one_session_behind(tmp_path):
    """One session behind (yesterday's snapshot, today not yet reconciled) is
    the routine daily gap, not a caveat-worthy staleness."""
    db = tmp_path / "one_behind.duckdb"
    store = Store(path=str(db)).connect()
    rid = store.allocate_run_id()
    store.conn.execute("""
        INSERT INTO account_snapshots
        (asof_date, equity, cash, buying_power, long_market_value,
         position_count, run_id, created_at)
        VALUES ('2026-08-19', 121600.00, 5000.0, 100000.0, 116600.0, 11, ?, ?)
    """, [rid, datetime(2026, 8, 19, 16, 30, 14)])
    _seed_spy_prices(store, date(2026, 8, 20))
    store.conn.close()

    result = CliRunner().invoke(cli, ["status", "--db", str(db)])

    assert result.exit_code == 0, result.output
    assert "WARNING" not in result.output


def test_status_empty_db_has_no_staleness_warning(tmp_path):
    db = tmp_path / "empty2.duckdb"
    Store(path=str(db)).connect()
    result = CliRunner().invoke(cli, ["status", "--db", str(db)])
    assert result.exit_code == 0, result.output
    assert "WARNING" not in result.output


def _insert_fill(store, *, order_id, ticker, side, shares, run_id, asof="2026-05-01"):
    """Minimal paper_fills row -- only the columns _open_positions_count reads."""
    store.conn.execute(
        "INSERT INTO paper_fills (alpaca_order_id, asof_date, ticker, side, "
        " filled_shares, fill_price, status, submitted_at, filled_at, run_id) "
        "VALUES (?, ?, ?, ?, ?, 100.0, 'filled', ?, ?, ?)",
        [
            order_id, date.fromisoformat(asof), ticker, side, shares,
            datetime.fromisoformat(f"{asof}T15:00:00"),
            datetime.fromisoformat(f"{asof}T15:00:00"),
            run_id,
        ],
    )


def test_open_positions_counts_net_long_tickers_from_fills(tmp_path):
    """A ticker bought and held counts; one bought then fully sold does not.

    Regression target: `_open_positions_count` used to COUNT(DISTINCT ticker)
    over every BUY ever placed via decide, with no SELL offset at all (its own
    docstring claimed otherwise) -- a permanently-wrong gauge that printed 59
    while the real book held 11 names.
    """
    db = tmp_path / "fills.duckdb"
    store = Store(path=str(db)).connect()
    rid = store.allocate_run_id()
    _insert_fill(store, order_id="o1", ticker="AAPL", side="BUY", shares=100, run_id=rid)
    _insert_fill(store, order_id="o2", ticker="MSFT", side="BUY", shares=50, run_id=rid)
    _insert_fill(store, order_id="o3", ticker="MSFT", side="SELL", shares=50, run_id=rid)
    store.conn.close()

    result = CliRunner().invoke(cli, ["status", "--db", str(db)])
    assert result.exit_code == 0, result.output
    assert "Open positions (from fills): 1" in result.output


def test_open_positions_counts_partial_sell_as_still_open(tmp_path):
    db = tmp_path / "partial.duckdb"
    store = Store(path=str(db)).connect()
    rid = store.allocate_run_id()
    _insert_fill(store, order_id="o1", ticker="NVDA", side="BUY", shares=100, run_id=rid)
    _insert_fill(store, order_id="o2", ticker="NVDA", side="SELL", shares=40, run_id=rid)
    store.conn.close()

    result = CliRunner().invoke(cli, ["status", "--db", str(db)])
    assert result.exit_code == 0, result.output
    assert "Open positions (from fills): 1" in result.output


def test_open_positions_excludes_float_dust_from_a_full_exit(tmp_path):
    """BUY 0.1 + BUY 0.2 then SELL 0.3 nets to ~5.5e-17 in IEEE754 double
    arithmetic, not exactly 0 -- the same kind of residual
    `_detect_ledger_position_drift` guards against with QTY_EPS. That dust
    must not read as a still-open position."""
    db = tmp_path / "dust.duckdb"
    store = Store(path=str(db)).connect()
    rid = store.allocate_run_id()
    _insert_fill(store, order_id="o1", ticker="DUST", side="BUY", shares=0.1, run_id=rid)
    _insert_fill(store, order_id="o2", ticker="DUST", side="BUY", shares=0.2, run_id=rid)
    _insert_fill(store, order_id="o3", ticker="DUST", side="SELL", shares=0.3, run_id=rid)
    store.conn.close()

    result = CliRunner().invoke(cli, ["status", "--db", str(db)])
    assert result.exit_code == 0, result.output
    assert "Open positions (from fills): 0" in result.output


def test_open_positions_falls_back_to_snapshot_when_no_fills_recorded(tmp_path):
    """Cold DB / pre-first-reconcile: no paper_fills rows yet. Fall back to the
    latest account_snapshots.position_count rather than reporting a bare 0."""
    db = tmp_path / "nofills.duckdb"
    store = Store(path=str(db)).connect()
    rid = store.allocate_run_id()
    store.conn.execute("""
        INSERT INTO account_snapshots
        (asof_date, equity, cash, buying_power, long_market_value,
         position_count, run_id, created_at)
        VALUES (?, 117797.0, 8000.0, 8000.0, 109797.0, 11, ?, ?)
    """, [date(2026, 8, 31), rid, datetime(2026, 8, 31, 16, 30, 0)])
    store.conn.close()

    result = CliRunner().invoke(cli, ["status", "--db", str(db)])
    assert result.exit_code == 0, result.output
    assert "Open positions (from fills): 11" in result.output


def test_open_positions_empty_db_falls_back_to_zero(tmp_path):
    """No paper_fills AND no account_snapshots (true cold start) -> 0, not a
    crash on an empty fetchone()."""
    db = tmp_path / "cold.duckdb"
    Store(path=str(db)).connect()  # migrations only
    result = CliRunner().invoke(cli, ["status", "--db", str(db)])
    assert result.exit_code == 0, result.output
    assert "Open positions (from fills): 0" in result.output


def test_open_positions_label_replaces_old_uncorrected_gauge(tmp_path):
    """The renamed label must appear; the old, permanently-wrong label must not."""
    db = _make_store_with_data(tmp_path)
    result = CliRunner().invoke(cli, ["status", "--db", str(db)])
    assert result.exit_code == 0, result.output
    assert "Open positions (from fills):" in result.output
    assert "Open intended positions:" not in result.output


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
