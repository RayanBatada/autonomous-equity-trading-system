"""sleeve_targets / sleeve_daily_returns: migration 10, persistence, scoring."""

from __future__ import annotations

from datetime import date

import pytest

from sma.ingest.store import MIGRATIONS, Store
from sma.strategies.allocator import SleeveProposal
from sma.strategies.attribution import (
    format_sleeve_status,
    persist_sleeve_targets,
    score_live,
    score_pending,
    score_shadow,
    sleeve_status,
)
from sma.strategies.base import TargetBook

D0, D1, D2 = date(2026, 9, 24), date(2026, 9, 25), date(2026, 9, 28)


@pytest.fixture
def store(tmp_path):
    s = Store(path=tmp_path / "t.duckdb").connect()
    yield s
    s.close()


def _price(store, t, d, px, source="yfinance"):
    store.conn.execute(
        "INSERT INTO prices (ticker, date, open, high, low, close, adj_close, volume, source, "
        "run_id) VALUES (?, ?, ?, ?, ?, ?, ?, 1000, ?, 1)",
        [t, d, px, px, px, px, px, source],
    )


def _snap(store, d, eq, source="portfolio_history_daily_close"):
    store.conn.execute(
        "INSERT INTO account_snapshots (asof_date, equity, cash, buying_power, "
        "long_market_value, position_count, total_unrealized_pnl, run_id, equity_source) "
        "VALUES (?, ?, 0, 0, ?, 0, 0, 1, ?)",
        [d, eq, eq, source],
    )


def _props():
    return [
        SleeveProposal("xgb_momentum", "live", 1.0, "open", TargetBook({"AAA": 0.5, "BBB": 0.7})),
        SleeveProposal("rev", "shadow", 0.0, "open", TargetBook({"AAA": 0.5, "CCC": 0.5})),
        SleeveProposal("broken", "shadow", 0.0, "open", None, error="x"),
    ]


def test_migration_10_is_registered_and_idempotent(tmp_path):
    assert any(v == 10 for v, _ in MIGRATIONS)
    sql = dict(MIGRATIONS)[10]
    s = Store(path=tmp_path / "m.duckdb").connect()
    try:
        s.conn.execute(sql)  # re-applying must be a no-op
        s.conn.execute(sql)
        tables = {r[0] for r in s.conn.execute(
            "SELECT table_name FROM information_schema.tables").fetchall()}
        assert {"sleeve_targets", "sleeve_daily_returns"} <= tables
    finally:
        s.close()
    # Reconnecting re-runs the migration loop without error.
    Store(path=tmp_path / "m.duckdb").connect().close()


def test_persist_writes_live_and_shadow_and_replaces_on_rerun(store):
    n = persist_sleeve_targets(store.conn, asof=D0, session="open", proposals=_props(), run_id=7)
    assert n == 4
    n2 = persist_sleeve_targets(store.conn, asof=D0, session="open", proposals=_props(), run_id=8)
    assert n2 == 4
    rows = store.conn.execute(
        "SELECT sleeve, ticker, weight, mode, capital_fraction, run_id FROM sleeve_targets "
        "ORDER BY sleeve, ticker").fetchall()
    assert rows == [
        ("rev", "AAA", 0.5, "shadow", 0.0, 8),
        ("rev", "CCC", 0.5, "shadow", 0.0, 8),
        ("xgb_momentum", "AAA", 0.5, "live", 1.0, 8),
        ("xgb_momentum", "BBB", 0.7, "live", 1.0, 8),
    ]


def test_score_shadow_uses_next_session_prices(store):
    persist_sleeve_targets(store.conn, asof=D0, session="open", proposals=_props(), run_id=1)
    for t, p0, p1 in [("SPY", 100, 101), ("AAA", 10, 11), ("CCC", 20, 19)]:
        _price(store, t, D0, p0)
        _price(store, t, D1, p1)
    # a lower-priority duplicate source must not be picked
    _price(store, "AAA", D1, 999, source="alpaca")
    out = score_shadow(store.conn, D0)
    assert set(out) == {"rev"}
    assert out["rev"].ret == pytest.approx(0.5 * 0.10 + 0.5 * (-0.05))
    assert out["rev"].gross == pytest.approx(1.0)
    assert out["rev"].realized_on == D1


def test_score_shadow_waits_for_partial_ingest_then_counts_missing_as_zero(store):
    persist_sleeve_targets(store.conn, asof=D0, session="open", proposals=_props(), run_id=1)
    _price(store, "SPY", D0, 100)
    _price(store, "SPY", D1, 101)
    _price(store, "AAA", D0, 10)
    _price(store, "AAA", D1, 11)
    _price(store, "CCC", D0, 20)  # no CCC close on D1 yet
    assert score_shadow(store.conn, D0) == {}
    _price(store, "SPY", D2, 102)  # a later session exists: CCC is genuinely missing
    out = score_shadow(store.conn, D0)
    assert out["rev"].ret == pytest.approx(0.05)
    assert out["rev"].missing == ["CCC"]


def test_score_live_single_sleeve_is_the_book_return(store):
    persist_sleeve_targets(store.conn, asof=D0, session="open", proposals=_props(), run_id=1)
    _price(store, "SPY", D0, 100)
    _price(store, "SPY", D1, 101)
    _snap(store, D0, 100_000.0)
    _snap(store, D1, 101_500.0)
    out = score_live(store.conn, D0)
    assert set(out) == {"xgb_momentum"}
    assert out["xgb_momentum"].ret == pytest.approx(0.015)


def test_score_pending_is_idempotent_and_waits_for_data(store):
    persist_sleeve_targets(store.conn, asof=D0, session="open", proposals=_props(), run_id=1)
    assert score_pending(store.conn, run_id=2) == 0  # nothing landed yet
    for t, p0, p1 in [("SPY", 100, 101), ("AAA", 10, 11), ("CCC", 20, 19)]:
        _price(store, t, D0, p0)
        _price(store, t, D1, p1)
    _snap(store, D0, 100_000.0)
    _snap(store, D1, 99_000.0)
    assert score_pending(store.conn, run_id=3) == 2
    assert score_pending(store.conn, run_id=4) == 0
    rows = dict(store.conn.execute(
        "SELECT sleeve, ret FROM sleeve_daily_returns WHERE asof_date = ?", [D0]).fetchall())
    assert rows["xgb_momentum"] == pytest.approx(-0.01)
    assert rows["rev"] == pytest.approx(0.025)


def test_sleeve_status_lines(store):
    class S:
        def __init__(self, name, f, mode, enabled=True):
            self.name, self.capital_fraction, self.mode, self.enabled = name, f, mode, enabled

    persist_sleeve_targets(store.conn, asof=D0, session="open", proposals=_props(), run_id=1)
    store.conn.execute(
        "INSERT INTO sleeve_daily_returns VALUES (?, 'rev', 'shadow', 0.02, 1.0, 1)", [D0])
    rows = sleeve_status(store.conn, [S("xgb_momentum", 1.0, "live"), S("rev", 0.0, "shadow")])
    lines = format_sleeve_status(rows)
    assert lines[0] == "Sleeves:"
    assert "xgb_momentum" in lines[1] and "live" in lines[1] and "100%" in lines[1]
    assert "2026-09-24" in lines[1]
    assert "+2.00%" in lines[2]


def test_sleeve_status_tolerates_unmigrated_db(tmp_path):
    import duckdb

    class S:
        name, capital_fraction, mode, enabled = "xgb_momentum", 1.0, "live", True

    conn = duckdb.connect(str(tmp_path / "old.duckdb"))
    rows = sleeve_status(conn, [S()])
    assert rows[0].last_asof is None
    assert "never" in format_sleeve_status(rows)[1]


def test_score_live_waits_for_official_close_inside_heal_window(store):
    from datetime import timedelta

    persist_sleeve_targets(store.conn, asof=D0, session="open", proposals=_props(), run_id=1)
    _price(store, "SPY", D0, 100)
    _price(store, "SPY", D1, 101)
    _snap(store, D0, 100_000.0)
    _snap(store, D1, 101_500.0, source="portfolio_history_1min_close")
    assert score_live(store.conn, D0) == {}  # proxy may still be healed
    for i in range(5):  # past reconcile's heal window: the proxy is final
        _snap(store, D1 + timedelta(days=3 + i), 100_000.0)
    assert score_live(store.conn, D0)["xgb_momentum"].ret == pytest.approx(0.015)
