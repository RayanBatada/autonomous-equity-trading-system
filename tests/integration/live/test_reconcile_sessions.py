"""Reconcile records intraday-session fills (with session/order_type), writes
fill counterfactuals, and groups fill quality per session. The open path's
numbers are computed exactly as before."""

import uuid
from datetime import date, datetime

import pytest

from sma.live.reconcile import (
    FILL_QUALITY_MAX_ABS_BP,
    counterfactual_stats,
    fill_quality_by_session,
    format_fill_quality,
    reconcile,
    record_fill_counterfactuals,
)
from tests.integration.live.test_reconcile import (
    _alpaca_with_filled_orders,
    _make_store,
)

ASOF = date(2026, 5, 1)


def _intended(store, *, ticker, side, source, oid, session=None, order_type=None,
              last_price=100.0, qty=10):
    iid = str(uuid.uuid4())
    store.conn.execute(
        "INSERT INTO intended_orders (intended_order_id, asof_date, ticker, side, "
        "target_shares, last_price, source, alpaca_order_id, status, run_id, session, "
        "order_type) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'submitted', 1, ?, ?)",
        [iid, ASOF, ticker, side, qty, last_price, source, oid, session, order_type])
    return iid


def _price(store, ticker, d, o, c, source="yfinance"):
    store.conn.execute(
        "INSERT INTO prices (ticker, date, open, high, low, close, adj_close, volume, "
        "source, run_id) VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, 1)",
        [ticker, d, o, max(o, c), min(o, c), c, c, source])


def test_session_fill_recorded_with_labels_and_decide_untouched(tmp_path):
    store = _make_store(tmp_path)
    _intended(store, ticker="AAPL", side="BUY", source="decide", oid="d1")
    _intended(store, ticker="MSFT", side="BUY", source="session-midday", oid="s1",
              session="midday", order_type="limit")
    alpaca, _ = _alpaca_with_filled_orders(
        [("d1", "AAPL", "buy", 10, 101.0, "filled"),
         ("s1", "MSFT", "buy", 10, 100.2, "filled")],
        positions={"AAPL": 10, "MSFT": 10})
    res = reconcile(asof=ASOF, store=store, alpaca=alpaca, notify_fn=lambda m: None,
                    now=datetime(2026, 5, 2, 16, 30))
    assert res.fills_recorded == 2
    got = dict(store.conn.execute(
        "SELECT alpaca_order_id, COALESCE(session,'-') || '/' || COALESCE(order_type,'-') "
        "FROM paper_fills").fetchall())
    assert got == {"d1": "-/-", "s1": "midday/limit"}


def test_counterfactuals_backfill_then_freeze(tmp_path):
    store = _make_store(tmp_path)
    iid = _intended(store, ticker="AAPL", side="BUY", source="decide", oid="d1")
    store.conn.execute(
        "INSERT INTO paper_fills (alpaca_order_id, intended_order_id, asof_date, ticker, "
        "side, filled_shares, fill_price, status, submitted_at, filled_at, run_id) "
        "VALUES ('d1', ?, ?, 'AAPL', 'BUY', 10, 101, 'filled', ?, ?, 1)",
        [iid, ASOF, datetime(2026, 5, 2, 4, 0), datetime(2026, 5, 2, 9, 32)])
    assert record_fill_counterfactuals(store=store) == 1   # no prints yet: row with NULLs
    _price(store, "AAPL", date(2026, 5, 2), 100.0, 99.0, source="alpaca")
    _price(store, "AAPL", date(2026, 5, 2), 100.0, 102.0)  # alpaca (fill scale) wins
    assert record_fill_counterfactuals(store=store) == 1   # prints landed: completed
    assert record_fill_counterfactuals(store=store) == 0   # frozen
    o, c, src, obp, cbp = store.conn.execute(
        "SELECT open_print, close_print, print_source, open_bp, close_bp "
        "FROM fill_counterfactuals").fetchone()
    assert (o, c, src) == (100.0, 99.0, "alpaca")
    assert obp == pytest.approx(100.0)                       # bought 1% above the open
    assert cbp == pytest.approx((101 - 99) / 99 * 1e4)


def _fill(store, oid, ticker, px, side="BUY", day=date(2026, 5, 2)):
    iid = _intended(store, ticker=ticker, side=side, source="decide", oid=oid)
    store.conn.execute(
        "INSERT INTO paper_fills (alpaca_order_id, intended_order_id, asof_date, ticker, "
        "side, filled_shares, fill_price, status, submitted_at, filled_at, run_id) "
        "VALUES (?, ?, ?, ?, ?, 10, ?, 'filled', ?, ?, 1)",
        [oid, iid, ASOF, ticker, side, px, datetime(2026, 5, 2, 4, 0),
         datetime(day.year, day.month, day.day, 9, 32)])


def _cf(store, oid):
    return store.conn.execute(
        "SELECT open_print, close_print, print_source, open_bp, close_bp "
        "FROM fill_counterfactuals WHERE alpaca_order_id = ?", [oid]).fetchone()


def test_counterfactuals_yfinance_fallback_only_on_fill_scale(tmp_path):
    """2026-10-01: no alpaca row -> yfinance only when its close is within 5% of
    the fill (same share scale). CRWD's split-adjusted 168 next to a 682 fill
    is a different scale: NULL + 'unavailable', not -30,000bp."""
    store = _make_store(tmp_path)
    d = date(2026, 5, 2)
    _fill(store, "ok", "AAPL", 101.0)
    _price(store, "AAPL", d, 100.0, 104.0)                    # yfinance, 3% from fill
    _fill(store, "crwd", "CRWD", 682.0)
    _price(store, "CRWD", d, 168.0, 170.0)                    # split-adjusted, 4x off
    assert record_fill_counterfactuals(store=store) == 2
    o, c, src, obp, _ = _cf(store, "ok")
    assert (o, c, src) == (100.0, 104.0, "yfinance") and obp == pytest.approx(100.0)
    assert _cf(store, "crwd") == (None, None, "unavailable", None, None)
    # the raw alpaca bar lands later: picked up (row still lacks a print)
    _price(store, "CRWD", d, 680.0, 690.0, source="alpaca")
    assert record_fill_counterfactuals(store=store) == 1
    o, c, src, obp, cbp = _cf(store, "crwd")
    assert (o, c, src) == (680.0, 690.0, "alpaca")
    assert obp == pytest.approx((682 - 680) / 680 * 1e4)


def test_counterfactuals_recompute_rewrites_frozen_rows(tmp_path):
    store = _make_store(tmp_path)
    d = date(2026, 5, 2)
    _fill(store, "c1", "CRWD", 682.0, side="SELL")
    _price(store, "CRWD", d, 680.0, 690.0)                     # yfinance, same scale
    assert record_fill_counterfactuals(store=store) == 1
    assert _cf(store, "c1")[2] == "yfinance"
    _price(store, "CRWD", d, 681.0, 689.0, source="alpaca")
    assert record_fill_counterfactuals(store=store) == 0       # frozen
    assert record_fill_counterfactuals(store=store, recompute=True) == 1
    o, c, src, obp, _ = _cf(store, "c1")
    assert (o, c, src) == (681.0, 689.0, "alpaca")
    assert obp == pytest.approx(-(682 - 681) / 681 * 1e4)       # sold above the open
    stats = counterfactual_stats(store.conn)
    assert stats["n"] == 1 and stats["by_source"] == {"alpaca": 1}
    assert stats["open_bp_null"] == 0


def test_fill_quality_groups_and_open_method(tmp_path):
    store = _make_store(tmp_path)
    fills = [
        # (oid, ticker, side, px, qty, source, session, otype, arrival)
        ("d1", "AAPL", "BUY", 101.0, 10, "decide", None, None, 100.0),
        ("d2", "MSFT", "SELL", 99.0, 20, "decide", None, None, 100.0),
        ("d3", "CRWD", "BUY", 400.0, 1, "decide", None, None, 100.0),   # split-scale: excluded
        ("s1", "XLK", "BUY", 50.05, 10, "session-close", "close", "limit", 50.0),
    ]
    for oid, t, side, px, q, src, sess, ot, arr in fills:
        iid = _intended(store, ticker=t, side=side, source=src, oid=oid, session=sess,
                        order_type=ot, last_price=arr, qty=q)
        store.conn.execute(
            "INSERT INTO paper_fills (alpaca_order_id, intended_order_id, asof_date, ticker, "
            "side, filled_shares, fill_price, status, submitted_at, filled_at, run_id, "
            "session, order_type) VALUES (?, ?, ?, ?, ?, ?, ?, 'filled', ?, ?, 1, ?, ?)",
            [oid, iid, ASOF, t, side, q, px, datetime(2026, 5, 2, 4),
             datetime(2026, 5, 2, 9, 31), sess, ot])
        _price(store, t, date(2026, 5, 2), 100.0, 100.0)
    rows = {(r["session"], r["order_type"]): r for r in fill_quality_by_session(store.conn)}
    op = rows[("open", "opg")]
    assert op["n"] == 2 and op["excluded"] == 1
    assert op["mean_bp"] == pytest.approx(100.0)            # +100 buy, +100 sell
    # notional-weighted: 1010*100 + 1980*100 over 2990 = 100
    assert op["nw_mean_bp"] == pytest.approx(100.0)
    cl = rows[("close", "limit")]
    assert cl["n"] == 1 and cl["median_bp"] == pytest.approx(10.0)   # vs arrival mid 50.00
    text = "\n".join(format_fill_quality(list(rows.values())))
    assert "open" in text and "limit" in text


def test_fill_quality_split_scale_mismatch_excluded_not_averaged_in(tmp_path):
    """2026-10-01: fill_quality_by_session must pick prints the same way
    record_fill_counterfactuals does (alpaca first, yfinance only within 5%
    of the fill price). A split-scale yfinance print close enough in
    magnitude to dodge the |bp| > max_abs_bp backstop (here ~9% off, giving
    ~1000bp, under the 2000bp threshold) must still be excluded rather than
    silently averaged in as a bogus ~1000bp cost -- there is no alpaca row
    to fall back to, so the fill has no same-scale print at all."""
    store = _make_store(tmp_path)
    oid = "mismatch1"
    iid = _intended(store, ticker="ADSK", side="BUY", source="decide", oid=oid)
    store.conn.execute(
        "INSERT INTO paper_fills (alpaca_order_id, intended_order_id, asof_date, ticker, "
        "side, filled_shares, fill_price, status, submitted_at, filled_at, run_id) "
        "VALUES (?, ?, ?, 'ADSK', 'BUY', 10, 110.0, 'filled', ?, ?, 1)",
        [oid, iid, ASOF, datetime(2026, 5, 2, 4), datetime(2026, 5, 2, 9, 31)])
    _price(store, "ADSK", date(2026, 5, 2), 100.0, 100.0)   # yfinance, ~9% off: not same scale

    # Sanity: under the OLD (magnitude-only) rule this would NOT have been
    # caught -- (110-100)/100*1e4 = 1000bp, well under the 2000bp backstop.
    from sma.live.reconcile import _signed_cost_bp
    assert abs(_signed_cost_bp("BUY", 110.0, 100.0)) < FILL_QUALITY_MAX_ABS_BP

    rows = {(r["session"], r["order_type"]): r for r in fill_quality_by_session(store.conn)}
    op = rows[("open", "opg")]
    assert op["n"] == 0
    assert op["excluded"] == 1
    assert op["mean_bp"] is None and op["nw_mean_bp"] is None

    # Once a raw alpaca print lands (same scale as the fill), it's used and
    # the fill counts normally.
    _price(store, "ADSK", date(2026, 5, 2), 108.0, 109.0, source="alpaca")
    rows = {(r["session"], r["order_type"]): r for r in fill_quality_by_session(store.conn)}
    op = rows[("open", "opg")]
    assert op["n"] == 1 and op["excluded"] == 0
    assert op["mean_bp"] == pytest.approx((110.0 - 108.0) / 108.0 * 1e4)


def test_fill_quality_opg_numbers_unchanged_for_ordinary_rows(tmp_path):
    """Ordinary (non-split) fills where alpaca and yfinance prints agree must
    score identically whichever source is preferred -- proving the 2026-10-01
    switch to alpaca-first print selection doesn't move the OPG numbers for
    the common case. Same fills/expected values as
    test_fill_quality_groups_and_open_method's AAPL/MSFT rows, but now with
    both an alpaca AND a yfinance print on file for each, to pin down that
    the new alpaca-first precedence gives the same answer as the old
    yfinance-first one did when the two sources match."""
    store = _make_store(tmp_path)
    fills = [
        ("d1", "AAPL", "BUY", 101.0, 10),
        ("d2", "MSFT", "SELL", 99.0, 20),
    ]
    for oid, t, side, px, q in fills:
        iid = _intended(store, ticker=t, side=side, source="decide", oid=oid, last_price=100.0,
                        qty=q)
        store.conn.execute(
            "INSERT INTO paper_fills (alpaca_order_id, intended_order_id, asof_date, ticker, "
            "side, filled_shares, fill_price, status, submitted_at, filled_at, run_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, 'filled', ?, ?, 1)",
            [oid, iid, ASOF, t, side, q, px, datetime(2026, 5, 2, 4),
             datetime(2026, 5, 2, 9, 31)])
        _price(store, t, date(2026, 5, 2), 100.0, 100.0, source="alpaca")
        _price(store, t, date(2026, 5, 2), 100.0, 100.0, source="yfinance")

    rows = {(r["session"], r["order_type"]): r for r in fill_quality_by_session(store.conn)}
    op = rows[("open", "opg")]
    assert op["n"] == 2 and op["excluded"] == 0
    assert op["mean_bp"] == pytest.approx(100.0)             # +100 buy, +100 sell
    assert op["median_bp"] == pytest.approx(100.0)
    assert op["nw_mean_bp"] == pytest.approx(100.0)           # same as the yfinance-only case


def test_fill_quality_empty():
    import duckdb

    from sma.ingest.store import MIGRATIONS
    con = duckdb.connect(":memory:")
    for _, sql in MIGRATIONS:
        con.execute(sql)
    assert fill_quality_by_session(con) == []
    assert format_fill_quality([])[-1].strip() == "(no fills)"


def test_reconcile_cli_recompute_counterfactuals_only(tmp_path, monkeypatch):
    from click.testing import CliRunner

    import sma.live.__main__ as live_main
    from sma.ingest.store import Store

    db = tmp_path / "cf.duckdb"
    store = Store(db).connect()
    _fill(store, "c1", "CRWD", 682.0)
    _price(store, "CRWD", date(2026, 5, 2), 168.0, 170.0)      # split-adjusted only
    record_fill_counterfactuals(store=store)
    store.conn.execute("UPDATE fill_counterfactuals SET print_source='yfinance', "
                       "open_print=168, close_print=170, open_bp=30000")  # legacy row
    store.close()

    def no_broker(*a, **k):
        raise AssertionError("recompute must not touch the broker")
    monkeypatch.setattr(live_main, "_build_alpaca", no_broker)
    r = CliRunner().invoke(live_main.cli, ["reconcile", "--db", str(db),
                                           "--recompute-counterfactuals"])
    assert r.exit_code == 0, r.output
    assert "rows rewritten: 1" in r.output
    store = Store(db).connect()
    try:
        assert _cf(store, "c1") == (None, None, "unavailable", None, None)
    finally:
        store.close()
