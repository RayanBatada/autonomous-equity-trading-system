"""Pre-open guard (incident 2026-07-07: Alpaca wiped all 11 paper positions
overnight; the 09:30 open then sold a QCOM the account no longer held → a short).

Two layers: (1) a per-ticker OVERSELL gate that cancels any queued SELL exceeding
live holdings (the harm-preventer, live-data, any scale, no false-halt); (2) a
book-wide WIPE halt corroborated by an equity collapse (so a stale ledger never
false-halts)."""

from __future__ import annotations

from unittest.mock import MagicMock

from sma.live.preopen_guard import (
    PreOpenDivergence,
    check_preopen_divergence,
    find_oversell_orders,
    run_preopen_guard,
)

# --- book-wide divergence (pure) ---------------------------------------------


def test_full_wipe_is_divergent():
    d = check_preopen_divergence(ledger={f"T{i}": 100 for i in range(11)}, broker={})
    assert isinstance(d, PreOpenDivergence)
    assert d.is_divergent is True and d.missing_fraction == 1.0
    assert len(d.missing_names) == 11


def test_matching_book_is_not_divergent():
    ledger = {f"T{i}": 100 for i in range(11)}
    d = check_preopen_divergence(ledger=ledger, broker={f"T{i}": 100 for i in range(11)})
    assert d.is_divergent is False


def test_single_missing_name_is_not_divergent():
    ledger = {f"T{i}": 100 for i in range(11)}
    d = check_preopen_divergence(ledger=ledger, broker={f"T{i}": 100 for i in range(1, 11)})
    assert d.is_divergent is False


def test_small_share_drift_is_not_missing():
    ledger = {f"T{i}": 100 for i in range(11)}
    d = check_preopen_divergence(ledger=ledger, broker={f"T{i}": 98 for i in range(11)})
    assert d.is_divergent is False and d.missing_names == []


def test_tiny_book_below_floor_never_halts():
    assert check_preopen_divergence(ledger={"T0": 100}, broker={}).is_divergent is False


def test_divergence_symbol_canonicalised():
    """A dot/dash convention mismatch is not read as missing."""
    d = check_preopen_divergence(
        ledger={"BRK-B": 10, "AAPL": 5, "MSFT": 5},
        broker={"BRK.B": 10, "AAPL": 5, "MSFT": 5},
    )
    assert d.is_divergent is False and d.missing_names == []


# --- per-ticker oversell gate (pure) -----------------------------------------


def test_oversell_flags_sell_exceeding_holdings():
    orders = [{"id": "o1", "symbol": "QCOM", "side": "SELL", "qty": 6}]
    got = find_oversell_orders(open_orders=orders, broker={"QCOM": 0})
    assert len(got) == 1
    assert got[0]["id"] == "o1" and got[0]["held"] == 0 and got[0]["remaining"] == 6


def test_oversell_measures_unfilled_remainder_not_original_qty():
    """A partially-filled SELL (100 submitted, 30 filled → 70 remaining) does NOT
    oversell a 70-share holding; it DOES oversell a 60-share holding."""
    order = {"id": "o", "symbol": "AAPL", "side": "SELL", "qty": 100, "filled_qty": 30}
    assert find_oversell_orders(open_orders=[order], broker={"AAPL": 70}) == []
    got = find_oversell_orders(open_orders=[order], broker={"AAPL": 60})
    assert len(got) == 1 and got[0]["remaining"] == 70


def test_oversell_ignores_full_exit_and_trim():
    orders = [
        {"id": "a", "symbol": "AAPL", "side": "SELL", "qty": 25},  # full exit == held
        {"id": "b", "symbol": "MSFT", "side": "SELL", "qty": 5},   # trim < held
    ]
    assert find_oversell_orders(open_orders=orders, broker={"AAPL": 25, "MSFT": 40}) == []


def test_oversell_ignores_buys():
    orders = [{"id": "a", "symbol": "HOOD", "side": "BUY", "qty": 999}]
    assert find_oversell_orders(open_orders=orders, broker={}) == []


def test_oversell_symbol_canonicalised():
    """A held BRK-B must satisfy a queued BRK.B sell (not read as a naked short)."""
    orders = [{"id": "a", "symbol": "BRK.B", "side": "SELL", "qty": 3}]
    assert find_oversell_orders(open_orders=orders, broker={"BRK-B": 3}) == []


# --- run_preopen_guard (orchestration) ---------------------------------------


def _store(tmp_path, ledger, snapshot_equity=None):
    from sma.ingest.store import Store

    s = Store(path=str(tmp_path / "t.duckdb")).connect()
    for i, (t, n) in enumerate(ledger.items()):
        s.conn.execute(
            "INSERT INTO paper_fills (alpaca_order_id, asof_date, ticker, side, "
            " filled_shares, fill_price, status, submitted_at, filled_at, run_id) "
            "VALUES (?, DATE '2026-05-01', ?, 'BUY', ?, 100.0, 'filled', "
            " TIMESTAMP '2026-05-01 15:00:00', TIMESTAMP '2026-05-01 15:00:00', 1)",
            [f"o{i}", t, n],
        )
    if snapshot_equity is not None:
        s.conn.execute(
            "INSERT INTO account_snapshots (asof_date, equity, cash, buying_power, "
            " long_market_value, position_count, total_unrealized_pnl, run_id) "
            "VALUES (DATE '2026-05-01', ?, 0, 0, ?, 0, 0, 1)",
            [snapshot_equity, snapshot_equity],
        )
    return s


def _alpaca(open_orders=None, equity=100_000.0, refetch_positions=None):
    a = MagicMock()
    a.list_open_orders.return_value = open_orders or []
    a.get_account.return_value = {"equity": str(equity)}
    # Re-fetch on an empty book returns {} by default (confirms a real wipe);
    # a test can pass refetch_positions to simulate a transient-empty read.
    a.get_positions.return_value = refetch_positions if refetch_positions is not None else {}
    return a


def test_guard_wipe_cancels_oversell_and_halts(tmp_path):
    """The 7/07 scenario: whole book gone + equity collapsed + a queued QCOM sell.
    → cancel the oversell sell, cancel ALL orders, page HALT, wipe_halt=True."""
    store = _store(tmp_path, {f"T{i}": 100 for i in range(11)}, snapshot_equity=110_000)
    alpaca = _alpaca(
        open_orders=[{"id": "q", "symbol": "T0", "side": "SELL", "qty": 100}],
        equity=6_969.0,  # crashed
    )
    pages = []
    try:
        r = run_preopen_guard(
            store=store, broker_positions={}, alpaca=alpaca,
            notify_fn=lambda title, message: pages.append((title, message)),
        )
    finally:
        store.close()
    assert r.wipe_halt is True
    alpaca.cancel_order.assert_called_once_with("q")   # the oversell sell
    alpaca.cancel_all_open_orders.assert_called_once()  # whole-book wipe
    assert any("HALTED" in t for t, _ in pages)


def test_guard_partial_glitch_cancels_oversell_without_halting(tmp_path):
    """A sub-threshold glitch (a few names zeroed) must still stop the phantom
    short, but NOT halt the whole day — this is the case the old 50%-only design
    missed."""
    ledger = {f"T{i}": 100 for i in range(11)}
    broker = {f"T{i}": {"shares": 100} for i in range(1, 11)}  # T0 zeroed only
    store = _store(tmp_path, ledger, snapshot_equity=110_000)
    alpaca = _alpaca(
        open_orders=[{"id": "q", "symbol": "T0", "side": "SELL", "qty": 100}],
        equity=109_000.0,  # equity fine (one name gone)
        refetch_positions=broker,  # re-fetch CONFIRMS T0 is really gone
    )
    pages = []
    try:
        r = run_preopen_guard(
            store=store, broker_positions=broker, alpaca=alpaca,
            notify_fn=lambda title, message: pages.append((title, message)),
        )
    finally:
        store.close()
    assert r.wipe_halt is False
    alpaca.cancel_order.assert_called_once_with("q")
    alpaca.cancel_all_open_orders.assert_not_called()
    assert any("oversell" in t.lower() for t, _ in pages)


def test_guard_stale_ledger_does_not_false_halt(tmp_path):
    """Ledger shows longs the broker lacks (e.g. a missed reconcile) BUT equity is
    intact and the queued orders match the real book → no cancel, no halt."""
    ledger = {f"OLD{i}": 100 for i in range(6)}  # stale names the broker sold
    ledger.update({f"CUR{i}": 100 for i in range(5)})
    broker = {f"CUR{i}": {"shares": 100} for i in range(5)}  # only current names held
    store = _store(tmp_path, ledger, snapshot_equity=110_000)
    alpaca = _alpaca(open_orders=[], equity=110_000.0)  # equity intact
    pages = []
    try:
        r = run_preopen_guard(
            store=store, broker_positions=broker, alpaca=alpaca,
            notify_fn=lambda title, message: pages.append((title, message)),
        )
    finally:
        store.close()
    assert r.divergence.is_divergent is True  # positions look missing...
    assert r.wipe_halt is False               # ...but equity intact → NO halt
    alpaca.cancel_all_open_orders.assert_not_called()
    # ...but it is NOT silent: a low-severity "verify" page fires (never a halt).
    assert any("equity intact" in t for t, _ in pages)


def test_guard_healthy_morning_is_a_noop(tmp_path):
    ledger = {f"T{i}": 100 for i in range(11)}
    broker = {f"T{i}": {"shares": 100} for i in range(11)}
    store = _store(tmp_path, ledger, snapshot_equity=110_000)
    alpaca = _alpaca(open_orders=[{"id": "s", "symbol": "T0", "side": "SELL", "qty": 100}])
    pages = []
    try:
        r = run_preopen_guard(
            store=store, broker_positions=broker, alpaca=alpaca,
            notify_fn=lambda title, message: pages.append((title, message)),
        )
    finally:
        store.close()
    assert r.wipe_halt is False and r.oversell_cancelled == []
    alpaca.cancel_order.assert_not_called()
    assert pages == []


def test_guard_transient_empty_read_is_reconfirmed_not_acted_on(tmp_path):
    """A transient empty get_positions() (Alpaca 200 with []) must NOT trigger
    false cancels — the guard re-fetches, sees the real book, and no-ops."""
    ledger = {f"T{i}": 100 for i in range(11)}
    store = _store(tmp_path, ledger, snapshot_equity=110_000)
    # First read (passed in) is empty; the re-fetch returns the real full book.
    alpaca = _alpaca(
        open_orders=[{"id": "s", "symbol": "T0", "side": "SELL", "qty": 100}],
        refetch_positions={f"T{i}": {"shares": 100} for i in range(11)},
    )
    pages = []
    try:
        r = run_preopen_guard(
            store=store, broker_positions={}, alpaca=alpaca,  # transient empty
            notify_fn=lambda title, message: pages.append((title, message)),
        )
    finally:
        store.close()
    assert r.wipe_halt is False and r.oversell_cancelled == []
    alpaca.cancel_order.assert_not_called()
    assert pages == []


def test_guard_still_pages_when_cancel_fails(tmp_path):
    store = _store(tmp_path, {f"T{i}": 100 for i in range(11)}, snapshot_equity=110_000)
    alpaca = _alpaca(
        open_orders=[{"id": "q", "symbol": "T0", "side": "SELL", "qty": 100}],
        equity=6_000.0,
    )
    alpaca.cancel_order.side_effect = RuntimeError("503")
    pages = []
    try:
        run_preopen_guard(
            store=store, broker_positions={}, alpaca=alpaca,
            notify_fn=lambda title, message: pages.append((title, message)),
        )
    finally:
        store.close()
    assert any("HALTED" in t for t, _ in pages)


def test_alpaca_cancel_all_open_orders_calls_broker():
    from sma.live.alpaca_client import AlpacaClient

    tc = MagicMock()
    AlpacaClient(trading_client=tc).cancel_all_open_orders()
    tc.cancel_orders.assert_called_once_with()
