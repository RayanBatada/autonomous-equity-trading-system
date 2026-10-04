"""Tests for live.reconcile.reconcile."""

import uuid
from datetime import date, datetime, time
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

from alpaca.trading.client import TradingClient

from sma.ingest.store import Store
from sma.live.alpaca_client import AlpacaClient
from sma.live.reconcile import reconcile
from sma.live.trade_push import record_trade_push

ET = ZoneInfo("America/New_York")


def _make_store(tmp_path):
    return Store(path=str(tmp_path / "test.duckdb")).connect()


def _seed_intended(
    store, *, asof: date, ticker: str, side: str = "BUY",
    target_shares: int = 10, last_price: float = 200.0,
    alpaca_order_id: str | None = "ord-1", source: str = "decide",
):
    rid = store.allocate_run_id()
    iid = str(uuid.uuid4())
    store.conn.execute("""
        INSERT INTO intended_orders
        (intended_order_id, asof_date, ticker, side, target_shares,
         target_weight, last_price, source, alpaca_order_id, status, run_id)
        VALUES (?, ?, ?, ?, ?, NULL, ?, ?, ?, 'submitted', ?)
    """, [iid, asof, ticker, side, target_shares, last_price, source,
          alpaca_order_id, rid])
    return iid


def _seed_snapshot(store, *, asof: date, equity: float):
    rid = store.allocate_run_id()
    store.conn.execute("""
        INSERT INTO account_snapshots
        (asof_date, equity, cash, buying_power, long_market_value,
         position_count, run_id)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, [asof, equity, equity * 0.5, equity, equity * 0.5, 5, rid])


def _alpaca_with_filled_orders(orders_data, equity=100_000.0, position_count=5,
                               positions=None):
    """orders_data: [(id, ticker, side, filled_qty, fill_price, status), ...]

    positions: optional {symbol: shares} for the live book (so it can be made
    consistent with the recorded fills for the ledger-drift check). Defaults to
    a placeholder T0..T{position_count-1} book.
    """
    tc = MagicMock(spec=TradingClient)

    orders = []
    for oid, ticker, side, qty, price, status in orders_data:
        o = MagicMock()
        o.id = oid
        o.symbol = ticker
        o.side = side
        o.filled_qty = str(qty)
        o.filled_avg_price = str(price)
        o.commission = 0
        o.fees = 0
        o.status = status
        o.submitted_at = datetime(2026, 5, 1, 18, 35)
        o.filled_at = datetime(2026, 5, 2, 9, 30)
        orders.append(o)
    tc.get_orders.return_value = orders

    # Reconcile now fetches each decide order by id (via the intended_orders
    # row), not a submission-date window. Map ids -> order; unknown ids raise
    # (mirrors a broker 404) so unmatched/unseeded orders are skipped.
    by_id = {o.id: o for o in orders}

    def _get_by_id(order_id):
        if order_id in by_id:
            return by_id[order_id]
        raise RuntimeError(f"order {order_id} not found")

    tc.get_order_by_id.side_effect = _get_by_id

    acct = MagicMock()
    acct.equity = str(equity)
    acct.cash = str(equity * 0.5)
    acct.buying_power = str(equity)
    acct.long_market_value = str(equity * 0.5)
    acct.trading_blocked = False
    acct.account_blocked = False
    tc.get_account.return_value = acct

    pos_objs = []
    if positions is not None:
        for sym, qty in positions.items():
            p = MagicMock()
            p.symbol = sym
            p.qty = str(qty)
            p.avg_entry_price = "100.0"
            pos_objs.append(p)
    else:
        for i in range(position_count):
            p = MagicMock()
            p.symbol = f"T{i}"
            p.qty = "10"
            p.avg_entry_price = "100.0"
            pos_objs.append(p)
    tc.get_all_positions.return_value = pos_objs

    # Default next-session-date so reconcile's drift time-guard always passes.
    # Tests that want to exercise the gate override tc.get_calendar after this.
    ancient_session = MagicMock()
    ancient_session.date = date(1970, 1, 1)
    tc.get_calendar.return_value = [ancient_session]

    alpaca = AlpacaClient(trading_client=tc)
    # Default the pre-close snapshot guard (_pre_close_skip_reason) to "already
    # closed", keyed off whatever day is asked about, so tests that don't care
    # about session timing behave exactly as before that guard existed. This is
    # mocked directly on the AlpacaClient instance rather than via tc.get_calendar
    # so it doesn't collide with the ancient-session / calendar-failure tricks
    # above, which are answering an unrelated question (next_session_date).
    # Tests exercising the guard itself override alpaca.session_window after
    # construction.
    alpaca.session_window = MagicMock(
        side_effect=lambda *, day: (
            datetime.combine(day, time(9, 30, tzinfo=ET)),
            datetime.combine(day, time(0, 0, tzinfo=ET)),
        )
    )

    return alpaca, tc


def test_reconcile_inserts_paper_fills_with_commission_and_fees(tmp_path):
    store = _make_store(tmp_path)
    asof = date(2026, 5, 1)
    _seed_intended(store, asof=asof, ticker="AAPL", target_shares=25,
                   alpaca_order_id="ord-1")
    alpaca, tc = _alpaca_with_filled_orders([
        ("ord-1", "AAPL", "BUY", 25, 200.0, "filled"),
    ])

    result = reconcile(asof=asof, store=store, alpaca=alpaca,
                       notify_fn=lambda _: None)
    assert result.fills_recorded == 1
    rows = store.conn.execute(
        "SELECT alpaca_order_id, ticker, filled_shares, fill_price, "
        " commission, fees, status "
        "FROM paper_fills"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0] == ("ord-1", "AAPL", 25, 200.0, 0.0, 0.0, "filled")


def test_reconcile_upgrades_partial_fill_to_final_on_rerun(tmp_path):
    """A re-run after a partial fill must record the FINAL quantity. ON CONFLICT
    DO NOTHING froze the partial (5 / partially_filled) in paper_fills forever
    even after the order completed; DO UPDATE records the final (10 / filled)."""
    store = _make_store(tmp_path)
    asof = date(2026, 5, 1)
    _seed_intended(store, asof=asof, ticker="AAPL", target_shares=10,
                   alpaca_order_id="ord-1")
    # First reconcile sees a partial fill (5 of 10).
    alpaca1, _ = _alpaca_with_filled_orders([
        ("ord-1", "AAPL", "BUY", 5, 200.0, "partially_filled"),
    ])
    reconcile(asof=asof, store=store, alpaca=alpaca1, notify_fn=lambda _: None)
    # A later reconcile sees the SAME order now fully filled (10).
    alpaca2, _ = _alpaca_with_filled_orders([
        ("ord-1", "AAPL", "BUY", 10, 201.0, "filled"),
    ])
    reconcile(asof=asof, store=store, alpaca=alpaca2, notify_fn=lambda _: None)
    rows = store.conn.execute(
        "SELECT filled_shares, fill_price, status FROM paper_fills "
        "WHERE alpaca_order_id='ord-1'"
    ).fetchall()
    assert len(rows) == 1, "should upsert one row, not duplicate"
    assert rows[0] == (10, 201.0, "filled"), "must upgrade to final, not freeze partial"


def test_reconcile_writes_account_snapshot_under_snapshot_date(tmp_path):
    """The account snapshot is keyed by snapshot_date (today's date in ET when
    reconcile runs), NOT by the reconcile asof. The decide date is yesterday
    from reconcile's POV; tagging today's live equity under yesterday's row
    would overwrite yesterday's historical snapshot."""
    from zoneinfo import ZoneInfo
    eastern = ZoneInfo("America/New_York")

    store = _make_store(tmp_path)
    asof = date(2026, 5, 1)
    snapshot_date = date(2026, 5, 2)  # next session, when reconcile actually runs
    now = datetime(2026, 5, 2, 16, 30, tzinfo=eastern)
    alpaca, tc = _alpaca_with_filled_orders([], equity=100_000.0)

    result = reconcile(asof=asof, store=store, alpaca=alpaca,
                       notify_fn=lambda _: None, now=now)
    assert result.snapshot_written is True
    row = store.conn.execute(
        "SELECT equity, cash, long_market_value, position_count "
        "FROM account_snapshots WHERE asof_date = ?",
        [snapshot_date],
    ).fetchone()
    assert row is not None, "snapshot should be keyed by snapshot_date not asof"
    assert row[0] == 100_000.0
    assert row[2] == 50_000.0   # long_market_value
    # And the asof row should NOT have been touched.
    asof_row = store.conn.execute(
        "SELECT 1 FROM account_snapshots WHERE asof_date = ?", [asof],
    ).fetchone()
    assert asof_row is None, "asof row should not be auto-created by snapshot write"


def test_account_snapshots_idempotent_per_snapshot_date(tmp_path):
    """Two reconcile calls on the same calendar day → one row, latest wins."""
    from zoneinfo import ZoneInfo
    eastern = ZoneInfo("America/New_York")

    store = _make_store(tmp_path)
    asof = date(2026, 5, 1)
    snapshot_date = date(2026, 5, 2)
    now = datetime(2026, 5, 2, 16, 30, tzinfo=eastern)
    alpaca, tc = _alpaca_with_filled_orders([], equity=100_000.0)

    reconcile(asof=asof, store=store, alpaca=alpaca, notify_fn=lambda _: None, now=now)
    # Second call → upsert, not error
    alpaca2, _ = _alpaca_with_filled_orders([], equity=99_000.0)
    reconcile(asof=asof, store=store, alpaca=alpaca2, notify_fn=lambda _: None, now=now)

    rows = store.conn.execute(
        "SELECT equity FROM account_snapshots WHERE asof_date = ?", [snapshot_date],
    ).fetchall()
    assert len(rows) == 1
    assert rows[0][0] == 99_000.0   # latest write wins


def test_drift_query_quiet_when_everything_matches(tmp_path):
    """All intended buys filled exactly → zero alerts."""
    store = _make_store(tmp_path)
    asof = date(2026, 5, 1)
    _seed_intended(store, asof=asof, ticker="AAPL", side="BUY",
                   target_shares=25, alpaca_order_id="ord-1")
    # Book is consistent with the fill (AAPL 25) so the ledger-vs-broker check
    # also stays quiet — "everything matches" end to end.
    alpaca, tc = _alpaca_with_filled_orders([
        ("ord-1", "AAPL", "BUY", 25, 200.0, "filled"),
    ], positions={"AAPL": 25})

    result = reconcile(asof=asof, store=store, alpaca=alpaca,
                       notify_fn=lambda _: None)
    assert len(result.alerts) == 0


def test_drift_query_threshold_alert_on_20pct_buy_miss(tmp_path):
    """5 BUYs, 2 missed (40%) → systemic-miss alert."""
    store = _make_store(tmp_path)
    asof = date(2026, 5, 1)
    for i in range(5):
        _seed_intended(store, asof=asof, ticker=f"T{i}", side="BUY",
                       alpaca_order_id=f"ord-{i}")
    # Only 3 of 5 fill
    alpaca, tc = _alpaca_with_filled_orders([
        ("ord-0", "T0", "BUY", 10, 100.0, "filled"),
        ("ord-1", "T1", "BUY", 10, 100.0, "filled"),
        ("ord-2", "T2", "BUY", 10, 100.0, "filled"),
    ])

    alerts_received = []
    result = reconcile(asof=asof, store=store, alpaca=alpaca,
                       notify_fn=alerts_received.append)
    kinds = [a.kind for a in result.alerts]
    assert "buy_miss_systemic" in kinds
    assert any("buy_miss_systemic" in m for m in alerts_received)


def test_drift_query_classifies_buy_miss_vs_sell_miss(tmp_path):
    """One missed BUY (under 20% → no alert), one missed SELL (always alert)."""
    store = _make_store(tmp_path)
    asof = date(2026, 5, 1)
    # 10 BUYs, 1 missed (10%) → no buy alert
    for i in range(10):
        _seed_intended(store, asof=asof, ticker=f"B{i}", side="BUY",
                       alpaca_order_id=f"buy-{i}")
    # 1 SELL, 1 missed → sell alert
    _seed_intended(store, asof=asof, ticker="S0", side="SELL",
                   alpaca_order_id="sell-0")
    # Fill all but buy-0 and sell-0
    fills = [(f"buy-{i}", f"B{i}", "BUY", 10, 100.0, "filled") for i in range(1, 10)]
    alpaca, tc = _alpaca_with_filled_orders(fills)

    result = reconcile(asof=asof, store=store, alpaca=alpaca,
                       notify_fn=lambda _: None)
    kinds = [a.kind for a in result.alerts]
    # 1 buy missed of 10 = 10%, below 20% threshold → no buy alert
    assert "buy_miss_systemic" not in kinds
    # Sells always alert on miss
    assert "sell_miss" in kinds


def test_drift_query_finds_partial_fills(tmp_path):
    """filled_shares < target * 0.9 → partial-fill alert."""
    store = _make_store(tmp_path)
    asof = date(2026, 5, 1)
    _seed_intended(store, asof=asof, ticker="AAPL", side="BUY",
                   target_shares=100, alpaca_order_id="ord-1")
    alpaca, tc = _alpaca_with_filled_orders([
        ("ord-1", "AAPL", "BUY", 80, 200.0, "partially_filled"),  # 80% fill
    ])

    result = reconcile(asof=asof, store=store, alpaca=alpaca,
                       notify_fn=lambda _: None)
    kinds = [a.kind for a in result.alerts]
    assert "partial_fill" in kinds


def test_drift_query_catastrophic_loss_above_10pct(tmp_path):
    """Today's equity dropped >10% from yesterday → catastrophic_loss alert."""
    from zoneinfo import ZoneInfo
    eastern = ZoneInfo("America/New_York")

    store = _make_store(tmp_path)
    asof = date(2026, 4, 30)  # decide date (yesterday from reconcile's POV)
    _seed_snapshot(store, asof=date(2026, 4, 30), equity=100_000.0)
    alpaca, tc = _alpaca_with_filled_orders([], equity=85_000.0)   # 15% drop
    now = datetime(2026, 5, 1, 16, 30, tzinfo=eastern)  # snapshot_date=5/1

    result = reconcile(asof=asof, store=store, alpaca=alpaca,
                       notify_fn=lambda _: None, now=now)
    kinds = [a.kind for a in result.alerts]
    assert "catastrophic_loss" in kinds


def test_drift_query_no_alert_below_10pct_drop(tmp_path):
    """Equity dropped 8% (< 10%) → no alert."""
    from zoneinfo import ZoneInfo
    eastern = ZoneInfo("America/New_York")

    store = _make_store(tmp_path)
    asof = date(2026, 4, 30)
    _seed_snapshot(store, asof=date(2026, 4, 30), equity=100_000.0)
    alpaca, tc = _alpaca_with_filled_orders([], equity=92_000.0)   # 8% drop
    now = datetime(2026, 5, 1, 16, 30, tzinfo=eastern)

    result = reconcile(asof=asof, store=store, alpaca=alpaca,
                       notify_fn=lambda _: None, now=now)
    kinds = [a.kind for a in result.alerts]
    assert "catastrophic_loss" not in kinds


def test_reconcile_skips_buy_miss_drift_before_next_open_auction(tmp_path):
    """No buy_miss_systemic alert when reconcile runs before the next session's
    open auction has occurred. The OPG orders are still queued at Alpaca and
    have had no opportunity to fill, so flagging them as 'failed to cross at
    open' is a false positive.

    Repro of the canary smoke 2026-04-30 23:30 ET: decide ran 22:20 ET,
    intended_orders.asof_date=4/30, OPG order accepted, queued for 5/1 09:30
    ET open. Pre-fix reconcile flagged AAPL as missed-at-open even though the
    auction had not happened yet.
    """
    from zoneinfo import ZoneInfo
    eastern = ZoneInfo("America/New_York")

    store = _make_store(tmp_path)
    asof = date(2026, 4, 30)
    _seed_intended(store, asof=asof, ticker="AAPL", side="BUY",
                   target_shares=18, alpaca_order_id="ord-canary")
    alpaca, tc = _alpaca_with_filled_orders([
        ("ord-canary", "AAPL", "BUY", 0, 0, "accepted"),
    ])
    # Mock get_calendar so AlpacaClient.next_session_date(asof) returns 5/1.
    next_session = MagicMock()
    next_session.date = date(2026, 5, 1)
    tc.get_calendar.return_value = [next_session]

    # Inject "now" as 4/30 23:30 ET — before the next session's 09:30 ET open.
    now_et = datetime(2026, 4, 30, 23, 30, tzinfo=eastern)

    result = reconcile(asof=asof, store=store, alpaca=alpaca,
                       notify_fn=lambda _: None, now=now_et)
    assert result.fills_recorded == 0
    kinds = [a.kind for a in result.alerts]
    assert "buy_miss_systemic" not in kinds, (
        f"false-positive drift before next open auction: {result.alerts}"
    )


def test_reconcile_detects_buy_miss_after_next_open_auction(tmp_path):
    """The drift-detection path still fires once we're past the next open
    auction and the order didn't fill. Pairs with the skip-before test."""
    from zoneinfo import ZoneInfo
    eastern = ZoneInfo("America/New_York")

    store = _make_store(tmp_path)
    asof = date(2026, 4, 27)  # Mon
    _seed_intended(store, asof=asof, ticker="AAPL", side="BUY",
                   target_shares=18, alpaca_order_id="ord-canary")
    alpaca, tc = _alpaca_with_filled_orders([
        ("ord-canary", "AAPL", "BUY", 0, 0, "canceled"),
    ])
    next_session = MagicMock()
    next_session.date = date(2026, 4, 28)  # Tue
    tc.get_calendar.return_value = [next_session]

    # Tue 16:30 ET — after the 09:30 ET auction that should have filled the order.
    now_et = datetime(2026, 4, 28, 16, 30, tzinfo=eastern)

    result = reconcile(asof=asof, store=store, alpaca=alpaca,
                       notify_fn=lambda _: None, now=now_et)
    kinds = [a.kind for a in result.alerts]
    assert "buy_miss_systemic" in kinds


def test_reconcile_records_fills_when_status_is_real_alpaca_enum(tmp_path):
    """Production bug: Alpaca's OrderStatus enum stringifies as
    'OrderStatus.FILLED', so `str(order.status).lower() in ("filled",
    ...)` was always False and every real fill was silently dropped. The
    existing helper passes raw strings as `status`, masking the bug. This
    test wires a MagicMock that returns the actual SDK enum."""
    from alpaca.trading.enums import OrderStatus

    store = _make_store(tmp_path)
    asof = date(2026, 5, 1)
    _seed_intended(store, asof=asof, ticker="AAPL", side="BUY",
                   target_shares=1, alpaca_order_id="ord-real-enum")

    alpaca, tc = _alpaca_with_filled_orders([])  # empty so we override below
    o = MagicMock()
    o.id = "ord-real-enum"
    o.symbol = "AAPL"
    o.side = OrderStatus.FILLED  # not used for direction; just need a status object
    o.side = "BUY"
    o.filled_qty = "1"
    o.filled_avg_price = "281.89"
    o.commission = 0
    o.fees = 0
    o.status = OrderStatus.FILLED   # the real enum, not a raw string
    o.submitted_at = datetime(2026, 5, 1, 13, 57)
    o.filled_at = datetime(2026, 5, 1, 13, 57, 1)
    tc.get_orders.return_value = [o]
    tc.get_order_by_id.side_effect = lambda _oid: o  # reconcile fetches by id

    result = reconcile(asof=asof, store=store, alpaca=alpaca,
                       notify_fn=lambda _: None)
    assert result.fills_recorded == 1, (
        "fill must be recorded when Alpaca returns the real OrderStatus enum"
    )
    rows = store.conn.execute(
        "SELECT alpaca_order_id, ticker, filled_shares, fill_price, status "
        "FROM paper_fills"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0][0] == "ord-real-enum"
    assert rows[0][4] == "filled"   # stored value should be normalized lowercase


def test_reconcile_runs_catastrophic_loss_when_calendar_lookup_fails(tmp_path):
    """Codex HIGH 2: a failure inside next_session_date (network outage,
    empty Alpaca calendar) used to abort reconcile before catastrophic-loss
    drift could fire. Wrap the calendar call so order-drift skips on failure
    but equity-loss detection always runs and the snapshot still lands."""
    store = _make_store(tmp_path)
    asof = date(2026, 5, 1)
    _seed_snapshot(store, asof=date(2026, 4, 30), equity=100_000.0)
    alpaca, tc = _alpaca_with_filled_orders([], equity=85_000.0)   # 15% drop
    tc.get_calendar.side_effect = RuntimeError("alpaca calendar unreachable")

    result = reconcile(asof=asof, store=store, alpaca=alpaca,
                       notify_fn=lambda _: None)
    assert result.snapshot_written is True, "snapshot must still land"
    kinds = [a.kind for a in result.alerts]
    assert "catastrophic_loss" in kinds, (
        "catastrophic_loss must fire even when calendar lookup fails"
    )


# ---- 2026-06-05 audit: reconcile fill-recording + snapshot-failure holes -----


def test_record_fills_records_partial_then_canceled(tmp_path):
    """An order partially filled THEN canceled has a terminal status but a real
    fill (filled_qty>0) — it must be recorded, not dropped as a no-fill."""
    asof = date(2026, 5, 1)
    store = _make_store(tmp_path)
    _seed_intended(store, asof=asof, ticker="AAPL", alpaca_order_id="ord-1")
    alpaca, _ = _alpaca_with_filled_orders([("ord-1", "AAPL", "BUY", 3, 200.0, "canceled")])
    reconcile(asof=asof, store=store, alpaca=alpaca, notify_fn=lambda _: None)
    rows = store.conn.execute(
        "SELECT filled_shares FROM paper_fills WHERE alpaca_order_id='ord-1'"
    ).fetchall()
    assert len(rows) == 1 and rows[0][0] == 3


def test_reconcile_skips_ledger_drift_when_disabled(tmp_path):
    """reconcile_cmd runs the ledger-drift check ONCE after the whole drain (not
    per-batch), so a fill recorded in a LATER batch nets an earlier batch's
    apparent gap instead of paging mid-drain (adversarial review 2026-07-04, also
    the same-day stop-exit page). reconcile() must honor check_ledger_drift=False."""
    asof = date(2026, 5, 1)
    # Ledger will show AAPL +10 (recorded BUY) but the book is empty → a mismatch.
    store_on = Store(path=str(tmp_path / "on.duckdb")).connect()
    _seed_intended(store_on, asof=asof, ticker="AAPL", target_shares=10, alpaca_order_id="ord-1")
    alpaca_on, _ = _alpaca_with_filled_orders(
        [("ord-1", "AAPL", "BUY", 10, 200.0, "filled")], positions={})
    res_on = reconcile(asof=asof, store=store_on, alpaca=alpaca_on, notify_fn=lambda _: None)
    assert any(a.kind == "ledger_position_drift" for a in res_on.alerts), "control: mismatch pages"

    store_off = Store(path=str(tmp_path / "off.duckdb")).connect()
    _seed_intended(store_off, asof=asof, ticker="AAPL", target_shares=10, alpaca_order_id="ord-1")
    alpaca_off, _ = _alpaca_with_filled_orders(
        [("ord-1", "AAPL", "BUY", 10, 200.0, "filled")], positions={})
    res_off = reconcile(asof=asof, store=store_off, alpaca=alpaca_off,
                        notify_fn=lambda _: None, check_ledger_drift=False)
    assert not any(a.kind == "ledger_position_drift" for a in res_off.alerts)


def test_no_fill_drift_excludes_stop_canceled_decide_orders(tmp_path):
    """A decide order the stop sweep canceled ('canceled_by_stop') is a deliberate
    cancel, not a failed open-auction cross — it must NOT count toward the
    buy_miss_systemic alert (#9/#10, review 2026-07-04). Two canceled buys would
    otherwise read as 100% missed and page every reconcile."""
    from sma.live.reconcile import _detect_no_fill_drift

    store = _make_store(tmp_path)
    asof = date(2026, 5, 1)
    for i in range(2):
        iid = _seed_intended(store, asof=asof, ticker=f"T{i}", side="BUY",
                             alpaca_order_id=f"o{i}")
        store.conn.execute(
            "UPDATE intended_orders SET status='canceled_by_stop' "
            "WHERE intended_order_id=?", [iid],
        )
    assert _detect_no_fill_drift(asof=asof, store=store) == []


def test_record_fills_skips_unmatched_orders(tmp_path):
    """An Alpaca order with no matching intended_orders row (manual/unrelated)
    must NOT be inserted into paper_fills (contamination)."""
    asof = date(2026, 5, 1)
    store = _make_store(tmp_path)
    alpaca, _ = _alpaca_with_filled_orders([("ord-manual", "TSLA", "BUY", 5, 100.0, "filled")])
    reconcile(asof=asof, store=store, alpaca=alpaca, notify_fn=lambda _: None)
    assert store.conn.execute("SELECT COUNT(*) FROM paper_fills").fetchone()[0] == 0


def test_record_fills_finds_order_submitted_on_different_day_than_asof(tmp_path):
    """OPG/catch-up/weekend orders submit on a DIFFERENT calendar day than their
    decide asof. Reconcile must find them by order id (via intended_orders), not
    a submission-date window — which returned the wrong orders and MISSED these
    (confirmed live 2026-06-08: the 6/05 fills were attributed to 6/08)."""
    store = _make_store(tmp_path)
    asof = date(2026, 6, 5)  # decide date (Friday)
    _seed_intended(store, asof=asof, ticker="AAPL", target_shares=10,
                   alpaca_order_id="ord-monday")
    alpaca, tc = _alpaca_with_filled_orders([
        ("ord-monday", "AAPL", "BUY", 10, 200.0, "filled"),
    ])
    # A submission-date window on the asof (Friday) returns NOTHING — the order
    # executed the next session. The old code recorded 0 fills here.
    tc.get_orders.return_value = []

    result = reconcile(asof=asof, store=store, alpaca=alpaca, notify_fn=lambda _: None)

    assert result.fills_recorded == 1
    row = store.conn.execute(
        "SELECT ticker, filled_shares, asof_date FROM paper_fills"
    ).fetchone()
    assert row == ("AAPL", 10, asof)


def test_reconcile_records_stop_loss_fills_not_just_decide(tmp_path):
    """A source='stop-loss' SELL fill must be recorded in paper_fills. Before the
    fix _record_fills filtered source='decide' only, so a price-exit sell was
    never recorded — the trailing-stop peak reconstruction never saw the exit and
    the ledger-drift detector fired spuriously after any stop (review 2026-07-04)."""
    store = _make_store(tmp_path)
    asof = date(2026, 5, 1)
    _seed_intended(store, asof=asof, ticker="NVDA", side="SELL", target_shares=10,
                   alpaca_order_id="ord-sl", source="stop-loss")
    alpaca, _ = _alpaca_with_filled_orders(
        [("ord-sl", "NVDA", "SELL", 10, 150.0, "filled")], positions={},
    )
    result = reconcile(asof=asof, store=store, alpaca=alpaca, notify_fn=lambda _: None)
    assert result.fills_recorded == 1
    rows = store.conn.execute(
        "SELECT alpaca_order_id, ticker, side, filled_shares FROM paper_fills"
    ).fetchall()
    assert rows == [("ord-sl", "NVDA", "SELL", 10)]


def test_backfill_recovers_stop_loss_order_via_source_aware_coid(tmp_path):
    """A crash-after-accept stop-loss order (alpaca_order_id=NULL) must be
    recovered via client_order_id(asof, ticker, 'SELL', source='stop-loss'): the
    stop-loss coid embeds the source, so the decide-form coid would never match."""
    from sma.live.orders import client_order_id

    store = _make_store(tmp_path)
    asof = date(2026, 5, 1)
    _seed_intended(store, asof=asof, ticker="NVDA", side="SELL", target_shares=10,
                   alpaca_order_id=None, source="stop-loss")
    alpaca, tc = _alpaca_with_filled_orders(
        [("ord-sl", "NVDA", "SELL", 10, 150.0, "filled")], positions={},
    )
    expected_coid = client_order_id(asof, "NVDA", "SELL", source="stop-loss")
    recovered = MagicMock()
    recovered.id = "ord-sl"
    recovered.status = "filled"
    tc.get_order_by_client_id.side_effect = (
        lambda coid: recovered if coid == expected_coid else None
    )

    result = reconcile(asof=asof, store=store, alpaca=alpaca, notify_fn=lambda _: None)
    assert result.fills_recorded == 1
    assert store.conn.execute(
        "SELECT alpaca_order_id FROM intended_orders WHERE source='stop-loss'"
    ).fetchone()[0] == "ord-sl"


def test_record_fills_skips_malformed_order_without_aborting_batch(tmp_path):
    """A single broker order with an unparseable payload (filled_qty='NaN') must
    be logged and skipped — NOT crash the whole reconcile and lose every other
    fill for the batch (Codex review of abca439)."""
    store = _make_store(tmp_path)
    asof = date(2026, 5, 1)
    _seed_intended(store, asof=asof, ticker="BAD", target_shares=1,
                   alpaca_order_id="ord-bad")
    _seed_intended(store, asof=asof, ticker="GOOD", target_shares=2,
                   alpaca_order_id="ord-good")

    bad = MagicMock()
    bad.id, bad.symbol, bad.side = "ord-bad", "BAD", "BUY"
    bad.filled_qty, bad.filled_avg_price, bad.status = "NaN", "x", "filled"
    good = MagicMock()
    good.id, good.symbol, good.side = "ord-good", "GOOD", "BUY"
    good.filled_qty, good.filled_avg_price, good.status = "2", "100.0", "filled"
    good.commission = good.fees = 0
    good.submitted_at = datetime(2026, 5, 1, 18, 35)
    good.filled_at = datetime(2026, 5, 2, 9, 30)

    alpaca, tc = _alpaca_with_filled_orders([])
    tc.get_order_by_id.side_effect = lambda oid: {"ord-bad": bad, "ord-good": good}[oid]

    result = reconcile(asof=asof, store=store, alpaca=alpaca, notify_fn=lambda _: None)

    assert result.fills_recorded == 1  # GOOD recorded; BAD skipped, not crashed
    tickers = [r[0] for r in store.conn.execute(
        "SELECT ticker FROM paper_fills").fetchall()]
    assert tickers == ["GOOD"]


def test_reconcile_updates_intended_status_to_filled(tmp_path):
    """Reconcile must update intended_orders.status from the broker — a filled
    order should no longer be stuck at 'submitted' forever (Problem 2)."""
    store = _make_store(tmp_path)
    asof = date(2026, 5, 1)
    iid = _seed_intended(store, asof=asof, ticker="AAPL", target_shares=10,
                         alpaca_order_id="ord-1")
    alpaca, _ = _alpaca_with_filled_orders([("ord-1", "AAPL", "BUY", 10, 200.0, "filled")])

    reconcile(asof=asof, store=store, alpaca=alpaca, notify_fn=lambda _: None)

    status = store.conn.execute(
        "SELECT status FROM intended_orders WHERE intended_order_id=?", [iid]
    ).fetchone()[0]
    assert status == "filled"


def test_reconcile_marks_expired_buys_and_still_detects_no_fill_drift(tmp_path):
    """A buy that EXPIRED unfilled: status becomes 'expired', AND no-fill drift
    still fires. Drift keys off the order being placed (alpaca_order_id set) +
    having no paper_fill — NOT status='submitted' — so updating status to a
    terminal value must not silence the missed-buy alert."""
    store = _make_store(tmp_path)
    asof = date(2026, 5, 1)
    for i in range(5):
        _seed_intended(store, asof=asof, ticker=f"T{i}", side="BUY",
                       alpaca_order_id=f"ord-{i}")
    expired = [(f"ord-{i}", f"T{i}", "BUY", 0, 0.0, "expired") for i in range(5)]
    alpaca, _ = _alpaca_with_filled_orders(expired)

    result = reconcile(asof=asof, store=store, alpaca=alpaca, notify_fn=lambda _: None)

    statuses = [r[0] for r in store.conn.execute(
        "SELECT DISTINCT status FROM intended_orders WHERE asof_date=?", [asof]
    ).fetchall()]
    assert statuses == ["expired"]
    assert "buy_miss_systemic" in [a.kind for a in result.alerts]


def test_reconcile_recovers_null_alpaca_order_id_via_client_order_id(tmp_path):
    """Crash-after-accept: a submitted row with alpaca_order_id=NULL (the broker
    accepted the order but the process died before the id was written) is
    recovered by its deterministic client_order_id so its fill is still recorded
    instead of being skipped forever (Codex d)."""
    store = _make_store(tmp_path)
    asof = date(2026, 5, 1)
    iid = _seed_intended(store, asof=asof, ticker="AAPL", side="BUY",
                         target_shares=10, alpaca_order_id=None)
    alpaca, tc = _alpaca_with_filled_orders([
        ("ord-recovered", "AAPL", "BUY", 10, 200.0, "filled"),
    ])
    recovered = MagicMock()
    recovered.id, recovered.status = "ord-recovered", "filled"
    tc.get_order_by_client_id.return_value = recovered

    result = reconcile(asof=asof, store=store, alpaca=alpaca, notify_fn=lambda _: None)

    assert result.fills_recorded == 1
    aid = store.conn.execute(
        "SELECT alpaca_order_id FROM intended_orders WHERE intended_order_id=?", [iid]
    ).fetchone()[0]
    assert aid == "ord-recovered"


def test_reconcile_alerts_when_snapshot_write_fails(tmp_path, monkeypatch):
    """A failed account-snapshot write must ALERT (catastrophic-loss detection
    relies on it) and must NOT silently proceed to a (stale) catastrophic check."""
    import sma.live.reconcile as rc

    asof = date(2026, 5, 1)
    store = _make_store(tmp_path)
    alpaca, _ = _alpaca_with_filled_orders([])
    monkeypatch.setattr(rc, "_write_account_snapshot", lambda **kw: (False, None))
    notified: list[str] = []
    result = reconcile(asof=asof, store=store, alpaca=alpaca, notify_fn=notified.append)
    assert result.snapshot_written is False
    kinds = {a.kind for a in result.alerts}
    assert "snapshot_failed" in kinds
    assert "catastrophic_loss" not in kinds
    # no book to share when the snapshot failed -> ledger drift check is skipped
    assert "ledger_position_drift" not in kinds
    assert any("snapshot" in m.lower() for m in notified)


# ---- pre-close snapshot guard (2026-08-20) --------------------------------
#
# 2026-08-20 live bug: a launchd catch-up ran reconcile at 05:36 ET, before
# that day's session had even opened. get_account() at that hour prices off
# the PRIOR session's after-hours mark, so the row written under
# asof_date=2026-08-20 was actually Wednesday-night marks ($121,930.99 vs
# live $121,534) mislabeled as Thursday's close.


def test_reconcile_skips_same_day_snapshot_before_session_close(tmp_path, caplog):
    """Pre-open run: session_window says the close hasn't happened yet ->
    write NO row at all, snapshot_written=False, and -- critically -- no
    'snapshot_failed' alert (this is an intentional skip, not a failure).
    Also logs one explanatory INFO line, per the fix's contract."""
    import logging
    caplog.set_level(logging.INFO, logger="sma.live.reconcile")

    store = _make_store(tmp_path)
    asof = date(2026, 8, 19)
    alpaca, _ = _alpaca_with_filled_orders([], equity=121_930.99)
    alpaca.session_window = MagicMock(return_value=(
        datetime(2026, 8, 20, 9, 30, tzinfo=ET), datetime(2026, 8, 20, 16, 0, tzinfo=ET),
    ))
    now = datetime(2026, 8, 20, 5, 36, tzinfo=ET)  # pre-open catch-up

    notified: list[str] = []
    result = reconcile(asof=asof, store=store, alpaca=alpaca,
                       notify_fn=notified.append, now=now)

    assert result.snapshot_written is False
    kinds = {a.kind for a in result.alerts}
    assert "snapshot_failed" not in kinds, (
        "an intentional pre-close skip must not page as a write failure"
    )
    assert notified == [], "a pre-close skip must not notify at all"
    row = store.conn.execute(
        "SELECT 1 FROM account_snapshots WHERE asof_date = ?", [date(2026, 8, 20)],
    ).fetchone()
    assert row is None, "no phantom row for a session that hasn't closed yet"
    assert any(
        "skipping account_snapshots write" in r.message for r in caplog.records
    ), "must log one explanatory INFO line"


def test_reconcile_writes_same_day_snapshot_after_session_close(tmp_path):
    """The normal 16:30 ET case: session_window says the close has happened ->
    write proceeds exactly as before this guard existed."""
    store = _make_store(tmp_path)
    asof = date(2026, 8, 19)
    alpaca, _ = _alpaca_with_filled_orders([], equity=121_534.0)
    alpaca.session_window = MagicMock(return_value=(
        datetime(2026, 8, 20, 9, 30, tzinfo=ET), datetime(2026, 8, 20, 16, 0, tzinfo=ET),
    ))
    # 16:05 -- past the 16:00 close (guard passes) but before the 16:10
    # after-hours-equity cutoff, so this stays on the plain get_account path
    # (portfolio-history sourcing is covered by test_snapshot_equity_source.py).
    now = datetime(2026, 8, 20, 16, 5, tzinfo=ET)

    result = reconcile(asof=asof, store=store, alpaca=alpaca,
                       notify_fn=lambda _: None, now=now)

    assert result.snapshot_written is True
    row = store.conn.execute(
        "SELECT equity FROM account_snapshots WHERE asof_date = ?", [date(2026, 8, 20)],
    ).fetchone()
    assert row is not None
    assert row[0] == 121_534.0


def test_reconcile_writes_half_day_snapshot_after_1300_close(tmp_path):
    """Half-days close at 13:00, not 16:00 -- a 14:00 run must write, using
    the REAL session close from session_window rather than a hardcoded hour."""
    store = _make_store(tmp_path)
    asof = date(2026, 11, 25)
    half_day = date(2026, 11, 27)  # Thanksgiving half-day
    alpaca, _ = _alpaca_with_filled_orders([], equity=110_000.0)
    alpaca.session_window = MagicMock(return_value=(
        datetime(2026, 11, 27, 9, 30, tzinfo=ET), datetime(2026, 11, 27, 13, 0, tzinfo=ET),
    ))
    now = datetime(2026, 11, 27, 14, 0, tzinfo=ET)

    result = reconcile(asof=asof, store=store, alpaca=alpaca,
                       notify_fn=lambda _: None, now=now)

    assert result.snapshot_written is True
    row = store.conn.execute(
        "SELECT 1 FROM account_snapshots WHERE asof_date = ?", [half_day],
    ).fetchone()
    assert row is not None


def test_reconcile_skips_snapshot_on_non_session_day(tmp_path):
    """session_window returns None (weekend/holiday) -> skip entirely, never
    write a row for a day that was never a trading session."""
    store = _make_store(tmp_path)
    asof = date(2026, 8, 21)
    saturday = date(2026, 8, 22)
    alpaca, _ = _alpaca_with_filled_orders([], equity=100_000.0)
    alpaca.session_window = MagicMock(return_value=None)
    now = datetime(2026, 8, 22, 12, 0, tzinfo=ET)

    result = reconcile(asof=asof, store=store, alpaca=alpaca,
                       notify_fn=lambda _: None, now=now)

    assert result.snapshot_written is False
    row = store.conn.execute(
        "SELECT 1 FROM account_snapshots WHERE asof_date = ?", [saturday],
    ).fetchone()
    assert row is None


# ---- ledger-vs-broker drift (2026-06-24 audit) ----------------------------


def _seed_fill(store, *, ticker, side, shares, asof=date(2026, 5, 1), price=100.0):
    """Insert a single paper_fills row (a recorded fill)."""
    rid = store.allocate_run_id()
    oid = f"f-{uuid.uuid4()}"
    store.conn.execute("""
        INSERT INTO paper_fills
        (alpaca_order_id, intended_order_id, asof_date, ticker, side,
         filled_shares, fill_price, commission, fees, status, submitted_at,
         filled_at, run_id)
        VALUES (?, NULL, ?, ?, ?, ?, ?, 0, 0, 'filled', ?, ?, ?)
    """, [oid, asof, ticker, side, shares, price,
          datetime(2026, 5, 1, 18, 35), datetime(2026, 5, 2, 9, 30), rid])


def _alpaca_with_positions(positions):
    """positions: {symbol: shares} -> AlpacaClient whose book is exactly that."""
    tc = MagicMock(spec=TradingClient)
    objs = []
    for sym, qty in positions.items():
        p = MagicMock()
        p.symbol = sym
        p.qty = str(qty)
        p.avg_entry_price = "100.0"
        objs.append(p)
    tc.get_all_positions.return_value = objs
    return AlpacaClient(trading_client=tc)


def test_ledger_drift_flags_only_mismatched_names(tmp_path):
    from sma.live.reconcile import _detect_ledger_position_drift

    store = _make_store(tmp_path)
    # clean held: AAPL ledger 25 == book 25
    _seed_fill(store, ticker="AAPL", side="BUY", shares=25)
    # clean closed: MSFT ledger 0 (10 buy, 10 sell), absent from the book
    _seed_fill(store, ticker="MSFT", side="BUY", shares=10)
    _seed_fill(store, ticker="MSFT", side="SELL", shares=10)
    # drift mirroring the real 6/24 finding:
    _seed_fill(store, ticker="COIN", side="BUY", shares=36)   # 36 vs book 0
    _seed_fill(store, ticker="CRWD", side="SELL", shares=5)   # -5 vs book 0 (impossible short)
    _seed_fill(store, ticker="HOOD", side="BUY", shares=113)  # 113 vs book 102
    alpaca = _alpaca_with_positions({"AAPL": 25, "HOOD": 102})

    alerts = _detect_ledger_position_drift(store=store, book=alpaca.get_positions())

    assert len(alerts) == 1
    a = alerts[0]
    assert a.kind == "ledger_position_drift"
    for name in ("COIN", "CRWD", "HOOD"):
        assert name in a.detail
    assert "AAPL" not in a.detail and "MSFT" not in a.detail
    # the negative-ledger bug surfaces with its sign
    assert "-5" in a.detail


def test_ledger_drift_clean_book_no_alert(tmp_path):
    from sma.live.reconcile import _detect_ledger_position_drift

    store = _make_store(tmp_path)
    _seed_fill(store, ticker="AAPL", side="BUY", shares=25)
    _seed_fill(store, ticker="NVDA", side="BUY", shares=10)
    _seed_fill(store, ticker="NVDA", side="SELL", shares=4)  # net 6
    alpaca = _alpaca_with_positions({"AAPL": 25, "NVDA": 6})

    assert _detect_ledger_position_drift(store=store, book=alpaca.get_positions()) == []


def test_reconcile_wires_ledger_drift_alert(tmp_path):
    asof = date(2026, 5, 1)
    store = _make_store(tmp_path)
    _seed_fill(store, ticker="COIN", side="BUY", shares=36, asof=asof)
    # broker book has zero positions -> COIN 36 vs 0 must alert
    alpaca, _ = _alpaca_with_filled_orders([], position_count=0)
    result = reconcile(asof=asof, store=store, alpaca=alpaca, notify_fn=lambda _: None)
    assert "ledger_position_drift" in {a.kind for a in result.alerts}


def test_reconcile_fires_trade_push_mismatch_after_open(tmp_path):
    """Push claimed 8.0% of equity for a $2,000 (2.0%-of-equity) AAPL BUY --
    an LLY-shaped bug (claimed weight far from the order's real notional).
    Reconcile must surface it as a trade_push_mismatch alert once the fill
    has had a chance to book (past the next session's open auction)."""
    eastern = ZoneInfo("America/New_York")

    store = _make_store(tmp_path)
    asof = date(2026, 4, 27)  # Mon
    _seed_intended(store, asof=asof, ticker="AAPL", side="BUY",
                   target_shares=10, alpaca_order_id="ord-aapl")
    alpaca, tc = _alpaca_with_filled_orders(
        [("ord-aapl", "AAPL", "BUY", 10, 200.0, "filled")], equity=100_000.0,
    )
    next_session = MagicMock()
    next_session.date = date(2026, 4, 28)  # Tue
    tc.get_calendar.return_value = [next_session]

    record_trade_push(
        asof=asof, title="t", message="m",
        orders=[{
            "ticker": "AAPL", "side": "BUY", "shares": 10,
            "decide_price": 200.0, "order_notional": 2_000.0,
            "order_pct_of_equity": 0.08,  # bug: real notional is 2.0%, not 8.0%
            "full_exit": False,
        }],
        equity=100_000.0, delivered=True,
    )

    now_et = datetime(2026, 4, 28, 16, 30, tzinfo=eastern)
    notified: list[str] = []
    result = reconcile(asof=asof, store=store, alpaca=alpaca,
                       notify_fn=notified.append, now=now_et)

    kinds = [a.kind for a in result.alerts]
    assert "trade_push_mismatch" in kinds
    assert any("AAPL" in m for m in notified)


def test_reconcile_silent_when_push_file_missing(tmp_path):
    """Same setup, minus the record_trade_push call -- no trade_push_mismatch
    alert. Best-effort check: no persisted push means nothing to verify."""
    eastern = ZoneInfo("America/New_York")

    store = _make_store(tmp_path)
    asof = date(2026, 4, 27)
    _seed_intended(store, asof=asof, ticker="AAPL", side="BUY",
                   target_shares=10, alpaca_order_id="ord-aapl2")
    alpaca, tc = _alpaca_with_filled_orders(
        [("ord-aapl2", "AAPL", "BUY", 10, 200.0, "filled")], equity=100_000.0,
    )
    next_session = MagicMock()
    next_session.date = date(2026, 4, 28)
    tc.get_calendar.return_value = [next_session]

    now_et = datetime(2026, 4, 28, 16, 30, tzinfo=eastern)
    result = reconcile(asof=asof, store=store, alpaca=alpaca,
                       notify_fn=lambda _: None, now=now_et)

    kinds = [a.kind for a in result.alerts]
    assert "trade_push_mismatch" not in kinds


def test_reconcile_does_not_fall_back_to_prior_sessions_push(tmp_path):
    """Regression lock (2026-09-11 investigation): a bug report proposed
    verifying `asof`'s fills against the PREVIOUS trading session's push
    instead of `asof`'s own push, on the theory that `asof` means "today"
    rather than "the decide day being reconciled". That theory is wrong --
    `asof` IS the decide day (see reconcile.py's module docstring), and
    push(asof) is the SAME batch as intended_orders(asof_date=asof) by
    construction (both written by the one decide_once(asof=asof) call).

    Prove it directly: `asof`'s OWN push is deliberately wrong (LLY-shaped:
    claims 8% of equity for a real 2%-of-equity fill), while the PRIOR
    session has a push that would look like a clean match for these same
    fills if it were (wrongly) the one consulted. reconcile(asof=...) must
    still fire the mismatch -- if it silently passed, that would mean it had
    fallen back to the prior session's push instead of asof's own."""
    eastern = ZoneInfo("America/New_York")

    store = _make_store(tmp_path)
    asof = date(2026, 4, 27)          # decide day being reconciled
    prior_session = date(2026, 4, 24)  # the session strictly before `asof`

    _seed_intended(store, asof=asof, ticker="AAPL", side="BUY",
                   target_shares=10, alpaca_order_id="ord-aapl-noFallback")
    alpaca, tc = _alpaca_with_filled_orders(
        [("ord-aapl-noFallback", "AAPL", "BUY", 10, 200.0, "filled")],
        equity=100_000.0,
    )
    next_session = MagicMock()
    next_session.date = date(2026, 4, 28)
    tc.get_calendar.return_value = [next_session]

    # The PRIOR session's push: a totally different, unrelated AAPL order
    # sized so it reads as a clean match against `asof`'s real $2,000 fill --
    # if the detector ever consulted this instead of asof's own push, the
    # mismatch below would be masked.
    record_trade_push(
        asof=prior_session, title="t", message="m",
        orders=[{
            "ticker": "AAPL", "side": "BUY", "shares": 10,
            "decide_price": 200.0, "order_notional": 2_000.0,
            "order_pct_of_equity": 0.02,  # matches the real fill -- a decoy
            "full_exit": False,
        }],
        equity=100_000.0, delivered=True,
    )
    # `asof`'s OWN push: the actual bug shape (claims 8%, real notional 2%).
    record_trade_push(
        asof=asof, title="t", message="m",
        orders=[{
            "ticker": "AAPL", "side": "BUY", "shares": 10,
            "decide_price": 200.0, "order_notional": 2_000.0,
            "order_pct_of_equity": 0.08,
            "full_exit": False,
        }],
        equity=100_000.0, delivered=True,
    )

    now_et = datetime(2026, 4, 28, 16, 30, tzinfo=eastern)
    notified: list[str] = []
    result = reconcile(asof=asof, store=store, alpaca=alpaca,
                       notify_fn=notified.append, now=now_et)

    kinds = [a.kind for a in result.alerts]
    assert "trade_push_mismatch" in kinds
    assert any("AAPL" in m for m in notified)


def test_reconcile_catch_up_two_batches_each_verify_own_push(tmp_path):
    """Mirrors sma.live.__main__'s catch-up loop: multiple missed decide
    batches get reconciled in the same run, one reconcile() call per asof
    (the `reconciled_asofs` list). Each batch's push must be checked against
    THAT batch's own fills -- session 1's deliberate mismatch must not leak
    into session 2's clean result."""
    eastern = ZoneInfo("America/New_York")

    store = _make_store(tmp_path)
    asof1 = date(2026, 4, 27)  # Mon -- mismatched push
    asof2 = date(2026, 4, 28)  # Tue -- clean push

    _seed_intended(store, asof=asof1, ticker="INTC", side="SELL",
                   target_shares=20, last_price=100.32,
                   alpaca_order_id="ord-intc-catchup")
    _seed_intended(store, asof=asof2, ticker="PWR", side="BUY",
                   target_shares=2, last_price=618.73,
                   alpaca_order_id="ord-pwr-catchup")
    alpaca, tc = _alpaca_with_filled_orders(
        [
            ("ord-intc-catchup", "INTC", "SELL", 20, 102.0, "filled"),
            ("ord-pwr-catchup", "PWR", "BUY", 2, 627.24, "filled"),
        ],
        equity=100_000.0,
    )
    # Both batches' order_drift_open gates just need to be open by `now_et`
    # below; the mock's next-session date is shared across both asof calls
    # (AlpacaClient.next_session_date ignores which asof it was asked about
    # here), same simplification the rest of this file's fixture uses.
    next_session = MagicMock()
    next_session.date = date(2026, 4, 29)
    tc.get_calendar.return_value = [next_session]

    record_trade_push(
        asof=asof1, title="t", message="m",
        orders=[{
            "ticker": "INTC", "side": "SELL", "shares": 20,
            "decide_price": 100.32, "order_notional": 2006.4,
            "order_pct_of_equity": 0.50,  # deliberately wrong (LLY-shaped)
            "full_exit": False,
        }],
        equity=100_000.0, delivered=True,
    )
    record_trade_push(
        asof=asof2, title="t", message="m",
        orders=[{
            "ticker": "PWR", "side": "BUY", "shares": 2,
            "decide_price": 618.73, "order_notional": 1237.46,
            "order_pct_of_equity": 1237.46 / 100_000.0,  # accurate
            "full_exit": False,
        }],
        equity=100_000.0, delivered=True,
    )

    now_et = datetime(2026, 4, 29, 16, 30, tzinfo=eastern)
    result1 = reconcile(asof=asof1, store=store, alpaca=alpaca,
                        notify_fn=lambda _: None, now=now_et,
                        check_ledger_drift=False)
    result2 = reconcile(asof=asof2, store=store, alpaca=alpaca,
                        notify_fn=lambda _: None, now=now_et,
                        check_ledger_drift=False)

    kinds1 = [a.kind for a in result1.alerts]
    kinds2 = [a.kind for a in result2.alerts]
    assert "trade_push_mismatch" in kinds1
    assert any("INTC" in a.detail for a in result1.alerts
               if a.kind == "trade_push_mismatch")
    assert "trade_push_mismatch" not in kinds2
