"""reconcile_cmd: default --asof-date resolves to the most recent decide date.

Bug B: prior to this fix, reconcile defaulted to date.today(). But
intended_orders.asof_date is the *decide* date, which is the prior trading
session from reconcile's POV (decide runs evening N, fill happens morning N+1,
reconcile fires evening N+1 with asof defaulting to N+1 — finding no
intended_orders dated N+1 and silently doing nothing).

The fix: when --asof-date is omitted, use MAX(asof_date) over intended_orders
that are 'submitted', so the canary flow (decide tonight, reconcile tomorrow)
finds yesterday's submitted orders.
"""

import uuid
from datetime import date

from sma.ingest.store import Store
from sma.live.__main__ import _resolve_reconcile_asofs


def _make_store(tmp_path):
    return Store(path=str(tmp_path / "test.duckdb")).connect()


def _seed(store, *, asof: date, ticker: str, status: str = "submitted",
          alpaca_order_id: str | None = "ord-1"):
    rid = store.allocate_run_id()
    iid = str(uuid.uuid4())
    store.conn.execute("""
        INSERT INTO intended_orders
        (intended_order_id, asof_date, ticker, side, target_shares,
         target_weight, last_price, source, alpaca_order_id, status, run_id)
        VALUES (?, ?, ?, 'BUY', 10, NULL, 100.0, 'decide', ?, ?, ?)
    """, [iid, asof, ticker, alpaca_order_id, status, rid])


def _seed_stop_loss(store, *, asof: date, ticker: str, status: str = "submitted",
                    alpaca_order_id: str | None = "ord-sl"):
    rid = store.allocate_run_id()
    iid = str(uuid.uuid4())
    store.conn.execute("""
        INSERT INTO intended_orders
        (intended_order_id, asof_date, ticker, side, target_shares,
         target_weight, last_price, source, alpaca_order_id, status, run_id)
        VALUES (?, ?, ?, 'SELL', 10, NULL, 150.0, 'stop-loss', ?, ?, ?)
    """, [iid, asof, ticker, alpaca_order_id, status, rid])


def test_default_asof_discovers_stop_loss_only_batch(tmp_path):
    """A day a stop fired but decide placed ZERO orders (all deltas 0) must still
    be reconciled, or the stop SELL fill is orphaned forever and the ledger-drift
    detector pages every reconcile (adversarial review 2026-07-04). The resolver
    must discover source='stop-loss' batches, not just source='decide'."""
    store = _make_store(tmp_path)
    _seed_stop_loss(store, asof=date(2026, 5, 1), ticker="NVDA")
    resolved = _resolve_reconcile_asofs(None, store)
    assert date(2026, 5, 1) in resolved


def test_explicit_asof_date_passes_through(monkeypatch, tmp_path):
    import sma.live.__main__ as _live_main
    monkeypatch.setattr(_live_main, "_today_et", lambda: date(2026, 5, 1))
    store = _make_store(tmp_path)
    _seed(store, asof=date(2026, 4, 27), ticker="AAPL")
    resolved = _resolve_reconcile_asofs("2026-05-01", store)
    assert resolved == [date(2026, 5, 1)]


def test_explicit_future_asof_is_rejected(monkeypatch, tmp_path):
    """Codex MED 1: an explicit --asof-date in the future (typo or operator
    error) would be silently accepted and write future-dated state. Raise."""
    import click

    import sma.live.__main__ as _live_main
    monkeypatch.setattr(_live_main, "_today_et", lambda: date(2026, 5, 1))
    store = _make_store(tmp_path)

    try:
        _resolve_reconcile_asofs("2026-05-10", store)
    except click.ClickException as e:
        assert "future" in str(e).lower() or "today" in str(e).lower()
    else:
        raise AssertionError("expected ClickException for future asof_date")


def test_default_asof_returns_none_when_no_intended_orders(monkeypatch, tmp_path):
    """With nothing submitted, there's nothing to reconcile. Returning today
    would write a phantom reconcile-sentinel for today's date before today's
    20:00 ET decide has even fired, stranding today's batch from future
    reconciliation forever. None is the correct sentinel for 'no-op'."""
    import sma.live.__main__ as _live_main
    fixed_today = date(2026, 5, 1)
    monkeypatch.setattr(_live_main, "_today_et", lambda: fixed_today)
    store = _make_store(tmp_path)
    resolved = _resolve_reconcile_asofs(None, store)
    assert resolved == []


def test_default_asof_picks_max_submitted_intended_order_date(monkeypatch, tmp_path):
    """Smoke flow: decide ran 4/30 evening; reconcile run next day with no
    --asof-date should resolve to 4/30, not today.

    Monkeypatch SENTINEL_DIR so the resolver doesn't read real production
    sentinels (which would cause it to skip dates that have already been
    reconciled and pick a different answer)."""
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path / "sentinels"))

    store = _make_store(tmp_path)
    _seed(store, asof=date(2026, 4, 27), ticker="AAPL", alpaca_order_id="o1")
    _seed(store, asof=date(2026, 4, 28), ticker="MSFT", alpaca_order_id="o2")
    _seed(store, asof=date(2026, 4, 30), ticker="GOOG", alpaca_order_id="o3")

    resolved = _resolve_reconcile_asofs(None, store)
    # 2026-07-01: ALL unreconciled batches, OLDEST first — the old newest-only
    # pick permanently stranded deferred/outage batches behind newer ones.
    assert resolved == [date(2026, 4, 27), date(2026, 4, 28), date(2026, 4, 30)]


def test_default_asof_includes_failed_submission_batches(tmp_path):
    """2026-07-01 (reversal of the old assumption): a 'submission_failed' row
    CAN be live at the broker — a submit that times out on the response may
    have been accepted (the coid backfill resolves it authoritatively). Such
    batches must therefore be reconcile candidates, oldest first."""
    store = _make_store(tmp_path)
    _seed(store, asof=date(2026, 4, 28), ticker="AAPL",
          status="submitted", alpaca_order_id="o1")
    _seed(store, asof=date(2026, 4, 30), ticker="MSFT",
          status="submission_failed", alpaca_order_id=None)

    resolved = _resolve_reconcile_asofs(None, store)
    assert resolved == [date(2026, 4, 28), date(2026, 4, 30)]


def test_default_asof_skips_already_reconciled_dates(monkeypatch, tmp_path):
    """Codex HIGH 1: prior to this fix, the default kept picking the same
    submitted asof every run, re-recording fills and overwriting the
    historical account snapshot under that asof. Skip any decide_date that
    already has a reconcile sentinel — pick the next-newest unreconciled
    one instead."""
    import sma.sentinels as _sentinels
    sentinel_dir = tmp_path / "sentinels"
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(sentinel_dir))

    store = _make_store(tmp_path)
    _seed(store, asof=date(2026, 4, 27), ticker="AAPL", alpaca_order_id="o1")
    _seed(store, asof=date(2026, 4, 30), ticker="MSFT", alpaca_order_id="o2")
    # 4/30 already reconciled
    _sentinels.write_sentinel(
        label="com.sma.live.reconcile.daily",
        asof=date(2026, 4, 30),
        payload={"label": "com.sma.live.reconcile.daily",
                 "asof": "2026-04-30",
                 "completed_at": "2026-05-01T13:30:00Z"},
    )

    resolved = _resolve_reconcile_asofs(None, store)
    assert resolved == [date(2026, 4, 27)], (
        "sentineled dates are skipped; the unreconciled one is returned"
    )


def test_default_asof_returns_none_when_all_reconciled(monkeypatch, tmp_path):
    """If every submitted decide_date has a reconcile sentinel, the daily
    cron has nothing new to reconcile and the resolver returns None. The
    caller (reconcile_cmd) is then expected to write today's account snapshot
    and skip the reconcile-sentinel write — writing one for today before
    today's 20:00 decide fires is the phantom-sentinel bug that strands
    today's batch from future reconciliation.

    Empirical confirmation (2026-05-21 audit): 38 paper_fills were missing
    across 5/7–5/21 because the prior fallback-to-today resolution wrote
    phantom sentinels covering every weekday."""
    import sma.live.__main__ as _live_main
    import sma.sentinels as _sentinels
    sentinel_dir = tmp_path / "sentinels"
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(sentinel_dir))

    fixed_today = date(2099, 12, 31)
    monkeypatch.setattr(_live_main, "_today_et", lambda: fixed_today)

    store = _make_store(tmp_path)
    _seed(store, asof=date(2026, 4, 30), ticker="AAPL", alpaca_order_id="o1")
    _sentinels.write_sentinel(
        label="com.sma.live.reconcile.daily",
        asof=date(2026, 4, 30),
        payload={"label": "com.sma.live.reconcile.daily",
                 "asof": "2026-04-30",
                 "completed_at": "2026-05-01T13:30:00Z"},
    )

    resolved = _resolve_reconcile_asofs(None, store)
    assert resolved == []
