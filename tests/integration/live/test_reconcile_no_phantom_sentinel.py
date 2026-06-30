"""Regression: reconcile_cmd must not write a phantom reconcile-sentinel for
today when there are no unreconciled batches.

Empirical 2026-05-21 audit: 38 paper_fills missing across 5/7–5/21 because
each weekday's 16:30 ET reconcile (running before that night's 20:00 decide)
fell through to `today` and wrote a sentinel for today. Then the NEXT day's
reconcile saw the prior day's now-submitted batch but with a sentinel already
present, fell through to today AGAIN, and never recorded the fills.

The fix: when nothing is unreconciled, the cron writes today's account snapshot
and exits without touching the reconcile-sentinel directory.
"""

from __future__ import annotations

import uuid
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

from sma.ingest.store import Store
from sma.live.__main__ import reconcile_cmd


@pytest.fixture
def isolated_sentinels(monkeypatch, tmp_path):
    """Point sentinels at a temp directory."""
    sentinel_dir = tmp_path / "sentinels"
    sentinel_dir.mkdir()
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(sentinel_dir))
    return sentinel_dir


def _open_order_mock():
    """An accepted-but-unfilled order (the next open hasn't occurred yet).
    Reconcile fetches each decide order by id; this records no fill."""
    o = MagicMock()
    o.filled_qty = "0"
    o.filled_avg_price = None
    o.status = "accepted"
    return o


def _stub_settings_and_alpaca(monkeypatch, *, equity: float = 100_000.0) -> MagicMock:
    """Replace load_settings + _build_alpaca on the live.__main__ module so
    reconcile_cmd skips real config/network entirely."""
    import sma.live.__main__ as _live_main

    fake = MagicMock()
    fake.get_account.return_value = {
        "equity": equity, "cash": equity / 2, "buying_power": equity,
        "long_market_value": equity / 2,
        "trading_blocked": False, "account_blocked": False,
    }
    fake.get_positions.return_value = {}
    fake.get_orders_for_date.return_value = []
    fake.get_order_by_id.return_value = _open_order_mock()
    fake.next_session_date.return_value = date(2099, 1, 1)
    fake.tc = MagicMock()

    monkeypatch.setattr(_live_main, "load_settings", lambda config_path: SimpleNamespace())
    monkeypatch.setattr(_live_main, "_build_alpaca", lambda settings: fake)
    return fake


def _empty_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "test.duckdb"
    Store(path=str(db_path)).connect().close()
    return db_path


def _db_with_pre_reconciled_batch(tmp_path: Path, *, asof: date) -> Path:
    db_path = tmp_path / "test.duckdb"
    store = Store(path=str(db_path)).connect()
    rid = store.allocate_run_id()
    store.conn.execute(
        """
        INSERT INTO intended_orders
        (intended_order_id, asof_date, ticker, side, target_shares,
         target_weight, last_price, source, alpaca_order_id, status, run_id)
        VALUES (?, ?, 'AAPL', 'BUY', 10, NULL, 200.0, 'decide',
                'alpaca-1', 'submitted', ?)
        """,
        [str(uuid.uuid4()), asof, rid],
    )
    store.close()
    return db_path


def test_no_unreconciled_batches_writes_no_sentinel(isolated_sentinels, monkeypatch, tmp_path):
    """Empty DB + nothing submitted → snapshot today, no reconcile sentinel."""
    db_path = _empty_db(tmp_path)
    _stub_settings_and_alpaca(monkeypatch)

    # config_path needs to be exists=True for click, even though load_settings is stubbed.
    config_path = tmp_path / "config.yaml"
    config_path.write_text("# stub\n")

    runner = CliRunner()
    result = runner.invoke(
        reconcile_cmd,
        ["--db", str(db_path), "--config", str(config_path)],
    )
    assert result.exit_code == 0, result.output
    assert "no unreconciled batches" in result.output

    sentinel_files = list(isolated_sentinels.glob("com.sma.live.reconcile.daily-*.json"))
    assert sentinel_files == [], (
        f"phantom sentinel(s) leaked: {[f.name for f in sentinel_files]}"
    )


def test_all_batches_already_reconciled_writes_no_sentinel(
    isolated_sentinels, monkeypatch, tmp_path,
):
    """An older batch exists with its sentinel; no new submissions → no new sentinel."""
    yesterday = date(2026, 4, 30)
    db_path = _db_with_pre_reconciled_batch(tmp_path, asof=yesterday)

    import sma.sentinels as _sentinels
    _sentinels.write_sentinel(
        label="com.sma.live.reconcile.daily",
        asof=yesterday,
        payload={"label": "com.sma.live.reconcile.daily",
                 "asof": yesterday.isoformat(),
                 "completed_at": "2026-05-01T20:30:00Z"},
    )
    pre_sentinels = {
        f.name for f in isolated_sentinels.glob("com.sma.live.reconcile.daily-*.json")
    }

    _stub_settings_and_alpaca(monkeypatch)
    config_path = tmp_path / "config.yaml"
    config_path.write_text("# stub\n")

    runner = CliRunner()
    result = runner.invoke(
        reconcile_cmd,
        ["--db", str(db_path), "--config", str(config_path)],
    )
    assert result.exit_code == 0, result.output
    assert "no unreconciled batches" in result.output

    # Only BATCH sentinels are phantom-protected; the run-date LIVENESS
    # sentinel (daily.ran-*) is written on every run by design (2026-06-09).
    post_sentinels = {
        f.name for f in isolated_sentinels.glob("com.sma.live.reconcile.daily-*.json")
    }
    assert post_sentinels == pre_sentinels, (
        f"unexpected new BATCH sentinel files: {post_sentinels - pre_sentinels}"
    )


def test_snapshot_still_written_in_no_op_path(isolated_sentinels, monkeypatch, tmp_path):
    """The no-op path must still write today's account snapshot (so the
    catastrophic-loss baseline + dashboard's equity ladder stay current)."""
    db_path = _empty_db(tmp_path)
    _stub_settings_and_alpaca(monkeypatch)
    config_path = tmp_path / "config.yaml"
    config_path.write_text("# stub\n")

    runner = CliRunner()
    result = runner.invoke(
        reconcile_cmd,
        ["--db", str(db_path), "--config", str(config_path)],
    )
    assert result.exit_code == 0, result.output

    store = Store(path=str(db_path)).connect(read_only=True)
    try:
        row = store.conn.execute(
            "SELECT COUNT(*), MAX(equity) FROM account_snapshots"
        ).fetchone()
    finally:
        store.close()
    assert row[0] == 1
    assert float(row[1]) == pytest.approx(100_000.0)


def test_sentinel_deferred_when_next_session_open_in_future(
    isolated_sentinels, monkeypatch, tmp_path,
):
    """Empirical Mon-after-holiday scenario: reconcile fires Mon 16:30 ET for
    Friday's batch; next session is Tue 09:30 ET; Mon orders are still queued
    at Alpaca. The old code wrote a 0-fills sentinel for Friday, then Tue's
    reconcile saw the sentinel and SKIPPED Friday's now-filled batch
    forever. The fix defers the sentinel write until the next session's open
    has actually occurred (order_drift_open=True)."""
    friday_decide_date = date(2026, 5, 22)
    # Submitted intended_order exists for Friday but not in paper_fills.
    db_path = _db_with_pre_reconciled_batch(tmp_path, asof=friday_decide_date)

    from sma.live.__main__ import _today_et

    fake = MagicMock()
    fake.get_account.return_value = {
        "equity": 100_000.0, "cash": 50_000.0, "buying_power": 100_000.0,
        "long_market_value": 50_000.0,
        "trading_blocked": False, "account_blocked": False,
    }
    fake.get_positions.return_value = {}
    fake.get_orders_for_date.return_value = []
    fake.get_order_by_id.return_value = _open_order_mock()
    # Next session is 2 days from today — realistic Memorial-Day-style holiday
    # gap, and within the 14-day sanity bound so the deferral path triggers
    # (rather than the calendar-failure fallback that would write the sentinel
    # anyway).
    fake.next_session_date.return_value = _today_et() + timedelta(days=2)
    fake.tc = MagicMock()

    import sma.live.__main__ as _live_main
    monkeypatch.setattr(_live_main, "load_settings", lambda config_path: SimpleNamespace())
    monkeypatch.setattr(_live_main, "_build_alpaca", lambda settings: fake)

    config_path = tmp_path / "config.yaml"
    config_path.write_text("# stub\n")

    runner = CliRunner()
    result = runner.invoke(
        reconcile_cmd,
        [
            "--asof-date", friday_decide_date.isoformat(),
            "--db", str(db_path),
            "--config", str(config_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "deferred sentinel" in result.output, (
        f"expected deferral message, got:\n{result.output}"
    )

    # Sentinel for Friday should NOT have been written.
    sentinel_files = list(isolated_sentinels.glob("com.sma.live.reconcile.daily-*.json"))
    assert sentinel_files == [], (
        f"phantom sentinel(s) leaked: {[f.name for f in sentinel_files]}"
    )

    # Snapshot still written.
    store = Store(path=str(db_path)).connect(read_only=True)
    try:
        snap_count = store.conn.execute(
            "SELECT COUNT(*) FROM account_snapshots"
        ).fetchone()[0]
    finally:
        store.close()
    assert snap_count == 1


def test_sentinel_deferred_when_next_session_implausibly_far_out(
    isolated_sentinels, monkeypatch, tmp_path,
):
    """An implausibly-far next_session (>14d: degraded SDK / stale calendar
    data) means the calendar CANNOT be trusted — same policy as a lookup
    failure (2026-06-11): DEFER the batch and page, rather than complete
    blind and risk stranding unfilled orders' fills. Retry is bounded (daily
    16:30) and the .ran liveness sentinel keeps the watchdog quiet."""
    friday_decide_date = date(2026, 5, 22)
    db_path = _db_with_pre_reconciled_batch(tmp_path, asof=friday_decide_date)

    from sma.live.__main__ import _today_et

    fake = MagicMock()
    fake.get_account.return_value = {
        "equity": 100_000.0, "cash": 50_000.0, "buying_power": 100_000.0,
        "long_market_value": 50_000.0,
        "trading_blocked": False, "account_blocked": False,
    }
    fake.get_positions.return_value = {}
    fake.get_orders_for_date.return_value = []
    fake.get_order_by_id.return_value = _open_order_mock()
    # 60 days out — implausible. Triggers the sanity bound.
    fake.next_session_date.return_value = _today_et() + timedelta(days=60)
    fake.tc = MagicMock()

    import sma.live.__main__ as _live_main
    monkeypatch.setattr(_live_main, "load_settings", lambda config_path: SimpleNamespace())
    monkeypatch.setattr(_live_main, "_build_alpaca", lambda settings: fake)

    config_path = tmp_path / "config.yaml"
    config_path.write_text("# stub\n")

    runner = CliRunner()
    result = runner.invoke(
        reconcile_cmd,
        [
            "--asof-date", friday_decide_date.isoformat(),
            "--db", str(db_path),
            "--config", str(config_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "deferred sentinel" in result.output, (
        "expected calendar-sanity fallback to write sentinel, got deferral"
    )

    sentinel_files = list(isolated_sentinels.glob("com.sma.live.reconcile.daily-*.json"))
    assert sentinel_files == [], (
        f"untrustworthy calendar must DEFER the batch sentinel, got "
        f"{[f.name for f in sentinel_files]}"
    )


def test_sentinel_deferred_when_calendar_lookup_fails(
    isolated_sentinels, monkeypatch, tmp_path,
):
    """Codex module review (2026-06-11 HIGH): on calendar-lookup failure we
    can't tell whether the open has passed — completing the batch could
    permanently strand unfilled orders' fills (premature 0-fill recording).
    DEFER the batch sentinel (daily 16:30 retry is bounded; the .ran liveness
    sentinel keeps the watchdog quiet) and page a human. The old write-anyway
    fallback predates both of those mechanisms."""
    friday_decide_date = date(2026, 5, 22)
    db_path = _db_with_pre_reconciled_batch(tmp_path, asof=friday_decide_date)

    fake = MagicMock()
    fake.get_account.return_value = {
        "equity": 100_000.0, "cash": 50_000.0, "buying_power": 100_000.0,
        "long_market_value": 50_000.0,
        "trading_blocked": False, "account_blocked": False,
    }
    fake.get_positions.return_value = {}
    fake.get_orders_for_date.return_value = []
    fake.get_order_by_id.return_value = _open_order_mock()
    # Calendar lookup itself blows up.
    fake.next_session_date.side_effect = RuntimeError("alpaca calendar unreachable")
    fake.tc = MagicMock()

    import sma.live.__main__ as _live_main
    monkeypatch.setattr(_live_main, "load_settings", lambda config_path: SimpleNamespace())
    monkeypatch.setattr(_live_main, "_build_alpaca", lambda settings: fake)

    config_path = tmp_path / "config.yaml"
    config_path.write_text("# stub\n")

    runner = CliRunner()
    result = runner.invoke(
        reconcile_cmd,
        [
            "--asof-date", friday_decide_date.isoformat(),
            "--db", str(db_path),
            "--config", str(config_path),
        ],
    )
    assert result.exit_code == 0, result.output

    sentinel_files = list(isolated_sentinels.glob("com.sma.live.reconcile.daily-*.json"))
    assert sentinel_files == [], (
        f"calendar failure must DEFER the batch sentinel, got {[f.name for f in sentinel_files]}"
    )
    assert "deferred" in result.output.lower()


def test_sentinel_deferred_when_an_order_fetch_fails(
    isolated_sentinels, monkeypatch, tmp_path,
):
    """Codex module review (2026-06-11 HIGH): one 503/malformed order fetch is
    skipped (correct — don't lose the other fills) but the batch then completed
    and the missing fill was NEVER retried. The batch sentinel must be
    deferred so tomorrow's 16:30 run re-records the missing order."""
    asof = date(2026, 4, 29)
    db_path = tmp_path / "test.duckdb"
    store = Store(path=str(db_path)).connect()
    rid = store.allocate_run_id()
    for i, oid in enumerate(["alpaca-ok", "alpaca-broken"]):
        store.conn.execute(
            """
            INSERT INTO intended_orders
            (intended_order_id, asof_date, ticker, side, target_shares,
             target_weight, last_price, source, alpaca_order_id, status, run_id)
            VALUES (?, ?, ?, 'BUY', 10, NULL, 200.0, 'decide', ?, 'submitted', ?)
            """,
            [str(uuid.uuid4()), asof, f"TK{i}", oid, rid],
        )
    store.close()

    filled = MagicMock()
    filled.status = "filled"
    filled.filled_qty = "10"
    filled.filled_avg_price = "200.0"
    filled.symbol, filled.side = "TK0", "buy"
    from datetime import datetime as _dt
    filled.id = "alpaca-ok"
    filled.submitted_at = _dt(2026, 4, 29, 20, 0)
    filled.filled_at = _dt(2026, 4, 30, 9, 30)
    filled.commission, filled.fees = 0, 0

    def _by_id(oid):
        if oid == "alpaca-ok":
            return filled
        raise RuntimeError("503 from alpaca")

    fake = MagicMock()
    fake.get_account.return_value = {
        "equity": 100_000.0, "cash": 50_000.0, "buying_power": 100_000.0,
        "long_market_value": 50_000.0,
        "trading_blocked": False, "account_blocked": False,
    }
    fake.get_positions.return_value = {}
    fake.get_orders_for_date.return_value = []
    fake.get_order_by_id.side_effect = _by_id
    fake.next_session_date.return_value = date(2026, 4, 30)
    fake.tc = MagicMock()

    import sma.live.__main__ as _live_main
    monkeypatch.setattr(_live_main, "load_settings", lambda config_path: SimpleNamespace())
    monkeypatch.setattr(_live_main, "_build_alpaca", lambda settings: fake)

    config_path = tmp_path / "config.yaml"
    config_path.write_text("# stub\n")

    result = CliRunner().invoke(
        reconcile_cmd,
        ["--asof-date", asof.isoformat(), "--db", str(db_path), "--config", str(config_path)],
    )
    assert result.exit_code == 0, result.output

    # the GOOD fill was recorded…
    store = Store(path=str(db_path)).connect()
    n_fills = store.conn.execute(
        "SELECT COUNT(*) FROM paper_fills WHERE asof_date = ?", [asof]
    ).fetchone()[0]
    store.close()
    assert n_fills == 1
    # …but the batch must stay open for tomorrow's retry
    sentinel_files = list(isolated_sentinels.glob("com.sma.live.reconcile.daily-*.json"))
    assert sentinel_files == [], "fetch failure must defer the batch sentinel"


def test_resolve_reconcile_asof_rediscovers_batch_after_status_updated(
    isolated_sentinels, tmp_path,
):
    """Self-healing (Codex HIGH): if a prior reconcile updated statuses to
    terminal but crashed/deferred BEFORE writing the sentinel, the batch must
    still be re-discoverable. Selection keys off placed orders (alpaca_order_id),
    not status='submitted' — otherwise the batch is stranded (no sentinel + no
    'submitted' rows = invisible forever)."""
    from sma.live.__main__ import _resolve_reconcile_asof

    asof = date(2026, 5, 22)
    db_path = _db_with_pre_reconciled_batch(tmp_path, asof=asof)  # id set, no sentinel
    store = Store(path=str(db_path)).connect()
    try:
        # Post-reconcile-but-pre-sentinel state: status is terminal, no sentinel.
        store.conn.execute(
            "UPDATE intended_orders SET status='filled' WHERE asof_date=?", [asof]
        )
        assert _resolve_reconcile_asof(None, store) == asof
    finally:
        store.close()


# --- run-date LIVENESS sentinel (2026-06-09) ----------------------------------
# The batch sentinel above is keyed by the reconciled decide-date (yesterday),
# never today — correct for fill-attribution, but it made the watchdog re-kick
# reconcile every hour 19:00-22:00 every day ("no sentinel for today").
# reconcile now ALSO writes a liveness sentinel under its run date, which is
# what the watchdog checks (schedule.liveness_sentinel_label).


def test_liveness_sentinel_written_in_no_batches_path(
    isolated_sentinels, monkeypatch, tmp_path
):
    db_path = _empty_db(tmp_path)
    _stub_settings_and_alpaca(monkeypatch)
    config_path = tmp_path / "config.yaml"
    config_path.write_text("# stub\n")

    result = CliRunner().invoke(
        reconcile_cmd, ["--db", str(db_path), "--config", str(config_path)]
    )
    assert result.exit_code == 0, result.output

    from sma.live.__main__ import _today_et
    from sma.sentinels import read_sentinel

    live = read_sentinel(
        label="com.sma.live.reconcile.daily.ran", asof=_today_et()
    )
    assert live is not None, "watchdog needs a run-date liveness sentinel"
    assert live["reconciled_asof"] is None
    # the phantom-protection invariant still holds: no BATCH sentinel today
    assert (
        read_sentinel(label="com.sma.live.reconcile.daily", asof=_today_et()) is None
    )


def test_liveness_sentinel_records_reconciled_batch_asof(
    isolated_sentinels, monkeypatch, tmp_path
):
    yesterday = date(2026, 4, 30)
    db_path = _db_with_pre_reconciled_batch(tmp_path, asof=yesterday)
    _stub_settings_and_alpaca(monkeypatch)
    config_path = tmp_path / "config.yaml"
    config_path.write_text("# stub\n")

    result = CliRunner().invoke(
        reconcile_cmd, ["--db", str(db_path), "--config", str(config_path)]
    )
    assert result.exit_code == 0, result.output

    from sma.live.__main__ import _today_et
    from sma.sentinels import read_sentinel

    live = read_sentinel(
        label="com.sma.live.reconcile.daily.ran", asof=_today_et()
    )
    assert live is not None
    assert live["reconciled_asof"] == yesterday.isoformat()
