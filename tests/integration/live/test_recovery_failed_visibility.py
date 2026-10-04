"""Codex post-audit review (2026-06-09): a crash-after-accept row whose
client_order_id recovery hit a TRANSIENT error is set status='recovery_failed'
(decide, fail-closed — correct). But reconcile's asof discovery and
_backfill_null_order_ids both filtered to status='submitted', so the batch
became INVISIBLE: a broker-accepted (and possibly filled) order would never
reach paper_fills without manual surgery. recovery_failed must stay
discoverable and retryable."""

import uuid
from datetime import date
from unittest.mock import MagicMock

from sma.ingest.store import Store
from sma.live.__main__ import _resolve_reconcile_asofs
from sma.live.reconcile import _backfill_null_order_ids


def _db_with_recovery_failed_row(tmp_path, *, asof: date):
    store = Store(path=tmp_path / "t.duckdb").connect()
    rid = store.allocate_run_id()
    store.conn.execute(
        """
        INSERT INTO intended_orders
        (intended_order_id, asof_date, ticker, side, target_shares,
         target_weight, last_price, source, alpaca_order_id, status, run_id)
        VALUES (?, ?, 'AAPL', 'BUY', 10, NULL, 200.0, 'decide',
                NULL, 'recovery_failed', ?)
        """,
        [str(uuid.uuid4()), asof, rid],
    )
    return store


def test_resolve_reconcile_asof_discovers_recovery_failed_batch(tmp_path, monkeypatch):
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path / "s"))
    asof = date(2026, 4, 30)
    store = _db_with_recovery_failed_row(tmp_path, asof=asof)
    try:
        assert _resolve_reconcile_asofs(None, store) == [asof], (
            "a NULL-id recovery_failed batch must stay discoverable — its "
            "order may be live (accepted) at the broker"
        )
    finally:
        store.conn.close()


def test_backfill_recovers_recovery_failed_rows(tmp_path):
    asof = date(2026, 4, 30)
    store = _db_with_recovery_failed_row(tmp_path, asof=asof)
    alpaca = MagicMock()
    alpaca.get_order_by_client_order_id.return_value = ("broker-id-123", "accepted")
    try:
        recovered, lookup_failures = _backfill_null_order_ids(
            asof=asof, store=store, alpaca=alpaca
        )
        assert recovered == 1
        assert lookup_failures == 0
        row = store.conn.execute(
            "SELECT alpaca_order_id FROM intended_orders WHERE asof_date = ?",
            [asof],
        ).fetchone()
        assert row[0] == "broker-id-123"
    finally:
        store.conn.close()
