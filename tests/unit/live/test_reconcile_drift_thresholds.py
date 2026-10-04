"""review 2026-07-04 finding #4: the live.drift.* config thresholds were dead —
reconcile's detectors read hardcoded module constants, so an operator who
tightened a threshold in config.yaml got no change. These tests pin that the
detectors (and reconcile) now honor a caller-supplied threshold."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from sma.ingest.store import Store
from sma.live.reconcile import DriftThresholds, _detect_catastrophic_loss


def _seed_snapshot(store: Store, asof: date, equity: float) -> None:
    store.conn.execute(
        "INSERT INTO account_snapshots "
        "(asof_date, equity, cash, buying_power, long_market_value, "
        " position_count, total_unrealized_pnl, run_id) "
        "VALUES (?, ?, 0, 0, ?, 0, 0, 1)",
        [asof, equity, equity],
    )


def test_detect_catastrophic_loss_honors_threshold(tmp_path: Path) -> None:
    store = Store(path=tmp_path / "t.duckdb").connect()
    try:
        _seed_snapshot(store, date(2026, 5, 13), 100_000.0)
        _seed_snapshot(store, date(2026, 5, 14), 85_000.0)  # 15% drop
        # Default (0.10): a 15% drop alerts.
        assert _detect_catastrophic_loss(snapshot_date=date(2026, 5, 14), store=store)
        # Loosened to 0.20: the same 15% drop is now below threshold → no alert.
        assert not _detect_catastrophic_loss(
            snapshot_date=date(2026, 5, 14), store=store, threshold=0.20
        )
    finally:
        store.close()


def test_drift_thresholds_defaults_match_the_historical_constants() -> None:
    t = DriftThresholds()
    assert t.buy_miss == 0.20
    assert t.partial_fill == 0.90
    assert t.catastrophic_loss == 0.10


def test_drift_thresholds_from_config_maps_each_key() -> None:
    """config.yaml's live.drift.* keys map to the right threshold (a mis-map would
    silently swap safety rails). Uses the real LiveDrift model as the source."""
    from sma.config import LiveDrift

    cfg = LiveDrift(
        buy_miss_alert_threshold_pct=0.11,
        partial_fill_alert_threshold_pct=0.22,
        catastrophic_loss_alert_pct=0.33,
        catastrophic_loss_abort_pct=0.44,
    )
    t = DriftThresholds.from_config(cfg)
    assert t.buy_miss == 0.11
    assert t.partial_fill == 0.22
    assert t.catastrophic_loss == 0.33
