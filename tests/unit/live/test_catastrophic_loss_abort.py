"""Regression: codex HIGH #4 (2026-05-14). Reconcile's
`_detect_catastrophic_loss` only ever EMITTED an alert; no code path actually
blocked the next decide. With this fix, decide_once raises
`CatastrophicLossAbort` when today's live equity has dropped >= the configured
abort threshold vs the prior account_snapshots row.

Tests cover:
- raises when drop exceeds threshold
- proceeds normally when drop is below threshold
- proceeds normally when there is no prior snapshot (cold start)
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from sma.ingest.store import Store
from sma.live.decide import CatastrophicLossAbort, decide_once
from sma.risk.rails import RiskRails


@pytest.fixture
def store(tmp_path: Path) -> Store:
    s = Store(path=tmp_path / "t.duckdb").connect()
    yield s
    s.close()


def _seed_snapshot(store: Store, asof: date, equity: float) -> None:
    store.conn.execute(
        "INSERT INTO account_snapshots "
        "(asof_date, equity, cash, buying_power, long_market_value, "
        " position_count, total_unrealized_pnl, run_id) "
        "VALUES (?, ?, 0, 0, ?, 0, 0, 1)",
        [asof, equity, equity],
    )


def _stub_alpaca(equity: float) -> MagicMock:
    """Minimal Alpaca client stub that returns the requested equity + no
    positions. We don't exercise the rest of decide; the abort fires
    before any pricing / strategy work."""
    client = MagicMock()
    client.get_account.return_value = {
        "equity": equity, "cash": equity, "buying_power": equity,
    }
    client.get_positions.return_value = {}
    return client


def test_catastrophic_loss_aborts_when_drop_exceeds_threshold(store):
    """Yesterday $100k, today $69k → 31% drop, threshold 30% → abort."""
    _seed_snapshot(store, date(2026, 5, 13), 100_000.0)
    alpaca = _stub_alpaca(equity=69_000.0)
    with pytest.raises(CatastrophicLossAbort, match="equity dropped"):
        decide_once(
            asof=date(2026, 5, 14),
            store=store,
            alpaca=alpaca,
            universe=[],
            strategy=MagicMock(),
            sector_for=lambda t: "Unknown",
            rails=RiskRails(),
            catastrophic_loss_abort_pct=0.30,
        )


def test_catastrophic_loss_does_not_abort_below_threshold(store):
    """Yesterday $100k, today $80k → 20% drop, threshold 30% → no abort
    (strategy still runs)."""
    _seed_snapshot(store, date(2026, 5, 13), 100_000.0)
    alpaca = _stub_alpaca(equity=80_000.0)
    # Provide a strategy that returns no decisions + no held positions, so
    # decide_once falls through without crashing on missing prices.
    strategy = MagicMock()
    strategy.decide.return_value = []
    # No raise: decide_once should proceed past the abort check.
    try:
        decide_once(
            asof=date(2026, 5, 14),
            store=store,
            alpaca=alpaca,
            universe=[],
            strategy=strategy,
            sector_for=lambda t: "Unknown",
            rails=RiskRails(),
            catastrophic_loss_abort_pct=0.30,
        )
    except CatastrophicLossAbort:
        pytest.fail("Should not abort at 20% drop with 30% threshold")
    except Exception:
        # Any *other* exception (e.g. downstream pipeline missing data) is
        # fine — we only care the catastrophic-loss path didn't fire.
        pass


def test_catastrophic_loss_no_prior_snapshot_passes(store):
    """First-day cold start: no prior snapshot → can't compute drop → proceed."""
    alpaca = _stub_alpaca(equity=10.0)  # arbitrary
    strategy = MagicMock()
    strategy.decide.return_value = []
    try:
        decide_once(
            asof=date(2026, 5, 14),
            store=store,
            alpaca=alpaca,
            universe=[],
            strategy=strategy,
            sector_for=lambda t: "Unknown",
            rails=RiskRails(),
            catastrophic_loss_abort_pct=0.30,
        )
    except CatastrophicLossAbort:
        pytest.fail("Should not abort when there is no prior snapshot")
    except Exception:
        pass


def test_catastrophic_loss_threshold_is_configurable(store):
    """The abort threshold comes from `catastrophic_loss_abort_pct` arg;
    a tighter threshold (10%) catches what 30% would miss."""
    _seed_snapshot(store, date(2026, 5, 13), 100_000.0)
    alpaca = _stub_alpaca(equity=88_000.0)  # 12% drop
    with pytest.raises(CatastrophicLossAbort):
        decide_once(
            asof=date(2026, 5, 14),
            store=store,
            alpaca=alpaca,
            universe=[],
            strategy=MagicMock(),
            sector_for=lambda t: "Unknown",
            rails=RiskRails(),
            catastrophic_loss_abort_pct=0.10,
        )
