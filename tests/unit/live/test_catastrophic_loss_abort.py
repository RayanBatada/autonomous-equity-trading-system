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


def test_catastrophic_loss_ignores_garbage_high_prior_snapshot(store):
    """2026-07-30: a spuriously HIGH prior-day snapshot (broker garbage, e.g.
    the 2026-07-07 Alpaca wipe class of incident) must not make today's real
    equity look like a catastrophic loss. Real history: $100k, $101k, then a
    garbage $500k row immediately before today. Today's real equity is $95k
    (a normal 5.9% dip off the real $101k) — without the fix, comparing
    against the $500k garbage row would show an 81% "drop" and wrongly
    abort."""
    _seed_snapshot(store, date(2026, 5, 11), 100_000.0)
    _seed_snapshot(store, date(2026, 5, 12), 101_000.0)
    _seed_snapshot(store, date(2026, 5, 13), 500_000.0)  # garbage
    alpaca = _stub_alpaca(equity=95_000.0)
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
        pytest.fail(
            "garbage-high prior snapshot must be filtered out, not treated "
            "as a real 81% loss"
        )
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


# ---------------------------------------------------------------------------
# Peak-relative catastrophic abort (2026-07-30 money-path review).
#
# The day-over-day check above only compares today vs YESTERDAY's snapshot,
# so it is unreachable by a slow bleed spread across many sub-threshold days
# (worst real single day so far: -5.1%) even though peak-to-trough drawdown
# has already hit -17.6% without tripping it. This second, independent check
# compares today's equity against the running (garbage-filtered) equity PEAK
# (the same cleaned peak the drawdown rail already computes) instead of
# yesterday's snapshot.
# ---------------------------------------------------------------------------


def test_catastrophic_peak_drawdown_aborts_when_drop_exceeds_threshold(store):
    """Peak $100k, today $74k -> 26% off peak clears the new 25% default
    threshold, even though the day-over-day drop (also 26%, same $100k prior
    snapshot) is under the separate 30% day-over-day threshold."""
    _seed_snapshot(store, date(2026, 5, 13), 100_000.0)
    alpaca = _stub_alpaca(equity=74_000.0)
    with pytest.raises(CatastrophicLossAbort, match="peak"):
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


def test_catastrophic_peak_drawdown_does_not_abort_below_threshold(store):
    """Peak $100k, today $76k -> 24% off peak, under the 25% default
    threshold -> no abort (strategy still runs)."""
    _seed_snapshot(store, date(2026, 5, 13), 100_000.0)
    alpaca = _stub_alpaca(equity=76_000.0)
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
        pytest.fail("Should not abort at 24% peak drawdown with 25% threshold")
    except Exception:
        # Any *other* exception (e.g. downstream pipeline missing data) is
        # fine -- we only care the peak-drawdown abort path didn't fire.
        pass


def test_catastrophic_peak_drawdown_threshold_is_configurable(store):
    """`catastrophic_peak_drawdown_abort_pct` is a distinct, configurable
    threshold from the day-over-day one. Day-over-day is set high (0.99) so
    only the peak check can plausibly fire; a tight peak threshold (5%)
    catches a 10% off-peak drop."""
    _seed_snapshot(store, date(2026, 5, 13), 100_000.0)
    alpaca = _stub_alpaca(equity=90_000.0)  # 10% off peak
    with pytest.raises(CatastrophicLossAbort, match="peak"):
        decide_once(
            asof=date(2026, 5, 14),
            store=store,
            alpaca=alpaca,
            universe=[],
            strategy=MagicMock(),
            sector_for=lambda t: "Unknown",
            rails=RiskRails(),
            catastrophic_loss_abort_pct=0.99,
            catastrophic_peak_drawdown_abort_pct=0.05,
        )


def test_catastrophic_peak_drawdown_ignores_garbage_high_peak(store):
    """A spuriously HIGH garbage snapshot (broker-wipe class of incident, same
    fixture shape as the day-over-day garbage test above) must not inflate
    the peak and false-trip the new check. Real history $100k/$101k, then a
    garbage $500k row; today's real equity $90k is a normal ~11% dip off the
    real (filtered) $101k peak, not an 82% "drop" off the unfiltered $500k
    one."""
    _seed_snapshot(store, date(2026, 5, 11), 100_000.0)
    _seed_snapshot(store, date(2026, 5, 12), 101_000.0)
    _seed_snapshot(store, date(2026, 5, 13), 500_000.0)  # garbage
    alpaca = _stub_alpaca(equity=90_000.0)
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
        pytest.fail(
            "garbage-high peak must be filtered (sanity guard) before "
            "computing peak drawdown, not treated as a real 82% loss"
        )
    except Exception:
        pass
