"""Per-strategy risk-rail configuration. Shared by simulator and live."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RiskRails:
    max_position_pct: float = 0.05      # 5% per ticker
    max_sector_pct: float = 0.25        # 25% per GICS sector
    max_drawdown_pct: float = 0.15      # 15% from rolling peak
    stop_loss_pct: float = 0.08         # 8% below entry (Phase 5 ships with 0.0 to disable)
    cash_floor_pct: float = 0.05        # always keep 5% in cash
    min_hold_days: int = 0              # block force-sells of positions held < N days (0 disables)
    # Skip partial rebalances smaller than N% of current position size.
    # 0 disables. See orders.translate() for the carve-outs.
    rebalance_dead_zone_pct: float = 0.10
    # Drawdown-scaled de-risking (2026-06-15): raise the effective cash floor
    # as drawdown-from-peak deepens, cutting exposure in persistent momentum
    # crashes (see sma.risk.derisk). slope=0 disables (default). When enabled,
    # past drawdown_derisk_start the floor rises by drawdown_derisk_slope per
    # unit of excess drawdown, clamped to drawdown_derisk_cap.
    drawdown_derisk_start: float = 0.05
    drawdown_derisk_slope: float = 0.0
    drawdown_derisk_cap: float = 0.60
