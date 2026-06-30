"""BacktestResult: the output of evaluate_strategy().

Single source of truth for what a backtest run reports. The auto-research
agent in Phase 6 will optimize on `sharpe`; everything else is for human
diagnostics and the overfit detector.
"""

from dataclasses import dataclass, field
from datetime import date
from typing import Literal


@dataclass(frozen=True)
class BacktestResult:
    strategy_name: str
    window: Literal["train", "val", "test"]
    start_date: date
    end_date: date

    # Primary metric. The number the agent optimizes.
    sharpe: float

    # Supporting metrics. Reported but not the optimization target.
    sortino: float
    calmar: float
    max_drawdown: float        # negative number, e.g. -0.18
    total_return: float
    annualized_return: float
    hit_rate: float            # fraction of profitable round-trips
    num_trades: int
    avg_holding_days: float

    # Diagnostics for the overfit detector.
    daily_returns: list[float]
    monthly_returns: list[float]

    # Provenance.
    code_commit: str
    data_run_id: int
    seed: int | None

    # Optional human-diagnostic detail. Defaulted so existing test fixtures
    # constructing BacktestResult by-field don't have to provide them.
    # monthly_periods is parallel to monthly_returns (e.g. "2025-07").
    # trades is one entry per buy/sell fill.
    monthly_periods: list[str] = field(default_factory=list)
    trades: list[dict] = field(default_factory=list)
