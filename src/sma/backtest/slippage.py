"""Slippage model for backtest fills.

Slippage is the difference between the price observed at decision time and
the price actually paid (buy) or received (sell). For liquid US large-caps,
5 bps (0.05%) is a reasonable default. Below a configurable liquidity floor
the model raises rather than just penalizing, since the universe selection
in Phase 0 already filters out illiquid tickers.

Sign convention: buys pay more than intended_price, sells receive less.
This is the conservative (worst-case) direction.
"""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class SlippageModel:
    base_bps: float = 5.0
    illiquidity_penalty_bps: float = 5.0
    illiquidity_threshold_dollars: float = 10_000_000.0
    min_adv_dollars: float = 5_000_000.0


def apply_slippage(
    intended_price: float,
    side: Literal["buy", "sell"],
    adv_dollars: float,
    model: SlippageModel,
) -> float:
    """Return the actual fill price after slippage.

    Args:
        intended_price: the price observed at decision time (e.g. next-day open).
        side: "buy" or "sell".
        adv_dollars: 20-day average daily dollar volume for the ticker.
        model: the SlippageModel parameters.
    """
    if adv_dollars < model.min_adv_dollars:
        raise ValueError(
            f"adv_dollars={adv_dollars} below minimum liquidity "
            f"{model.min_adv_dollars}; ticker should not be in universe"
        )

    bps = model.base_bps
    if adv_dollars < model.illiquidity_threshold_dollars:
        bps += model.illiquidity_penalty_bps

    multiplier = 1.0 + (bps / 10_000.0) * (1 if side == "buy" else -1)
    return intended_price * multiplier
