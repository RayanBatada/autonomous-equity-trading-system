"""Buy-and-hold SPY baseline strategy.

The harness honesty test. On day 1 of the window, target 5% of account into
SPY (matching the position cap from the risk rails). Then no further
decisions. The simulator handles the actual buy at day 2's open.

Note on the 5% target: a true buy-and-hold-100% would be rejected by the
position cap. This baseline measures "what if 5% of the book sits in SPY
and we hold for the whole window," which is a useful floor for any other
strategy to beat. Phase 3 may revisit whether benchmark holdings get a
position-cap exemption.
"""

from datetime import date

import pandas as pd

from sma.backtest.strategies.base import StrategyDecision


class BuyAndHoldSPYStrategy:
    name = "buy_and_hold_spy"

    def __init__(self, target_weight: float = 0.05) -> None:
        self._target_weight = target_weight
        self._bought = False

    def decide(
        self,
        asof_date: date,
        prices: pd.DataFrame,
        fundamentals: pd.DataFrame | None = None,
    ) -> list[StrategyDecision]:
        if self._bought:
            return []
        self._bought = True
        return [
            StrategyDecision(
                asof_date=asof_date,
                ticker="SPY",
                target_weight=self._target_weight,
            )
        ]
