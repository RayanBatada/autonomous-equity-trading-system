"""Equal-weight buy-and-hold baseline.

Allocates 1/N of the account to each of N tickers in the universe on day 1,
then holds. The simulator's risk rails (5% per-ticker cap) will reject any
target weight above 5%, so this strategy is most useful when the universe
has at least 20 tickers (1/20 = 0.05 = right at the cap).

For our 80-ticker universe, 1/80 = 0.0125 = 1.25% per name, well below the
cap. This produces a well-diversified buy-and-hold baseline that any active
strategy must beat (or justify the increased concentration).
"""

from datetime import date

import pandas as pd

from sma.backtest.strategies.base import StrategyDecision


class EqualWeightStrategy:
    name = "equal_weight"

    def __init__(self, universe: list[str]) -> None:
        if not universe:
            raise ValueError("universe must be non-empty")
        self.universe = list(universe)
        self._allocated = False

    def decide(
        self,
        asof_date: date,
        prices: pd.DataFrame,
        fundamentals: pd.DataFrame | None = None,
    ) -> list[StrategyDecision]:
        if self._allocated:
            return []
        self._allocated = True
        weight = 1.0 / len(self.universe)
        return [
            StrategyDecision(asof_date=asof_date, ticker=t, target_weight=weight)
            for t in self.universe
        ]
