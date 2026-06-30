"""Strategy Protocol and StrategyDecision dataclass.

A strategy is anything that takes (asof_date, market_data) and returns target
portfolio weights for the next trading day. The simulator applies these
weights at next-day open, enforces risk rails, and tracks P&L.

Sums of target_weight across all decisions for a given asof_date must be
<= 1.0. The simulator rejects decisions for tickers outside the universe.
"""

from dataclasses import dataclass
from datetime import date
from typing import Protocol

import pandas as pd


@dataclass(frozen=True)
class StrategyDecision:
    asof_date: date
    ticker: str
    target_weight: float       # 0.0 to 1.0; fraction of account in this position


class Strategy(Protocol):
    name: str

    def decide(
        self,
        asof_date: date,
        prices: pd.DataFrame,
        fundamentals: pd.DataFrame,
    ) -> list[StrategyDecision]:
        """Return target portfolio weights for the next trading day."""
        ...
