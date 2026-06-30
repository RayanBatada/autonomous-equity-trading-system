"""Random long-only baseline strategy.

On each trading day, picks K distinct tickers from the universe (seeded RNG)
and assigns each a uniform-random weight in [0, max_weight_per_pick].

This is the noise-floor baseline. Any real strategy that doesn't beat this
on Sharpe is no better than random; ship it as the floor every other
strategy must clear. Seeded RNG so that the same seed produces byte-
identical decisions, which is required by the autoresearch determinism
guarantee in the Phase 1 spec.
"""

import random
from datetime import date

import pandas as pd

from sma.backtest.strategies.base import StrategyDecision


class RandomLongStrategy:
    name = "random_long"

    def __init__(
        self,
        universe: list[str],
        seed: int,
        num_picks: int = 5,
        max_weight_per_pick: float = 0.05,
    ) -> None:
        if not universe:
            raise ValueError("universe must be non-empty")
        if num_picks < 1:
            raise ValueError("num_picks must be >= 1")
        self.universe = list(universe)
        self.seed = seed
        self.num_picks = min(num_picks, len(universe))
        self.max_weight_per_pick = max_weight_per_pick
        self._rng = random.Random(seed)

    def decide(
        self,
        asof_date: date,
        prices: pd.DataFrame,
        fundamentals: pd.DataFrame | None = None,
    ) -> list[StrategyDecision]:
        picks = self._rng.sample(self.universe, self.num_picks)
        return [
            StrategyDecision(
                asof_date=asof_date,
                ticker=t,
                target_weight=self._rng.uniform(0.0, self.max_weight_per_pick),
            )
            for t in picks
        ]
