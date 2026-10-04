"""The sleeve contract: a Strategy proposes a TargetBook for its own capital.

A sleeve is one strategy with its own signal, trade session and capital
fraction (see ~/StockMarket/strategy-expansion.md, section 3). Weights in a
TargetBook are fractions of THE SLEEVE's capital, not of account equity; the
allocator (sma.strategies.allocator) scales them by the sleeve's
capital_fraction and nets every live sleeve into one aggregate book, which then
runs through the existing risk rails and order translation unchanged.

Kept deliberately plain: a dataclass, a context, an ABC with one method.
"""

from __future__ import annotations

import inspect
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date
from typing import Any, ClassVar

import pandas as pd

from sma.backtest.strategies.base import StrategyDecision

SESSIONS: tuple[str, ...] = ("open", "midday", "close")

# Float slack for "sum of weights <= limit" checks.
GROSS_EPS = 1e-9


@dataclass
class TargetBook:
    """What a sleeve wants to hold, as fractions of its own capital.

    `weights` keeps insertion order on purpose: the aggregate book is built in
    that order and the risk rails process decisions in order, so reordering
    would be a behaviour change. A ticker at weight 0.0 is meaningful: it says
    "the sleeve looked at this name and wants none of it".
    """

    weights: dict[str, float]
    # Optional per-ticker limit price hint for sessions that trade with limit
    # orders. None (or a missing key) means "no hint, execution decides".
    limit_hint: dict[str, float | None] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def gross(self) -> float:
        return float(sum(self.weights.values()))

    def validate(self, *, max_gross: float | None = 1.0) -> None:
        """Raise ValueError on a malformed book. Long-only: every weight must be
        finite and >= 0. `max_gross=None` skips the gross check (see
        Strategy.max_gross for the one sleeve that needs that)."""
        for ticker, w in self.weights.items():
            if not isinstance(ticker, str) or not ticker:
                raise ValueError(f"bad ticker in TargetBook: {ticker!r}")
            if not math.isfinite(w) or w < 0:
                raise ValueError(f"bad weight for {ticker}: {w!r} (must be finite and >= 0)")
        for ticker in self.limit_hint:
            if ticker not in self.weights:
                raise ValueError(f"limit_hint for {ticker} which has no weight")
        if max_gross is not None and self.gross > max_gross + GROSS_EPS:
            raise ValueError(f"TargetBook gross {self.gross:.6f} exceeds {max_gross}")


@dataclass(frozen=True)
class SleeveContext:
    """Read-only inputs every sleeve sees at propose time."""

    prices: pd.DataFrame
    current_holdings: frozenset[str] = frozenset()


class Strategy(ABC):
    """One sleeve. Subclasses set `name`, `session`, `horizon_days`."""

    name: ClassVar[str]
    session: ClassVar[str] = "open"
    horizon_days: ClassVar[int] = 1
    # Upper bound on TargetBook.gross, checked by the allocator. None means
    # unchecked: only the incumbent uses that, because its raw book grosses
    # ~1.24 by design and the rails (position cap, sector cap, cash floor)
    # clip it downstream. Normalizing it would change tonight's orders.
    max_gross: ClassVar[float | None] = 1.0

    @abstractmethod
    def propose(self, asof: date, ctx: SleeveContext) -> TargetBook:
        """Return the sleeve's target book for `asof`."""


def call_strategy_decide(
    strategy, *, asof: date, prices: pd.DataFrame, current_holdings: set[str],
) -> list[StrategyDecision]:
    """Call a legacy `.decide()` strategy, passing live holdings when it takes
    them (signature check mirrors the backtest simulator). Without holdings the
    bearish-thesis veto saw held=empty and force-sold merely-bearish HELD names
    in live trading (Codex post-audit review, 2026-06-09).

    This is THE one place that call happens: decide_once uses it for whatever
    strategy object it is handed, and the xgb_momentum sleeve uses it to run
    the incumbent, so the two paths cannot drift.
    """
    if "current_holdings" in inspect.signature(strategy.decide).parameters:
        return strategy.decide(
            asof_date=asof, prices=prices, current_holdings=current_holdings,
        )
    return strategy.decide(asof_date=asof, prices=prices)


def book_from_decisions(decisions: list[StrategyDecision]) -> TargetBook:
    """Lossless StrategyDecision list -> TargetBook (order preserved). A
    duplicate ticker would silently collapse, so it raises instead."""
    weights: dict[str, float] = {}
    for d in decisions:
        if d.ticker in weights:
            raise ValueError(f"duplicate ticker {d.ticker} in strategy decisions")
        weights[d.ticker] = d.target_weight
    return TargetBook(weights=weights)
