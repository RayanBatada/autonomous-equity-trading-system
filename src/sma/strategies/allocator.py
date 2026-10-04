"""Capital fractions per sleeve, validation, and the combiner.

Config shape (config.yaml `strategies.sleeves`, model in sma.config):

    - {name: xgb_momentum, capital_fraction: 1.0, mode: live, enabled: true}

Rules:
  * mode is "live" (its book trades) or "shadow" (its book is proposed and
    persisted for scoring, and NEVER reaches the order path).
  * live, enabled capital fractions sum to <= 1.0; the rest is cash.
  * at least one live, enabled sleeve (zero would hand decide an empty book,
    which force-sells everything held and trips the paranoia rail).

Combining: aggregate weight of a ticker = sum over live sleeves of
capital_fraction * sleeve weight, as a fraction of account equity. The
aggregate is handed to the existing risk rails and order translation as
ordinary StrategyDecisions, so the rails, stop-loss and reconcile all run on
the aggregate book exactly as before. With the default single sleeve at 1.0,
`w * 1.0 == w` exactly in IEEE floats, so the aggregate IS the incumbent's
decision list, same tickers, same order, same weights.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd

from sma.backtest.strategies.base import StrategyDecision
from sma.strategies.base import GROSS_EPS, SleeveContext, Strategy, TargetBook

logger = logging.getLogger(__name__)

MODES: tuple[str, ...] = ("live", "shadow")


def validate_sleeves(sleeves) -> None:
    """Raise ValueError if the sleeve list is not a valid allocation. Accepts
    anything with name / capital_fraction / mode / enabled attributes."""
    seen: set[str] = set()
    live_total = 0.0
    live_count = 0
    for s in sleeves:
        if s.name in seen:
            raise ValueError(f"duplicate sleeve name {s.name!r}")
        seen.add(s.name)
        if s.mode not in MODES:
            raise ValueError(f"sleeve {s.name}: mode {s.mode!r} not in {MODES}")
        if not (0.0 <= s.capital_fraction <= 1.0):
            raise ValueError(
                f"sleeve {s.name}: capital_fraction {s.capital_fraction} not in [0, 1]"
            )
        if s.enabled and s.mode == "live":
            live_total += s.capital_fraction
            live_count += 1
    if live_total > 1.0 + GROSS_EPS:
        raise ValueError(f"live sleeve capital fractions sum to {live_total:.4f} > 1.0")
    if live_count == 0:
        raise ValueError("no enabled live sleeve; decide needs at least one")


def combine_weights(live_books: list[tuple[float, TargetBook]]) -> dict[str, float]:
    """Net live sleeve books into one {ticker: weight of account equity}.

    Insertion order: first sleeve's tickers in its order, then any new
    tickers from later sleeves in theirs.
    """
    out: dict[str, float] = {}
    for fraction, book in live_books:
        for ticker, w in book.weights.items():
            contrib = fraction * w
            out[ticker] = out[ticker] + contrib if ticker in out else contrib
    return out


def combine_notional(
    live_books: list[tuple[float, TargetBook]], *, equity: float,
) -> dict[str, float]:
    """The aggregate book as {ticker: target dollars}. Decide hands the rails
    WEIGHTS (combine_weights), not these, because translate sizes from
    weights and a weight -> dollars -> weight round trip is not exact."""
    return {t: w * equity for t, w in combine_weights(live_books).items()}


@dataclass
class SleeveProposal:
    name: str
    mode: str
    capital_fraction: float
    session: str
    book: TargetBook | None
    error: str | None = None


class SleeveBook:
    """Duck-types a legacy strategy (`.decide(asof_date, prices,
    current_holdings)`) so decide_once runs every sleeve without knowing
    sleeves exist. Proposals from the last call are kept on
    `self.proposals` for attribution (sleeve_targets)."""

    name = "sleeves"

    def __init__(self, sleeves: list[tuple[Any, Strategy]], *, session: str = "open"):
        validate_sleeves([spec for spec, _ in sleeves])
        self.session = session
        # Only this session's enabled sleeves run here.
        self.sleeves = [
            (spec, strat) for spec, strat in sleeves
            if spec.enabled and strat.session == session
        ]
        self.proposals: list[SleeveProposal] = []

    def decide(
        self, *, asof_date: date, prices: pd.DataFrame,
        current_holdings: set[str] | None = None,
    ) -> list[StrategyDecision]:
        ctx = SleeveContext(prices=prices, current_holdings=frozenset(current_holdings or ()))
        self.proposals = []
        live_books: list[tuple[float, TargetBook]] = []
        for spec, strat in self.sleeves:
            if spec.mode == "shadow":
                # A shadow sleeve must never be able to break a live trade
                # night: any failure is logged and recorded, nothing raised.
                try:
                    book = strat.propose(asof_date, ctx)
                    book.validate(max_gross=strat.max_gross)
                except Exception as e:  # noqa: BLE001
                    logger.warning("shadow sleeve %s failed: %s", spec.name, e)
                    self.proposals.append(SleeveProposal(
                        spec.name, spec.mode, spec.capital_fraction, strat.session, None,
                        error=f"{type(e).__name__}: {e}",
                    ))
                    continue
            else:
                # Live sleeve failures propagate, same as the incumbent always has.
                book = strat.propose(asof_date, ctx)
                book.validate(max_gross=strat.max_gross)
                live_books.append((spec.capital_fraction, book))
            self.proposals.append(SleeveProposal(
                spec.name, spec.mode, spec.capital_fraction, strat.session, book,
            ))
        weights = combine_weights(live_books)
        return [
            StrategyDecision(asof_date=asof_date, ticker=t, target_weight=w)
            for t, w in weights.items()
        ]


def build_sleeve_book(strategies_settings, build_ctx, *, session: str = "open") -> SleeveBook:
    """Config -> SleeveBook. Disabled sleeves are not even constructed."""
    from sma.strategies import registry

    validate_sleeves(strategies_settings.sleeves)
    pairs = []
    for spec in strategies_settings.sleeves:
        if not spec.enabled:
            continue
        strat = registry.build(spec.name, build_ctx)
        # Guard until cross-session netting exists: the open decide nets the
        # whole broker book toward ITS aggregate, so it would sell anything a
        # live midday/close sleeve holds. Such sleeves may run in shadow only.
        if spec.mode == "live" and strat.session != "open":
            raise ValueError(
                f"sleeve {spec.name}: live mode is only supported in the open session "
                f"(got {strat.session!r}); run it in shadow until cross-session netting exists"
            )
        pairs.append((spec, strat))
    return SleeveBook(pairs, session=session)
