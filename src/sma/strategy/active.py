"""Agent-editable post-processor tilt function for xgb_top_k.

Phase 6 autoresearch: the LLM proposer rewrites the body of `tilt()` once
per loop iteration. The autoresearch loop runner enforces these constraints:

  - The signature of `tilt(...)` may NOT change across iterations.
  - All edits must be inside this file. Imports may be added.
  - `tilt(...)` must be deterministic given `(asof_date, decisions, ctx)`.
  - Output `target_weight` for each decision must remain in [0, max_position_pct].
  - Tickers may be REMOVED from the decision set; new tickers may NOT be
    introduced (the quant model produces the candidate universe).

The risk pipeline (sector cap, earnings blackout, drawdown, position cap)
runs AFTER this function; the cash floor enforces at order translation.
Both bound the agent's worst-case impact regardless of what tilt produces.

Default behavior: identity (no-op). The autoresearch agent rewrites the
body. See `specs/2026-04-28-phase-6-autoresearch.md` for the locked
design decisions.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date

from sma.backtest.strategies.base import StrategyDecision


@dataclass(frozen=True)
class TiltContext:
    """Read-only context the agent sees alongside quant decisions.

    Frozen so a misbehaving agent edit can't mutate it across iterations.
    """

    quant_scores: dict[str, float]
    """Raw model output: ticker → predicted forward return (untilted)."""

    theses: dict[str, dict] | None
    """Phase 4 thesis output: ticker → strategist payload, or None when
    `--use-theses=False`. Keys per thesis row: conviction, score, flags,
    action_hint, reasoning, catalyst_window."""

    portfolio_dollars: dict[str, float]
    """ticker → current position dollar value (NOT shares)."""

    sector_exposure: dict[str, float]
    """sector_name → fraction of equity in that sector. Sums to ≤ 1.0."""

    sector_for: Callable[[str], str]
    """ticker → GICS sector name. Same callable used by the risk pipeline."""

    account_equity: float
    """Total account value in dollars (long_market_value + cash)."""

    cash: float
    """Available cash in dollars."""


def tilt(
    *,
    asof_date: date,
    decisions: list[StrategyDecision],
    ctx: TiltContext,
) -> list[StrategyDecision]:
    """Post-process quant decisions before risk rails.

    Default body: identity (return decisions unchanged). The autoresearch
    agent rewrites this body, subject to the constraints listed in the
    module docstring.

    Returns:
        A list of `StrategyDecision` with possibly adjusted `target_weight`
        and/or a filtered subset of the input tickers. Cannot introduce
        new tickers.
    """
    return decisions
