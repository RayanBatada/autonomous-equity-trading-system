"""Risk-rail pipeline. Composes per-rail checks in fixed order.

Both the simulator and live trading consume `apply(decisions, ctx)` so a
single source of truth governs sim-vs-paper parity. The simulator currently
calls `check_order` (back-compat shim below) inline within its daily loop;
the Phase 5 live module calls `apply` once per decide invocation.

Order of buy-side checks (fixed):
  1. drawdown — account-level; aborts ALL new buys when triggered
  2. earnings_blackout — per-ticker; rejects new buys within window
  3. sector_cap — per-decision; rejects when post-trade sector exposure > cap
  4. position_cap — per-decision; rejects when target_weight > max_position_pct

Stop-loss is NOT in this pipeline — it's a forced-sell trigger handled
separately in live.stop_loss and the simulator's open-of-day scan, not a
buy-side gate. Cash floor is checked at order-translation time (orders.py)
since it depends on the actual share count, not the target weight.
"""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date

from sma.backtest.strategies.base import StrategyDecision
from sma.risk.drawdown import check_drawdown
from sma.risk.earnings_blackout import in_earnings_blackout
from sma.risk.rails import RiskRails
from sma.risk.sector_cap import check_sector_cap


@dataclass
class RiskContext:
    rails: RiskRails
    account_value: float
    cash: float
    current_positions_dollars: dict[str, float]
    sector_exposure_pct: dict[str, float]   # sector → fraction of account_value
    sector_for: Callable[[str], str]         # ticker → sector
    current_drawdown: float                  # positive 0..1
    upcoming_earnings: dict[str, list[date]]
    asof_date: date
    # Held tickers whose FULL exit translate() will veto via min_hold_days.
    # apply() must NOT free their sector room when the model drops them —
    # the force-sell won't actually happen this cycle, so the room is fiction
    # (Codex module review 2026-06-11: cap could be exceeded for real).
    min_hold_protected: frozenset = frozenset()


def apply(decisions: list[StrategyDecision], ctx: RiskContext) -> list[StrategyDecision]:
    """Return only decisions that pass all rails.

    Rail semantics (2026-06-05 audit):
    - drawdown + earnings_blackout gate EXPOSURE-INCREASING decisions only (a new
      buy or an add). A reduction/derisk of a held name must always be allowed —
      you have to be able to sell when in drawdown or near earnings.
    - the position cap is enforced per TICKER: duplicate decisions for one ticker
      are coalesced (keep the first/highest-priority) so two sub-cap decisions
      can't bypass the cap.
    - the sector cap bounds the intended FINAL portfolio. It starts from current
      sector exposure but REMOVES held names that aren't in the model-approved
      target set, because translate() force-sells those (orders.py) — counting
      them would block same-sector buys against capital that's about to be freed.
      Reductions are applied to the running exposure BEFORE increases, so a
      same-sector trim frees cap room before buys in the same batch are evaluated.
    """
    av = ctx.account_value

    def current_weight(ticker: str) -> float:
        return ctx.current_positions_dollars.get(ticker, 0.0) / av if av > 0 else 0.0

    # Coalesce duplicate tickers (keep the first/highest-priority decision).
    unique: list[StrategyDecision] = []
    seen: set[str] = set()
    for d in decisions:
        if d.ticker not in seen:
            seen.add(d.ticker)
            unique.append(d)
    model_approved = seen  # == {d.ticker for d in unique}

    drawdown_triggered, _ = check_drawdown(
        rails=ctx.rails, current_drawdown=ctx.current_drawdown
    )

    # Start from current sector exposure, then free the room of holdings that
    # will be force-sold (held but not in the model-approved target set).
    running_sector_exposure: dict[str, float] = dict(ctx.sector_exposure_pct)
    for ticker, dollars in ctx.current_positions_dollars.items():
        if ticker in ctx.min_hold_protected:
            continue  # its exit will be min-hold-vetoed; room stays reserved
        if ticker not in model_approved:
            sector = ctx.sector_for(ticker)
            running_sector_exposure[sector] = (
                running_sector_exposure.get(sector, 0.0)
                - (dollars / av if av > 0 else 0.0)
            )

    def is_increasing(d: StrategyDecision) -> bool:
        return d.target_weight > current_weight(d.ticker)

    # Evaluate reductions (and holds) before increases so trims free cap room
    # first; stable sort preserves priority order within each group. Keep the
    # original index so the returned list stays in priority order.
    ordered = sorted(enumerate(unique), key=lambda iv: is_increasing(iv[1]))

    accepted_idx: list[tuple[int, StrategyDecision]] = []
    for idx, d in ordered:
        incr = is_increasing(d)
        if incr and drawdown_triggered:
            continue
        if incr and in_earnings_blackout(d.ticker, ctx.asof_date, ctx.upcoming_earnings):
            continue

        sector = ctx.sector_for(d.ticker)
        post_sector = (
            running_sector_exposure.get(sector, 0.0)
            - current_weight(d.ticker)
            + d.target_weight
        )
        # Caps gate EXPOSURE-INCREASING decisions only. A reduction/hold always
        # passes — you must be able to derisk an oversized position even if it's
        # still above a cap — and it still updates running exposure so the trim
        # frees room for later buys (Codex review, 2026-06-05).
        if incr:
            triggered, _ = check_sector_cap(
                rails=ctx.rails, sector=sector, sector_exposure_after=post_sector,
            )
            if triggered:
                continue
            if d.target_weight > ctx.rails.max_position_pct:
                continue

        accepted_idx.append((idx, d))
        running_sector_exposure[sector] = post_sector

    return [d for _, d in sorted(accepted_idx, key=lambda iv: iv[0])]


def check_order(
    *,
    rails: RiskRails,
    ticker: str,
    order_dollars: float,
    account_value: float,
    current_positions: dict[str, float],
    sector: str,
    sector_exposure: dict[str, float],
    current_drawdown: float,
) -> tuple[bool, str]:
    """Legacy single-order check kept for simulator back-compat.

    Returns (ok, reason) where ok=True means the order is allowed.
    Equivalent to running drawdown + position_cap + sector_cap on a single order.
    """
    if account_value <= 0:
        return False, "account_value <= 0"

    triggered, reason = check_drawdown(rails=rails, current_drawdown=current_drawdown)
    if triggered:
        return False, reason

    existing = current_positions.get(ticker, 0.0)
    new_position_value = existing + order_dollars
    if account_value > 0 and new_position_value / account_value > rails.max_position_pct:
        return False, (
            f"position {ticker} would be {new_position_value/account_value:.2%} "
            f"of account, exceeds cap {rails.max_position_pct:.2%}"
        )

    sector_after = sector_exposure.get(sector, 0.0) + order_dollars / account_value
    triggered, reason = check_sector_cap(
        rails=rails, sector=sector, sector_exposure_after=sector_after,
    )
    if triggered:
        return False, reason

    return True, ""
