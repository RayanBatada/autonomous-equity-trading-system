"""Decision-to-order translation for live paper trading.

Math (per spec §5):
  target_shares = floor(target_weight * account_equity / last_price)
  delta = target_shares - current_held_shares
  delta > 0 → BUY (DAY market, queues to next session, fills near the open)
  delta < 0 → SELL (DAY, fills any time same day)
  delta == 0 → no order
Tickers in current_positions but NOT in decisions → full SELL.

Guards:
  - Empty decisions + held positions → raise (paranoia rail).
  - Stale price (>3 calendar days old) → skip ticker.
  - Missing price → skip ticker.
  - Zero/negative equity → return empty.
  - Cash floor (rails.cash_floor_pct > 0) → SELLs always allowed; BUYs are
    skipped if they would drop projected cash below floor.

Buys submit as DAY market orders (they queue to the next session and fill in
continuous trading near the open); sells submit as DAY too. OPG (open-auction-
only) was retired because it expired unfilled in Alpaca's paper engine — see
decide._submit_with_audit_trail.
"""

from dataclasses import dataclass
from datetime import date, timedelta
from math import floor

from loguru import logger

from sma.backtest.strategies.base import StrategyDecision
from sma.live.exceptions import EmptyDecisionsWithHeldPositionsError
from sma.risk.rails import RiskRails

STALE_PRICE_DAYS = 3   # weekend buffer + one halt day


def client_order_id(asof: date, ticker: str, side: str, *, source: str | None = None) -> str:
    """Deterministic Alpaca client_order_id for (asof, ticker, side). Submitting
    the same id again is idempotent (Alpaca rejects duplicates), and it lets
    reconcile recover an order that the broker accepted just before a crash
    recorded its broker id. Single source of truth for decide + reconcile."""
    if source is not None:
        return f"sma-{asof.isoformat()}-{ticker}-{source}-{side}"
    return f"sma-{asof.isoformat()}-{ticker}-{side}"


@dataclass
class Order:
    ticker: str
    side: str            # 'BUY' | 'SELL'
    shares: int
    type: str            # 'DAY' (buys + sells; OPG retired — expired in paper)
    last_price: float | None  # price used to compute target_shares (None if delisted)


def translate(
    *,
    decisions: list[StrategyDecision],
    current_positions: dict[str, int],
    account_equity: float,
    cash: float,
    last_prices: dict[str, tuple[float, date]],
    asof_date: date,
    rails: RiskRails | None = None,
    model_approved_tickers: set[str] | None = None,
    position_entry_dates: dict[str, date] | None = None,
    current_drawdown: float = 0.0,
) -> list[Order]:
    if not decisions and current_positions:
        # Paranoia rail. Suppress when min_hold is active AND every held
        # position is still inside the min_hold window — that's a valid
        # "hold everything one more day" outcome, not a programming bug.
        # Same logic decide_once uses; matched here so callers that go
        # straight to translate (e.g. integration tests) get the same.
        min_hold = rails.min_hold_days if rails is not None else 0
        entry_dates = position_entry_dates or {}
        if min_hold > 0:
            all_protected = True
            for ticker, shares in current_positions.items():
                if shares <= 0:
                    continue
                ed = entry_dates.get(ticker)
                if ed is None:
                    all_protected = False
                    break
                hd = (asof_date - ed).days
                if hd < 0 or hd >= min_hold:
                    all_protected = False
                    break
            if all_protected:
                return []  # legitimate "hold everything one more day"
        raise EmptyDecisionsWithHeldPositionsError(
            f"decide() returned 0 decisions but {len(current_positions)} positions held"
        )

    if account_equity <= 0:
        return []

    stale_threshold = asof_date - timedelta(days=STALE_PRICE_DAYS)
    decision_tickers = {d.ticker for d in decisions}

    sells: list[Order] = []
    buys: list[Order] = []

    # 2026-05-12: min_hold_days rail. The 5/5 paper run did 13 same-day
    # round-trips (BUY at OPEN + SELL same evening → -$197 slippage). When
    # the rail is set, skip SELLs of positions whose most-recent BUY was
    # within `min_hold_days` *calendar* days of asof_date. Reduces churn at
    # the cost of a 1-2 day delay on legitimate "sell signal" reactions.
    # Applied to BOTH the in-decisions full-liquidation branch (target=0 via
    # delta=-held_shares) AND the force-sell branch (held but not in
    # decisions), since both effect a full exit. Partial reductions still
    # fire — those are model-intended rebalances, not churn.
    min_hold = rails.min_hold_days if rails is not None else 0
    entry_dates = position_entry_dates or {}

    def _blocked_by_min_hold(ticker: str, sell_shares: int) -> bool:
        if min_hold <= 0:
            return False
        held = current_positions.get(ticker, 0)
        if held <= 0 or sell_shares < held:
            # Only protect FULL exits. Partial sells are model rebalances.
            return False
        entry_date = entry_dates.get(ticker)
        if entry_date is None:
            return False  # no recorded entry → can't protect
        held_days = (asof_date - entry_date).days
        # held_days < 0 indicates a backfill/clock-skew bug, not a fresh
        # position. Don't pin the position forever — treat as held long
        # enough so the operator can still exit.
        if held_days < 0:
            return False
        if held_days < min_hold:
            logger.info(
                "min_hold: blocked SELL of {} ({} shares, held {}d, "
                "min_hold={}d)", ticker, sell_shares, held_days, min_hold,
            )
            return True
        return False

    dead_zone = rails.rebalance_dead_zone_pct if rails is not None else 0.0

    def _in_dead_zone(delta: int, held: int) -> bool:
        """Skip partial rebalances smaller than `dead_zone * held`. Never
        applies to new entries (held==0) or full exits (delta==-held)."""
        if dead_zone <= 0:
            return False
        if held <= 0:
            return False  # new entry, not a rebalance
        if delta == -held:
            return False  # full exit, not a rebalance
        return abs(delta) < dead_zone * held

    for d in decisions:
        if d.ticker not in last_prices:
            continue
        price, price_date = last_prices[d.ticker]
        if price_date < stale_threshold:
            continue
        if price <= 0:
            continue
        target_value = d.target_weight * account_equity
        target_shares = floor(target_value / price)
        held = current_positions.get(d.ticker, 0)
        delta = target_shares - held
        if delta > 0:
            if _in_dead_zone(delta, held):
                logger.info(
                    "rebalance dead-zone: skipped BUY of {} ({} shares "
                    "vs held {}, dead_zone={:.0%})",
                    d.ticker, delta, held, dead_zone,
                )
                continue
            buys.append(Order(
                ticker=d.ticker, side="BUY", shares=delta,
                type="DAY", last_price=price,
            ))
        elif delta < 0:
            if _blocked_by_min_hold(d.ticker, -delta):
                # Fresh full-liquidation via target_weight=0 — block.
                continue
            if _in_dead_zone(delta, held):
                logger.info(
                    "rebalance dead-zone: skipped SELL of {} ({} shares "
                    "vs held {}, dead_zone={:.0%})",
                    d.ticker, -delta, held, dead_zone,
                )
                continue
            sells.append(Order(
                ticker=d.ticker, side="SELL", shares=-delta,
                type="DAY", last_price=price,
            ))

    # Force-sell held names that the MODEL itself dropped — not names that
    # rails (sector_cap, earnings_blackout, etc.) merely blocked from re-buying.
    # `model_approved_tickers` is the raw pre-rails top-K from the strategy.
    # When None, we fall back to the post-rails decision set (legacy behavior;
    # backward-compatible for callers/tests that don't pass the new arg).
    keep_set = decision_tickers | (model_approved_tickers or set())
    for ticker, held_shares in current_positions.items():
        if ticker not in keep_set and held_shares > 0:
            if _blocked_by_min_hold(ticker, held_shares):
                continue
            price = last_prices.get(ticker, (None, None))[0]
            sells.append(Order(
                ticker=ticker, side="SELL", shares=held_shares,
                type="DAY", last_price=price,
            ))

    # Cash floor, drawdown-scaled (2026-06-16): in a drawdown the effective
    # floor rises, cutting buys as a persistent momentum crash develops
    # (validated: slope 1.5 improved val sharpe/return/maxDD). slope=0 (the
    # default) leaves the floor static.
    if rails is not None and rails.cash_floor_pct > 0:
        from sma.risk.derisk import derisk_cash_floor
        _eff_floor = derisk_cash_floor(
            rails.cash_floor_pct, current_drawdown,
            start=rails.drawdown_derisk_start,
            slope=rails.drawdown_derisk_slope,
            cap=rails.drawdown_derisk_cap,
        )
        floor_dollars = _eff_floor * account_equity
    else:
        floor_dollars = 0.0

    sell_proceeds = sum(
        (s.last_price or 0.0) * s.shares for s in sells
    )
    running_cash = cash + sell_proceeds

    accepted_buys: list[Order] = []
    for buy in buys:
        cost = (buy.last_price or 0.0) * buy.shares
        if running_cash - cost < floor_dollars:
            continue
        accepted_buys.append(buy)
        running_cash -= cost

    return sells + accepted_buys
