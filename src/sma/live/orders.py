"""Decision-to-order translation for live paper trading.

Math (per spec §5):
  target_shares = floor(target_weight * account_equity / last_price)
  (fractional mode truncates to sizing.fractional_precision instead of flooring)
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

from loguru import logger

from sma.backtest.strategies.base import StrategyDecision
from sma.live.exceptions import EmptyDecisionsWithHeldPositionsError
from sma.live.quantity import QTY_EPS, fmt_qty, is_zero, qty_eq
from sma.live.sizing import SizingPolicy
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
    shares: float        # int in whole-share mode, fractional when enabled
    type: str            # 'DAY' (buys + sells; OPG retired — expired in paper)
    last_price: float | None  # price used to compute target_shares (None if delisted)
    # Set when the ADV participation rail trimmed this order. The untraded
    # remainder is NOT queued anywhere: the next rebalance recomputes the same
    # delta from the (now larger) held position, so a big book walks into a
    # name over several sessions instead of printing it into one open auction.
    capped_by: str | None = None
    # A full liquidation of the held position. Exempt from the min-notional
    # floor — a dust position must always remain sellable.
    full_exit: bool = False

    @property
    def notional(self) -> float:
        return (self.last_price or 0.0) * self.shares


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
    sizing: SizingPolicy | None = None,
    adv_dollars: dict[str, float] | None = None,
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

    # Default policy == today's behaviour: whole shares, no floors, no cap.
    sizing = sizing or SizingPolicy()
    adv_dollars = adv_dollars or {}

    stale_threshold = asof_date - timedelta(days=STALE_PRICE_DAYS)
    decision_tickers = {d.ticker for d in decisions}

    # Fund the highest-conviction names first when cash is tight. The strategy
    # already emits decisions in descending target_weight (weight_tilt). Sort by
    # -target_weight ONLY: Python's sort is stable, so TIED weights keep the
    # strategy's emission order (which IS the conviction order). A secondary
    # ticker key would break ties alphabetically and could fund a lower-
    # conviction name over a higher one when weights tie (Codex HIGH, 2026-07-01).
    decisions = sorted(decisions, key=lambda d: -d.target_weight)

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

    def _blocked_by_min_hold(ticker: str, sell_shares: float) -> bool:
        if min_hold <= 0:
            return False
        held = current_positions.get(ticker, 0)
        # `sell_shares < held` with a QTY_EPS slack: under fractional sizing a
        # full exit is computed as `held - target` and lands a float-ulp under
        # `held`, which bare `<` would misread as a partial (model rebalance)
        # and wave straight past the churn rail.
        if held <= 0 or sell_shares < held - QTY_EPS:
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
                "min_hold={}d)", ticker, fmt_qty(sell_shares), held_days, min_hold,
            )
            return True
        return False

    dead_zone = rails.rebalance_dead_zone_pct if rails is not None else 0.0

    def _in_dead_zone(delta: float, held: float) -> bool:
        """Skip partial rebalances at-or-smaller than `dead_zone * held`. Never
        applies to new entries (held==0) or full exits (delta==-held).

        `<=` not `<` (2026-07-30): with held=10 and the default 10% dead zone,
        a 1-share delta is EXACTLY at the threshold (1 == 0.10*10) and the old
        `<` comparison let it through — CAT did four 1-share round-trips
        7/23-7/28 paying spread each time on exactly this boundary case.

        FRACTIONAL SEMANTICS (2026-08-27). The dead zone is defined on SHARES,
        and that is also the answer for notional: both sides of
        `|delta| <= dead_zone * held` are multiplied by the SAME last_price to
        become notional, so `|delta*p| <= dead_zone*held*p` is the identical
        test. Shares and percent-of-held-notional are the same rail here; no
        choice has to be made, and none of the calibration above changes. What
        DOES change is that `delta` and `held` are floats, so the full-exit
        carve-out needs a tolerance (see qty_eq) or a 1e-16 residue turns a
        liquidation into a "partial" the dead zone then swallows, leaving the
        position permanently un-exitable."""
        if dead_zone <= 0:
            return False
        if held <= 0:
            return False  # new entry, not a rebalance
        if qty_eq(delta, -held):
            return False  # full exit, not a rebalance
        return abs(delta) <= dead_zone * held

    for d in decisions:
        entry = last_prices.get(d.ticker)
        price, price_date = entry if entry else (None, None)
        fresh = entry is not None and price > 0 and price_date >= stale_threshold
        held = current_positions.get(d.ticker, 0)
        if fresh:
            target_shares = sizing.target_qty(d.target_weight, account_equity, price)
            delta = target_shares - held
        else:
            # The missing/stale-price veto applies to exposure INCREASES only.
            # Swallowing a held name's reduction/exit here breached the sector
            # cap for real: pipeline.apply frees the reduction's sector room
            # BEFORE approving same-sector buys, so the buys go out while the
            # sell silently vanishes — oversized position, no alert (review
            # 2026-07-01, MEDIUM; same rails/translate divergence class as the
            # min_hold_protected fix). Full exits need no price (the force-sell
            # branch below already submits with last_price=None); partial
            # reductions size on the last KNOWN price, stale beats nothing.
            if held <= 0:
                continue
            if d.target_weight <= 0:
                target_shares = 0
            elif price and price > 0:
                target_shares = sizing.target_qty(d.target_weight, account_equity, price)
            else:
                continue  # held, target>0, no price at all: cannot size a partial
            delta = target_shares - held
            if delta >= -QTY_EPS:
                continue  # stale veto stands for anything not reducing
        if delta > QTY_EPS:
            if _in_dead_zone(delta, held):
                logger.info(
                    "rebalance dead-zone: skipped BUY of {} ({} shares "
                    "vs held {}, dead_zone={:.0%})",
                    d.ticker, fmt_qty(delta), fmt_qty(held), dead_zone,
                )
                continue
            buys.append(Order(
                ticker=d.ticker, side="BUY", shares=delta,
                type="DAY", last_price=price,
            ))
        elif delta < -QTY_EPS:
            if _blocked_by_min_hold(d.ticker, -delta):
                # Fresh full-liquidation via target_weight=0 — block.
                continue
            if _in_dead_zone(delta, held):
                logger.info(
                    "rebalance dead-zone: skipped SELL of {} ({} shares "
                    "vs held {}, dead_zone={:.0%})",
                    d.ticker, fmt_qty(-delta), fmt_qty(held), dead_zone,
                )
                continue
            sells.append(Order(
                ticker=d.ticker, side="SELL", shares=-delta,
                type="DAY", last_price=price,
                full_exit=qty_eq(delta, -held),
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
                full_exit=True,
            ))

    # ---- Liquidity + order-size rails (2026-08-27 money-path scale audit).
    # Both run AFTER sizing and BEFORE the cash floor, in that order: the ADV
    # cap can shrink an order below the min-notional floor, and a floor applied
    # first would then let a sub-minimum order through. Both are OFF by default
    # and are no-ops on today's config (SizingPolicy() has cap=0, floor=0).
    sells = _apply_participation_cap(sells, sizing=sizing, adv_dollars=adv_dollars)
    buys = _apply_participation_cap(buys, sizing=sizing, adv_dollars=adv_dollars)
    sells = _apply_min_notional(sells, sizing=sizing)
    buys = _apply_min_notional(buys, sizing=sizing)

    # Cash floor, drawdown-scaled (2026-06-16): in a drawdown the effective
    # floor rises, cutting buys as a persistent momentum crash develops.
    # The rail exists but is OFF by default (slope=0, static floor): the
    # 2026-07-30 multi-window validation rejected enabling it — 0/4 reversal
    # windows showed a shallower maxDD at any tested slope (1.5/2.0/5.0), see
    # ~/.sma-pit/derisk-validation-2026-07-30/FINAL_RESULTS.txt.
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

    # Estimated sell proceeds fund same-day rotation buys, but they are priced
    # at the prior close and actually fill near the (usually lower) open. Apply
    # a haircut so we do not over-commit cash we may not realize. 1.0 = legacy.
    haircut = rails.sell_proceeds_haircut if rails is not None else 1.0
    sell_proceeds = haircut * sum((s.last_price or 0.0) * s.shares for s in sells)
    running_cash = cash + sell_proceeds

    accepted_buys: list[Order] = []
    for buy in buys:
        cost = (buy.last_price or 0.0) * buy.shares
        if running_cash - cost < floor_dollars:
            continue
        accepted_buys.append(buy)
        running_cash -= cost

    return sells + accepted_buys


def _apply_participation_cap(
    orders: list[Order],
    *,
    sizing: SizingPolicy,
    adv_dollars: dict[str, float],
) -> list[Order]:
    """Trim any order that would take more than `max_participation_of_adv` of
    the name's 20-day average daily DOLLAR volume.

    The remainder is deliberately not carried in any queue. Tomorrow's decide
    re-derives `target - held` from the larger held position and emits the next
    slice, so the book walks into (or out of) a name over as many sessions as
    the cap requires — with a fresh price and a fresh signal each time, which is
    strictly better than honouring a stale plan.

    Fails OPEN on an unknown ADV: a missing price history must not silently
    stop trading a name. The log line names it so the gap is visible.
    """
    if sizing.max_participation_of_adv <= 0:
        return orders
    out: list[Order] = []
    for o in orders:
        price = o.last_price
        if not price or price <= 0:
            out.append(o)
            continue
        adv = adv_dollars.get(o.ticker, 0.0)
        if adv <= 0:
            logger.warning(
                "participation cap: no ADV for {}; order passes uncapped "
                "({} shares, ${:,.0f})", o.ticker, fmt_qty(o.shares), o.notional,
            )
            out.append(o)
            continue
        cap_qty = sizing.participation_cap_qty(adv_dollars=adv, price=price)
        if cap_qty is None or o.shares <= cap_qty:
            out.append(o)
            continue
        if is_zero(cap_qty) or cap_qty <= 0:
            logger.warning(
                "participation cap: {} {} DROPPED — {:.1%} of ADV ${:,.0f} is "
                "less than one tradeable unit at ${:.2f}",
                o.side, o.ticker, sizing.max_participation_of_adv, adv, price,
            )
            continue
        logger.warning(
            "participation cap: {} {} trimmed {} → {} shares "
            "(${:,.0f} → ${:,.0f} = {:.2%} of 20d ADV ${:,.0f}); remainder "
            "carries to the next rebalance",
            o.side, o.ticker, fmt_qty(o.shares), fmt_qty(cap_qty),
            o.notional, cap_qty * price, sizing.max_participation_of_adv, adv,
        )
        out.append(Order(
            ticker=o.ticker, side=o.side, shares=cap_qty, type=o.type,
            last_price=o.last_price, capped_by="adv_participation",
            # A trimmed exit is no longer a full exit: it must stay eligible for
            # the min-notional floor's exemption ONLY while it really liquidates.
            full_exit=False,
        ))
    return out


def _apply_min_notional(
    orders: list[Order], *, sizing: SizingPolicy,
) -> list[Order]:
    """Drop orders below `min_order_notional` dollars.

    Alpaca rejects a fractional order under $1, so at a small account this is
    the difference between "k-1 names traded and one broker rejection every
    night" and a clean book. FULL EXITS are never dropped: a position worth
    $0.40 must still be sellable, and Alpaca's own close-position path handles
    liquidating a sub-minimum holding.
    """
    if sizing.min_order_notional <= 0:
        return orders
    out: list[Order] = []
    for o in orders:
        if o.full_exit:
            out.append(o)
            continue
        if o.last_price is None or o.last_price <= 0:
            out.append(o)   # unpriced: nothing to compare, leave the decision alone
            continue
        if o.notional < sizing.min_order_notional:
            logger.info(
                "min notional: skipped {} of {} ({} shares = ${:.2f} < ${:.2f})",
                o.side, o.ticker, fmt_qty(o.shares), o.notional,
                sizing.min_order_notional,
            )
            continue
        out.append(o)
    return out
