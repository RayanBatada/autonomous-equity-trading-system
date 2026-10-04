"""Renders tonight's SUBMITTED decide orders as a short ntfy push so Rayan
can manually mirror them on his own account.

Deliberately reports percent-of-equity, never share counts, so one message
is valid at any account size. This module is pure -- it takes the
already-computed Order/StrategyDecision structures decide_once produces and
formats them; it never touches the network, the DB, or Alpaca. The caller
(sma.live.decide.decide_once) builds the message right after the submit
loop, and sma.live.__main__ sends it via sma.ingest.notify.send_ntfy at the
end of the decide CLI flow, after orders are submitted and the sentinel is
written -- see docs/REAL_MONEY_CHECKLIST.md "Mirroring signals manually".

TWO DIFFERENT PERCENTAGES, BY DESIGN (2026-09-03 bug fix). A full-exit SELL
reports the PRIOR POSITION's weight ("was X% of book") -- useful context
for a full liquidation, since the whole position is sold regardless of its
size. Every other line -- a partial SELL trim, or a BUY, fresh entry or a
top-up of an existing position -- reports THIS ORDER's own notional as % of
equity: a manual mirrorer trades the order, not the position it leaves
behind. Reading the decision's post-trade TARGET weight for a partial order
was exactly the bug: a SELL of 1-of-6 LLY shares (~1.0% of equity) was
pushed as "5.7% of equity" -- LLY's post-trade POSITION weight -- a manual
mirrorer would have sold 5.7x too much.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

from sma.backtest.strategies.base import StrategyDecision
from sma.live.orders import Order
from sma.sentinels import write_sentinel

# ntfy payloads render fine well past this, but 600 keeps a push readable at
# a glance on a lock screen notification banner.
MAX_MESSAGE_CHARS = 600

# 2026-09-04: sentinel label record_trade_push writes under -- see its
# docstring. Read back by sma.live.reconcile._detect_trade_push_drift.
TRADE_PUSH_SENTINEL_LABEL = "com.sma.trade-push"


def build_trade_push(
    *,
    asof: date,
    submitted_orders: list[Order],
    decisions: list[StrategyDecision],
    equity: float,
    position_count: int,
) -> tuple[str, str]:
    """Return (title, message) for tonight's trade push.

    `submitted_orders` must be ONLY the orders whose outcome was actually
    "submitted" (fresh submit or crash-recovery adoption) -- failed/skipped
    orders are reported by the existing notify_failure() page, not here.
    Order is preserved as given (decide_once already emits sells before
    buys); this function never re-sorts.
    """
    # ASCII only: this string becomes an HTTP header (see send_ntfy).
    title = f"SMA trades - {asof.strftime('%a')} {asof.month}/{asof.day}"

    # `decisions` is accepted (and still threaded through by every caller) but
    # deliberately unused below: every order line renders from the ORDER
    # itself (see module docstring) -- a decision's target_weight is the
    # post-trade POSITION weight, not this order's size, and using it here is
    # exactly the bug this function was rewritten to fix (2026-09-03).
    date_line = asof.isoformat()
    order_lines = [_order_line(o, equity) for o in submitted_orders]
    footer = _footer(equity, position_count)

    if not order_lines:
        holds_noun = "hold" if position_count == 1 else "holds"
        body = [
            date_line,
            f"No trades tonight — book unchanged ({position_count} {holds_noun})",
        ]
    else:
        body = [date_line, *order_lines]

    message = "\n".join([*body, footer])
    if len(message) > MAX_MESSAGE_CHARS:
        message = _truncate(date_line, order_lines, footer)
    return title, message


def build_push_orders_payload(
    *, submitted_orders: list[Order], equity: float,
) -> list[dict]:
    """Structured per-order record of tonight's push, for persistence + later
    verification against booked fills (see record_trade_push below and
    sma.live.reconcile._detect_trade_push_drift).

    Deliberately shares the EXACT math `_order_line` renders into the pushed
    text (`order.notional / equity`) -- the whole point is that the persisted
    numbers can never diverge from what Rayan actually read on his phone.
    Note this math is IDENTICAL for a full exit and every other order: a full
    exit's rendered "(was X% of book)" and every other line's "X% of equity"
    are both `shares * last_price / equity` (see _order_line) -- there is no
    separate full-exit branch needed here.
    """
    return [_push_order_payload(o, equity) for o in submitted_orders]


def _push_order_payload(order: Order, equity: float) -> dict:
    # order_notional depends only on price -- it's a real dollar figure
    # regardless of what the account's equity happens to be. order.notional
    # itself is 0.0 (never None) when last_price is falsy, so gate explicitly
    # rather than trust its default.
    notional = order.notional if order.last_price else None
    pct = notional / equity if (notional is not None and equity > 0) else None
    return {
        "ticker": order.ticker,
        "side": order.side,
        "shares": order.shares,
        "decide_price": order.last_price,
        "order_notional": notional,
        "order_pct_of_equity": pct,
        "full_exit": order.full_exit,
    }


def record_trade_push(
    *,
    asof: date,
    title: str | None,
    message: str | None,
    orders: list[dict],
    equity: float,
    delivered: bool,
) -> None:
    """Persist tonight's push at send time: data/sentinels/com.sma.trade-push-
    <asof>.json, via the existing atomic sentinel writer.

    Written even when the send itself failed (`delivered=False`) -- what was
    CLAIMED matters for verification regardless of whether ntfy actually
    delivered it. ntfy.sh's own cache expires in 12h, so without this file
    nothing lets a push be checked against what actually filled after the
    fact (2026-09-04: this is what would have caught the LLY 5x-overstated
    push automatically instead of a human eyeballing it -- see
    sma.live.reconcile._detect_trade_push_drift, which reads this file back).

    Caller (sma.live.__main__) is responsible for never letting a persist
    failure here break the decide CLI -- same discipline as the send itself.
    """
    payload = {
        "label": TRADE_PUSH_SENTINEL_LABEL,
        "asof": asof.isoformat(),
        "completed_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "title": title,
        "message": message,
        "delivered": delivered,
        "equity": equity,
        "orders": orders,
    }
    write_sentinel(label=TRADE_PUSH_SENTINEL_LABEL, asof=asof, payload=payload)


def _footer(equity: float, position_count: int) -> str:
    positions_noun = "position" if position_count == 1 else "positions"
    return f"Equity ${equity:,.0f} | {position_count} {positions_noun}"


def _order_line(order: Order, equity: float) -> str:
    if order.side == "SELL" and order.full_exit:
        # A full exit's `shares` IS the entire prior position (that is what
        # full_exit means -- see orders.translate), so shares * last_price
        # is exactly the dollar value being liquidated. last_price can be
        # None for a force-sell of a priceless/delisted ticker (orders.py
        # never gates a full exit on price freshness); degrade to a bare
        # "all" rather than crash on None * float.
        if order.last_price and equity > 0:
            prior_pct = (order.shares * order.last_price) / equity
            return f"SELL {order.ticker} — all (was {prior_pct:.1%} of book)"
        return f"SELL {order.ticker} — all"

    # Every other order -- a partial SELL trim, or a BUY (fresh entry or a
    # top-up of an existing position) -- reports THIS ORDER's own notional,
    # never the decision's post-trade target weight (see module docstring).
    # `order.last_price` is the decide-time price orders.translate() used to
    # size `order.shares` in the first place (see Order.last_price), so
    # `order.notional` (= shares * that price, via the Order property) is
    # exactly this order's dollar size -- the same price reference the
    # full-exit branch above uses for "of book".
    if order.last_price and equity > 0:
        order_pct = order.notional / equity
        return f"{order.side} {order.ticker} — {order_pct:.1%} of equity"
    return f"{order.side} {order.ticker} — (weight unavailable)"


def _truncate(date_line: str, order_lines: list[str], footer: str) -> str:
    """Keep as many leading order_lines as fit under MAX_MESSAGE_CHARS,
    noting how many were dropped. Deterministic: always keeps the earliest
    lines (decide_once's sells-then-buys, highest-conviction-first order),
    never re-orders or samples."""
    n = len(order_lines)
    for keep in range(n, -1, -1):
        omitted = n - keep
        lines = [date_line, *order_lines[:keep]]
        if omitted:
            lines.append(f"...(+{omitted} more)")
        lines.append(footer)
        message = "\n".join(lines)
        if len(message) <= MAX_MESSAGE_CHARS:
            return message
    # Pathological: date_line + footer alone exceed the bound. Hard-clip as
    # a last resort -- never raise out of a notification path.
    message = "\n".join([date_line, footer])
    if len(message) > MAX_MESSAGE_CHARS:
        message = message[: MAX_MESSAGE_CHARS - 1] + "…"
    return message
