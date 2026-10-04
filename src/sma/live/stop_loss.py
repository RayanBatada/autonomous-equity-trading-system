"""The 09:25 ET morning stop-loss sweep.

Five minutes before market open, scans held positions and force-sells any
where current_price <= cost_basis * (1 - stop_loss_pct). The simulator
triggers stop-loss at the OPEN price; broker-side trailing stops trigger at
intra-day prices, which would break paper-vs-sim P&L parity. So we run our
own sweep timed to fire ≤5 min before the open, using Alpaca's pre-market
or last-close price as the proxy for "what the open will likely be".

Phase 5 ships with `RiskRails(stop_loss_pct=0)` per the 2026-04-28 rail
diagnostic. With stop_loss disabled, this sweep is a no-op: zero rows
written, zero submissions made. Future tuning (e.g., raising the threshold
to 15%) just changes config; no code change.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import date, timedelta

from sma.live.alpaca_client import AlpacaClient
from sma.live.decide import _DEAD_ORDER_STATES, _current_holding_entry_dates
from sma.live.orders import client_order_id
from sma.live.quantity import as_qty
from sma.risk.rails import RiskRails
from sma.risk.stop_loss import check_stop_loss, check_take_profit, check_trailing_stop

logger = logging.getLogger(__name__)


@dataclass
class StopLossResult:
    triggered: int
    submitted: int
    failed: int


def stop_loss_sweep(
    *,
    asof: date,
    store,                           # sma.ingest.store.Store (already connected)
    alpaca: AlpacaClient,
    rails: RiskRails,
    price_lookback_days: int = 7,
    run_id: int | None = None,
) -> StopLossResult:
    """Scan positions, force-sell any that hit a PRICE EXIT.

    Applies, in priority order, take-profit > trailing-stop > fixed stop-loss —
    mirroring the simulator's Step-0 scan so live == sim. NOOP when ALL of
    rails.stop_loss_pct / trailing_stop_pct / take_profit_pct are <= 0.

    Live approximation for the trailing stop: the post-entry PEAK is recomputed
    from the store's DAILY raw close since the position was FIRST opened (the
    earliest buy of the current continuous holding, matching the simulator's
    peak window; NOT reset by a later top-up), and the trigger fires on the
    last-close proxy for the coming open — the same raw close-proxy the fixed
    stop and take-profit checks already use, so all three price exits share
    one (raw) basis (2026-07-30 review, Finding 8). Intraday highs are not
    captured. All three exits DEFAULT OFF, so this path is inert until a knob
    is enabled.
    """
    if (
        rails.stop_loss_pct <= 0
        and rails.trailing_stop_pct <= 0
        and rails.take_profit_pct <= 0
    ):
        logger.info(
            "price exits disabled (stop=%s trail=%s take=%s); skipping sweep",
            rails.stop_loss_pct, rails.trailing_stop_pct, rails.take_profit_pct,
        )
        return StopLossResult(triggered=0, submitted=0, failed=0)

    positions = alpaca.get_positions()
    if not positions:
        return StopLossResult(triggered=0, submitted=0, failed=0)

    last_prices = _last_prices_from_store(
        store=store,
        tickers=list(positions.keys()),
        start=asof - timedelta(days=price_lookback_days),
        end=asof,
    )

    # Post-entry peaks (only needed for the trailing stop). Seeded at cost_basis
    # (raw) and raised by the max RAW close since the current holding was FIRST
    # opened (earliest buy of the continuous holding, so a later average-up does
    # NOT restart the peak window and drop the peak below the sim's — that was
    # the live-vs-sim drift bug). A never-updated position still has a sane peak
    # (== entry, trailing off until it makes a new high). Empty when disabled.
    peaks: dict[str, float] = {}
    if rails.trailing_stop_pct > 0:
        entry_dates = _current_holding_entry_dates(
            store=store, tickers=set(positions.keys())
        )
        peaks = _peak_prices_from_store(
            store=store,
            positions=positions,
            entry_dates=entry_dates,
            asof=asof,
        )

    # The CLI allocates the run_id so the sweep's sentinel and the rows it
    # writes carry the SAME id (see _stop_loss_sweep_impl). Direct callers
    # (tests, one-offs) can still let this allocate its own.
    if run_id is None:
        run_id = store.allocate_run_id()
    triggered, submitted, failed = 0, 0, 0
    for ticker, info in positions.items():
        cost_basis = info["cost_basis"]
        shares = info["shares"]
        price_info = last_prices.get(ticker)
        if price_info is None:
            logger.warning("no current price for %s; skipping price-exit check", ticker)
            continue
        raw_price, adj_price, price_date = price_info
        price_age_days = (asof - price_date).days
        if price_age_days > 5:
            logger.warning(
                "price-exit price for %s is stale: latest close %s is %d calendar days "
                "before asof %s",
                ticker,
                price_date,
                price_age_days,
                asof,
            )
        # Priority take-profit > trailing-stop > fixed stop-loss (sim parity).
        #
        # All three checks below share ONE basis convention: RAW (unadjusted)
        # prices, matching cost_basis (Alpaca's raw avg_entry_price). Take-
        # profit and stop-loss compare cost_basis against `raw_price`
        # directly; the trailing-stop's peak (see _peak_prices_from_store) is
        # max(raw cost_basis floor, MAX raw close since entry), also compared
        # against `raw_price`. adj_close is not used in any of these three
        # compares — mixing it in would mean comparing a dividend/split-
        # adjusted price against a cost_basis that never saw that adjustment,
        # which mis-fires (or masks) the exit for any name with a corporate
        # action since entry. This closes out the basis mismatch across all
        # three price exits (2026-07-30 review, Finding 8 — fully remediated).
        #
        # Split caveat (documented once, applies to all three): Alpaca
        # retroactively adjusts avg_entry_price on a split, but the raw close
        # series here is not re-based off that adjustment, so a split between
        # entry and now can still skew any of these three compares.
        is_triggered, reason = check_take_profit(
            rails=rails, ticker=ticker,
            cost_basis=cost_basis, current_price=raw_price,
        )
        if not is_triggered:
            is_triggered, reason = check_trailing_stop(
                rails=rails, ticker=ticker,
                peak_price=peaks.get(ticker, cost_basis), current_price=raw_price,
            )
        if not is_triggered:
            is_triggered, reason = check_stop_loss(
                rails=rails, ticker=ticker,
                cost_basis=cost_basis, current_price=raw_price,
            )
        if not is_triggered:
            continue

        triggered += 1
        # Cancel last night's still-open decide order(s) for this name BEFORE
        # selling: a pending OPG BUY would otherwise re-open the position at the
        # same 09:30 open we're selling into (a no-op round-trip paying double
        # slippage — the sim's _exited_today guard), and a pending decide SELL
        # would double-submit against our full-exit sell (oversell). FAIL CLOSED:
        # if a cancel can't be CONFIRMED (still live / unknown), do NOT submit the
        # stop sell — a live opposing order + our sell would oversell/short.
        if not _cancel_pending_decide_orders(ticker=ticker, store=store, alpaca=alpaca):
            logger.error(
                "stop-loss: could not confirm-cancel a pending decide order for "
                "%s; NOT selling (fail closed) to avoid oversell — manual review",
                ticker,
            )
            from sma.ingest.notify import notify_failure
            notify_failure(
                title="sma: stop-loss fail-closed",
                message=(
                    f"{ticker} tripped a price exit but a pending decide order "
                    "could not be confirmed-canceled; the stop sell was NOT "
                    "submitted to avoid an oversell/short. Resolve manually."
                ),
            )
            failed += 1
            continue
        if _submit_stop_sell(
            ticker=ticker, shares=shares, last_price=adj_price,
            asof=asof, store=store, alpaca=alpaca, run_id=run_id, reason=reason,
        ):
            submitted += 1
        else:
            failed += 1

    return StopLossResult(triggered=triggered, submitted=submitted, failed=failed)


# Decide-order lifecycle statuses that are still cancelable at the broker (the
# order hasn't reached a terminal state yet). At 09:25 the prior evening's decide
# batch is unreconciled, so its orders sit in these pre-terminal states.
_OPEN_DECIDE_STATES = ("submitted", "recovery_failed", "submission_failed")


# Broker statuses that mean a canceled order is genuinely gone with no new fill,
# so it can't oversell against our stop sell. Anything else (filled/partially/
# accepted/new/pending/held/unknown) means we can't safely proceed.
_CONFIRMED_GONE_STATES = ("canceled", "cancelled", "expired", "rejected")


def _order_confirmed_gone(alpaca: AlpacaClient, order_id: str) -> bool:
    """After a cancel raised, re-fetch the order and return True only if it is
    confirmed terminal WITHOUT a new fill. Any live/filled/partially-filled/
    unknown state → False (the caller must fail closed). A partial fill before the
    cancel moved the book, so the pre-sweep share count is stale — not safe."""
    try:
        order = alpaca.get_order_by_id(order_id)
    except Exception:  # noqa: BLE001 — can't confirm → not gone
        return False
    status = str(getattr(order.status, "value", order.status)).lower()
    if status not in _CONFIRMED_GONE_STATES:
        return False
    filled = as_qty(getattr(order, "filled_qty", 0) or 0)
    return filled == 0


def _cancel_pending_decide_orders(*, ticker: str, store, alpaca: AlpacaClient) -> bool:
    """Cancel every still-open decide order for `ticker`; return True iff it is
    SAFE to submit the stop sell (all pending orders confirmed canceled/gone).

    For each open decide order (from our own intended_orders ledger): mark it
    status='canceled_by_stop' BEFORE the broker cancel — crash-safe, so a death
    between the cancel and the DB write still leaves the intent recorded and
    reconcile won't misread it as a missed buy. Then cancel at Alpaca. If the
    cancel raises, re-fetch: a confirmed-terminal order is benign; a still-live or
    unknown one returns False so the caller fails closed (skips the sell + pages)
    rather than oversell against a live opposing order (adversarial review
    2026-07-04).
    """
    placeholders = ", ".join("?" for _ in _OPEN_DECIDE_STATES)
    # NOT filtered on alpaca_order_id IS NOT NULL: a crash-after-accept decide row
    # (status='submitted', id=NULL) can still be a LIVE order at the broker; the
    # broker id is recovered below via its deterministic coid (re-review 2026-07-04).
    rows = store.conn.execute(
        f"SELECT intended_order_id, alpaca_order_id, asof_date, side FROM intended_orders "
        f"WHERE ticker = ? AND source = 'decide' AND status IN ({placeholders})",
        [ticker, *_OPEN_DECIDE_STATES],
    ).fetchall()
    ok = True
    for intended_id, order_id, o_asof, o_side in rows:
        if order_id is None:
            # Recover the broker id (decide's coid embeds no source).
            try:
                found = alpaca.get_order_by_client_order_id(
                    client_order_id(o_asof, ticker, o_side)
                )
            except Exception as e:  # noqa: BLE001 — can't confirm live-ness
                logger.error(
                    "stop-loss: coid recovery for a null-id decide order of %s "
                    "failed: %s; failing closed for this name", ticker, e,
                )
                ok = False
                continue
            if found is None:
                continue  # broker never accepted it — nothing live to cancel
            order_id, brk_status = found
            if brk_status in _CONFIRMED_GONE_STATES:
                continue  # already terminal at the broker — nothing to cancel
        # Record the deliberate-cancel intent FIRST (crash-safe), backfilling any
        # recovered id so reconcile can still record a fill if the cancel fails.
        store.conn.execute(
            "UPDATE intended_orders SET status = 'canceled_by_stop', alpaca_order_id = ? "
            "WHERE intended_order_id = ?",
            [order_id, intended_id],
        )
        try:
            alpaca.cancel_order(order_id)
        except Exception as e:
            if _order_confirmed_gone(alpaca, order_id):
                logger.warning(
                    "stop-loss: cancel of decide order %s for %s raised but the "
                    "order is confirmed terminal (%s); proceeding",
                    order_id, ticker, e,
                )
                continue
            logger.error(
                "stop-loss: cancel of decide order %s for %s failed and the order "
                "is still live/unknown: %s; failing closed for this name",
                order_id, ticker, e,
            )
            ok = False
            continue
        logger.info(
            "stop-loss: canceled pending decide order %s for %s (exiting the name)",
            order_id, ticker,
        )
    return ok


def _submit_stop_sell(
    *, ticker: str, shares: float, last_price: float,
    asof: date, store, alpaca: AlpacaClient, run_id: int, reason: str,
) -> bool:
    """Write intended_orders BEFORE submit; recover crash-window rows idempotently."""
    coid = client_order_id(asof, ticker, "SELL", source="stop-loss")

    existing = store.conn.execute(
        "SELECT intended_order_id, alpaca_order_id, status FROM intended_orders "
        "WHERE asof_date = ? AND ticker = ? AND source = 'stop-loss'",
        [asof, ticker],
    ).fetchone()
    if existing is not None and existing[1] is not None:
        logger.warning(
            "skip: stop-loss %s already submitted earlier today "
            "(alpaca_order_id=%s, status=%s); preventing double-submit",
            ticker,
            existing[1],
            existing[2],
        )
        return True
    if existing is not None:
        try:
            found = alpaca.get_order_by_client_order_id(coid)
        except Exception as e:
            logger.error(
                "stop-loss recovery lookup failed for %s (coid=%s): %s; NOT "
                "resubmitting (fail closed to avoid a double-submit)",
                ticker,
                coid,
                e,
            )
            store.conn.execute(
                "UPDATE intended_orders SET status = 'recovery_failed', error = ? "
                "WHERE intended_order_id = ?",
                [f"recovery lookup failed: {e!r}", existing[0]],
            )
            return False
        if found is not None:
            rec_id, rec_status = found
            if rec_status not in _DEAD_ORDER_STATES:
                logger.warning(
                    "recover: stop-loss %s already at Alpaca (id=%s, status=%s, "
                    "coid=%s); adopting instead of resubmitting",
                    ticker,
                    rec_id,
                    rec_status,
                    coid,
                )
                store.conn.execute(
                    "UPDATE intended_orders SET alpaca_order_id = ? WHERE intended_order_id = ?",
                    [rec_id, existing[0]],
                )
                return True
            logger.warning(
                "recover: stop-loss %s prior order terminal at broker "
                "(status=%s, coid=%s); not resubmitting today",
                ticker,
                rec_status,
                coid,
            )
            store.conn.execute(
                "UPDATE intended_orders SET status = 'submission_failed', error = ? "
                "WHERE intended_order_id = ?",
                [f"prior order terminal at broker: {rec_status}", existing[0]],
            )
            return False
        store.conn.execute(
            "DELETE FROM intended_orders WHERE asof_date = ? AND ticker = ? "
            "AND source = 'stop-loss'",
            [asof, ticker],
        )

    intended_id = str(uuid.uuid4())
    store.conn.execute("""
        INSERT INTO intended_orders
        (intended_order_id, asof_date, ticker, side, target_shares,
         target_weight, last_price, source, status, run_id)
        VALUES (?, ?, ?, 'SELL', ?, NULL, ?, 'stop-loss', 'submitted', ?)
    """, [intended_id, asof, ticker, shares, last_price, run_id])
    try:
        alpaca_id = alpaca.submit_market_sell(
            ticker, shares, client_order_id=coid
        )
        store.conn.execute(
            "UPDATE intended_orders SET alpaca_order_id = ? "
            "WHERE intended_order_id = ?",
            [alpaca_id, intended_id],
        )
        logger.info("stop-loss sell submitted for %s: %s", ticker, reason)
        return True
    except Exception as e:
        logger.exception("stop-loss submit failed for %s: %s", ticker, e)
        store.conn.execute(
            "UPDATE intended_orders SET status = 'submission_failed', error = ? "
            "WHERE intended_order_id = ?",
            [str(e), intended_id],
        )
        return False


def _peak_prices_from_store(
    *, store, positions: dict, entry_dates: dict[str, date], asof: date,
) -> dict[str, float]:
    """Post-entry peak RAW close per ticker, for the trailing stop.

    Peak = max(cost_basis, max close in [entry_date, asof]), where
    `entry_date` is the FIRST buy of the current continuous holding (see
    `_current_holding_entry_dates`). Mirrors the simulator's peak-tracking
    LOGIC (seeded at the entry fill, raised by each subsequent close, kept
    across a top-up) but not the simulator's basis: this uses the RAW close,
    not adj_close, so the peak shares one basis with cost_basis (Alpaca's
    raw avg_entry_price) and current_price — closing out the last piece of
    the raw-vs-adjusted mismatch that used to affect this rail (a
    never-updated position's peak floors at cost_basis; before this fix that
    floor was raw while the historical high-water mark was adj_close, which
    could itself mis-trigger against an adj_close current_price — 2026-07-30
    review, Finding 8). Tickers without a known entry date fall back to a
    cost_basis-only peak (trailing stays off until they make a new recorded
    high). LIVE APPROXIMATION: daily closes only, no intraday highs, and
    cost_basis (weighted avg) rather than each individual fill price.
    """
    peaks: dict[str, float] = {}
    for ticker, info in positions.items():
        basis = float(info.get("cost_basis", 0.0) or 0.0)
        peaks[ticker] = basis
        entry = entry_dates.get(ticker)
        if entry is None:
            continue
        row = store.conn.execute(
            "SELECT MAX(close) FROM prices "
            "WHERE ticker = ? AND date BETWEEN ? AND ? AND close IS NOT NULL",
            [ticker, entry, asof],
        ).fetchone()
        if row is not None and row[0] is not None:
            peaks[ticker] = max(basis, float(row[0]))
    return peaks


def _last_prices_from_store(
    *, store, tickers: list[str], start: date, end: date,
) -> dict[str, tuple[float, float, date]]:
    """Most-recent (raw close, adj_close) per ticker between start and end
    (inclusive).

    `close` (raw) is what all three price-exit compares use — take-profit,
    trailing-stop, and the fixed stop-loss — so each is raw-vs-raw against
    cost_basis (Alpaca's raw, unadjusted avg_entry_price; 2026-07-30 review,
    Finding 8). `adj_close` is kept only for the submitted stop order's
    reported last_price, not for any trigger decision.
    """
    rows = store.conn.execute("""
        SELECT ticker, close, adj_close, date
        FROM (
            SELECT ticker, date, close, adj_close,
                   ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date DESC) AS rn
            FROM prices
            WHERE ticker = ANY($tickers)
              AND date BETWEEN $start AND $end
              AND close IS NOT NULL
              AND adj_close IS NOT NULL
        ) t
        WHERE rn = 1
    """, {"tickers": tickers, "start": start, "end": end}).fetchall()
    return {row[0]: (float(row[1]), float(row[2]), row[3]) for row in rows}
