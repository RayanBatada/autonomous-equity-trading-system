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
from sma.live.decide import _DEAD_ORDER_STATES
from sma.live.orders import client_order_id
from sma.risk.rails import RiskRails
from sma.risk.stop_loss import check_stop_loss

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
) -> StopLossResult:
    """Scan positions, force-sell any below stop-loss threshold.

    NOOP when rails.stop_loss_pct <= 0.
    """
    if rails.stop_loss_pct <= 0:
        logger.info("stop_loss disabled (stop_loss_pct=%s); skipping sweep",
                    rails.stop_loss_pct)
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

    run_id = store.allocate_run_id()
    triggered, submitted, failed = 0, 0, 0
    for ticker, info in positions.items():
        cost_basis = info["cost_basis"]
        shares = info["shares"]
        price_info = last_prices.get(ticker)
        if price_info is None:
            logger.warning("no current price for %s; skipping stop-loss check", ticker)
            continue
        current_price, price_date = price_info
        price_age_days = (asof - price_date).days
        if price_age_days > 5:
            logger.warning(
                "stop-loss price for %s is stale: latest close %s is %d calendar days "
                "before asof %s",
                ticker,
                price_date,
                price_age_days,
                asof,
            )
        is_triggered, reason = check_stop_loss(
            rails=rails, ticker=ticker,
            cost_basis=cost_basis, current_price=current_price,
        )
        if not is_triggered:
            continue

        triggered += 1
        if _submit_stop_sell(
            ticker=ticker, shares=shares, last_price=current_price,
            asof=asof, store=store, alpaca=alpaca, run_id=run_id, reason=reason,
        ):
            submitted += 1
        else:
            failed += 1

    return StopLossResult(triggered=triggered, submitted=submitted, failed=failed)


def _submit_stop_sell(
    *, ticker: str, shares: int, last_price: float,
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


def _last_prices_from_store(
    *, store, tickers: list[str], start: date, end: date,
) -> dict[str, tuple[float, date]]:
    """Most-recent adj_close per ticker between start and end (inclusive)."""
    rows = store.conn.execute("""
        SELECT ticker, adj_close, date
        FROM (
            SELECT ticker, date, adj_close,
                   ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date DESC) AS rn
            FROM prices
            WHERE ticker = ANY($tickers)
              AND date BETWEEN $start AND $end
              AND adj_close IS NOT NULL
        ) t
        WHERE rn = 1
    """, {"tickers": tickers, "start": start, "end": end}).fetchall()
    return {row[0]: (float(row[1]), row[2]) for row in rows}
