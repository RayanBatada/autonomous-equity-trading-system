"""The 18:35 ET daily decide job.

Lifecycle:
  1. Pre-flight runs BEFORE this function (caller: __main__.py) and reads
     sentinels only. Caller acquires writer_lock, then opens Store, then
     calls decide_once. This ordering avoids the writer-blocks-self deadlock
     that occurs when a writable Store conn is held during preflight retries.
  2. Read most-recent prices from DuckDB.
  3. Compute last_prices: ticker → (price, price_date).
  4. Run strategy.decide(asof_date) → list[StrategyDecision].
  5. Apply --canary filter if set (debug only — not part of rollout).
  6. Build RiskContext from Alpaca account/positions + DB.
  7. Apply sma.risk.pipeline.apply → adjusted decisions.
  8. Empty-decisions paranoia rail (spec §7 step 8).
  9. orders.translate → list[Order].
 10. For each order: write intended_orders BEFORE submit, then submit, then
     update intended_orders with the alpaca_order_id (or status='submission_failed').
 11. (--dry-run skips step 10.)
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from sma.backtest.strategies.base import StrategyDecision
from sma.live.alpaca_client import AlpacaClient
from sma.live.exceptions import EmptyDecisionsWithHeldPositionsError
from sma.live.orders import Order, client_order_id, translate
from sma.risk.pipeline import RiskContext
from sma.risk.pipeline import apply as apply_rails
from sma.risk.rails import RiskRails

logger = logging.getLogger(__name__)


_ET = ZoneInfo("America/New_York")


def _opg_window_open(now: datetime | None = None) -> bool:
    """True if Alpaca's OPG-order acceptance window is open right now.

    Alpaca accepts DAY_OPG submissions between 19:00 ET (previous evening)
    and 09:28 ET (auction cutoff). Outside this window the API returns
    `40310000: opg orders must be submitted after 7:00pm and before 9:28am`.
    Decide uses this to choose between OPG (preferred, single-shot at open
    auction) and DAY market (fallback, queues until next session open).
    """
    if now is None:
        now = datetime.now(_ET)
    t = now.timetz().replace(tzinfo=None)
    return t >= time(19, 0) or t < time(9, 28)


@dataclass
class DecideResult:
    submitted: int
    failed: int
    dry_run: bool
    decisions_after_rails: int
    skipped: int = 0


class CatastrophicLossAbortError(Exception):
    """Raised by decide_once when equity drop vs yesterday exceeds the abort
    threshold. Stops live trading until an operator investigates and
    manually clears the condition (e.g. by waiving via env var)."""


# Backward-compatible alias for callers that imported the original name.
CatastrophicLossAbort = CatastrophicLossAbortError


def _drawdown_from_peak(peak_equity: float, current_equity: float) -> float:
    """Fraction below the running equity peak, in [0, 1].

    0.0 when at/above the peak (incl. a non-positive peak). Used to drive the
    live max_drawdown rail — previously hardcoded to 0.0 in decide_once, so the
    rail never fired (only the separate catastrophic-loss abort remained)."""
    if peak_equity <= 0:
        return 0.0
    return max(0.0, (peak_equity - current_equity) / peak_equity)


def decide_once(
    *,
    asof: date,
    store,  # sma.ingest.store.Store (already connected)
    alpaca: AlpacaClient,
    universe: list[str],
    strategy,  # any object with .decide(asof_date, prices)
    sector_for: Callable[[str], str],
    rails: RiskRails,
    catastrophic_loss_abort_pct: float = 0.30,
    price_lookback_days: int = 10,
    canary: str | None = None,
    dry_run: bool = False,
) -> DecideResult:
    """Run one full decide cycle. Pre-flight must be run by caller before this.

    Caller (sma.live.__main__) is responsible for:
      1. Calling run_preflight(asof=asof, db_path=..., alpaca=alpaca) BEFORE
         acquiring the writer lock or opening the Store.
      2. Acquiring writer_lock(label="decide").
      3. Opening the writable Store.
      4. Calling this function.

    This function fetches account/positions directly from Alpaca (no DB needed
    for those reads) then proceeds with strategy + risk rails + order submission.
    """
    account = alpaca.get_account()
    positions = alpaca.get_positions()

    # Catastrophic-loss circuit breaker (codex HIGH 2026-05-14). Pre-existing
    # reconcile detection only emitted an alert; nothing actually blocked the
    # next decide. If today's live equity has dropped >= the abort threshold
    # vs yesterday's recorded snapshot, refuse to trade until an operator
    # clears the condition. Skips gracefully when no prior snapshot exists.
    today_equity = float(account.get("equity", 0))
    last_snap = store.conn.execute(
        "SELECT equity FROM account_snapshots WHERE asof_date < ? "
        "ORDER BY asof_date DESC LIMIT 1",
        [asof],
    ).fetchone()
    if last_snap is not None and last_snap[0]:
        prior_eq = float(last_snap[0])
        drop = (prior_eq - today_equity) / prior_eq
        if drop >= catastrophic_loss_abort_pct:
            raise CatastrophicLossAbortError(
                f"equity dropped {drop:.1%} since prior snapshot "
                f"(${prior_eq:,.0f} → ${today_equity:,.0f}); decide aborted "
                f"(threshold={catastrophic_loss_abort_pct:.1%}). Review + "
                "clear before resuming."
            )

    # Live drawdown for the max_drawdown rail: fraction below the running equity
    # peak (account_snapshots history + today's equity). Was hardcoded to 0.0,
    # which disabled the rail entirely (#1 from the live-path audit).
    peak_row = store.conn.execute(
        "SELECT MAX(equity) FROM account_snapshots WHERE asof_date <= ?",
        [asof],
    ).fetchone()
    peak_equity = max(
        float(peak_row[0]) if peak_row and peak_row[0] else 0.0,
        today_equity,
    )
    current_drawdown = _drawdown_from_peak(peak_equity, today_equity)

    prices = _load_prices(
        store=store, universe=universe, start=asof - timedelta(days=price_lookback_days), end=asof
    )
    last_prices = _last_prices_per_ticker(prices, universe)
    upcoming_earnings = _load_upcoming_earnings_from_store(
        store=store,
        start=asof,
        end=asof + timedelta(days=14),
    )

    # Pass live holdings when the strategy supports them (signature check
    # mirrors simulator.py). Without this the bearish-thesis veto saw held=∅
    # and force-sold merely-bearish HELD names in live trading — the exact
    # regression the 2e2c085 fix removed from the backtest path (Codex
    # post-audit review, 2026-06-09).
    import inspect

    if "current_holdings" in inspect.signature(strategy.decide).parameters:
        raw_decisions = strategy.decide(
            asof_date=asof,
            prices=prices,
            current_holdings=set(positions.keys()),
        )
    else:
        raw_decisions = strategy.decide(asof_date=asof, prices=prices)
    # Capture the strategy's raw approval set BEFORE rails. translate() uses
    # this to distinguish "model dropped this name" (force-sell held position)
    # from "rails blocked the re-buy" (keep held position).
    model_approved_tickers = {d.ticker for d in raw_decisions}
    if canary is not None:
        # Force-include the canary ticker. The point of canary mode is to
        # exercise the live submit path with a known ticker — not to test
        # strategy logic. If the strategy didn't surface this ticker, we
        # synthesize one decision at max_position_pct (5% default), then
        # discard the rest. Risk rails still apply (earnings blackout,
        # sector cap, etc.) so the canary can still legitimately be rejected.
        existing = next(
            (d for d in raw_decisions if d.ticker == canary),
            None,
        )
        if existing is not None:
            raw_decisions = [existing]
            logger.info(
                "canary mode: kept strategy-surfaced %s (weight=%.4f)",
                canary,
                existing.target_weight,
            )
        else:
            raw_decisions = [
                StrategyDecision(
                    asof_date=asof,
                    ticker=canary,
                    target_weight=rails.max_position_pct,
                )
            ]
            logger.info(
                "canary mode: synthesized %s @ max_position_pct=%.2f%% (not in strategy top-K)",
                canary,
                rails.max_position_pct * 100,
            )

    # Entry dates feed BOTH translate's min_hold rail and the risk pipeline's
    # sector-room reservation: a held name whose full exit min_hold will veto
    # must keep its sector room reserved (Codex module review 2026-06-11).
    position_entry_dates = _latest_buy_dates(
        store=store, tickers=set(positions.keys()),
    )
    _min_hold = rails.min_hold_days if rails is not None else 0
    min_hold_protected = frozenset(
        t for t, entry in position_entry_dates.items()
        if _min_hold > 0
        and entry is not None
        and 0 <= (asof - entry).days < _min_hold
        and positions.get(t, {}).get("shares", 0) > 0
    )

    risk_ctx = RiskContext(
        rails=rails,
        account_value=account["equity"],
        cash=account["cash"],
        current_positions_dollars=_to_dollars(positions, last_prices),
        sector_exposure_pct=_sector_exposure(
            positions,
            last_prices,
            account["equity"],
            sector_for,
        ),
        sector_for=sector_for,
        current_drawdown=current_drawdown,  # computed from equity peak above
        upcoming_earnings=upcoming_earnings,
        asof_date=asof,
        min_hold_protected=min_hold_protected,
    )
    decisions = apply_rails(raw_decisions, risk_ctx)
    logger.info("rails: %d → %d decisions", len(raw_decisions), len(decisions))

    current_position_shares = {t: p["shares"] for t, p in positions.items()}

    # Position-entry dates were computed above (shared with the risk
    # pipeline's min-hold sector-room reservation). Most-recent BUY in
    # paper_fills = "entered on that date"; tickers absent from paper_fills
    # (pre-seeded accounts) get no min-hold protection.

    # Paranoia rail: 0 decisions + non-empty positions usually indicates a
    # programming bug (model died, universe loaded empty, etc.). BUT: if
    # min_hold is active and EVERY held position is still inside the
    # min_hold window, "do nothing today" is a legitimate outcome — we
    # protect those positions and skip the day's trading. Only raise when
    # at least one held position has aged past min_hold or has no entry
    # date (those are the names a real bug would force us to leave
    # un-managed). Codex round-2 MED 1, fixed 2026-05-13.
    if not decisions and current_position_shares:
        min_hold = rails.min_hold_days if rails is not None else 0
        unprotected = []
        for ticker, shares in current_position_shares.items():
            if shares <= 0:
                continue
            entry_date = position_entry_dates.get(ticker)
            if entry_date is None:
                unprotected.append(ticker)
                continue
            held_days = (asof - entry_date).days
            # Negative held_days (bad backfill / clock skew) treated as
            # not-fresh → unprotected → fires the paranoia rail.
            if held_days < 0 or held_days >= min_hold:
                unprotected.append(ticker)
        if min_hold <= 0 or unprotected:
            raise EmptyDecisionsWithHeldPositionsError(
                f"strategy + rails returned 0 decisions but {len(current_position_shares)} "
                f"positions held ({len(unprotected)} past min_hold or without "
                "entry date); aborting batch (paranoia rail)"
            )
        logger.warning(
            "0 decisions + %d held positions, all inside min_hold window; "
            "skipping trading for today",
            len(current_position_shares),
        )

    orders = translate(
        decisions=decisions,
        current_positions=current_position_shares,
        account_equity=account["equity"],
        cash=account["cash"],
        last_prices=last_prices,
        asof_date=asof,
        rails=rails,
        model_approved_tickers=model_approved_tickers,
        position_entry_dates=position_entry_dates,
        current_drawdown=current_drawdown,
    )
    logger.info("translated %d decisions → %d orders", len(decisions), len(orders))

    if dry_run:
        for o in orders:
            price_str = f"${o.last_price:.2f}" if o.last_price else "N/A"
            print(f"  [dry-run] {o.side:4s} {o.shares:5d} {o.ticker:6s} @ ~{price_str} ({o.type})")
        return DecideResult(
            submitted=0,
            failed=0,
            dry_run=True,
            decisions_after_rails=len(decisions),
        )

    run_id = store.allocate_run_id()
    submitted, failed, skipped = 0, 0, 0
    for o in orders:
        outcome = _submit_with_audit_trail(
            o, asof=asof, store=store, alpaca=alpaca, run_id=run_id, decisions=decisions
        )
        if outcome == "submitted":
            submitted += 1
        elif outcome == "skipped":
            skipped += 1
        else:
            failed += 1

    return DecideResult(
        submitted=submitted,
        failed=failed,
        skipped=skipped,
        dry_run=False,
        decisions_after_rails=len(decisions),
    )


# Broker order statuses that mean the order will NOT result in a position
# (won't fill). Anything else (new/accepted/pending_new/partially_filled/filled/
# held/...) is treated as live and ADOPTED on recovery rather than resubmitted.
_DEAD_ORDER_STATES = frozenset(
    {"canceled", "cancelled", "rejected", "expired", "replaced", "done_for_day"}
)


def _submit_with_audit_trail(
    order: Order,
    *,
    asof: date,
    store,
    alpaca: AlpacaClient,
    run_id: int,
    decisions,
) -> str:
    """Write intended_orders BEFORE Alpaca submit so audit row exists even on hang.

    Returns one of "submitted", "skipped", or "failed".

    Retry-safe semantics on the unique key (asof_date, ticker, source='decide'):
      - If a prior row exists with `alpaca_order_id IS NOT NULL` → it was
        already accepted by Alpaca; SKIP this submit attempt to prevent
        double-submit. Returns "skipped".
      - If a prior row exists with `alpaca_order_id IS NULL` → previous
        attempt failed BEFORE Alpaca accepted (constraint, network, OPG-window
        rejection, etc.); DELETE the failed row and proceed to (re-)insert
        + submit.
    """
    # Deterministic broker client_order_id for (asof, ticker, side). A retry
    # submits the SAME id, which Alpaca rejects as a duplicate, and it lets us
    # recover an order that was accepted just before a crash recorded its id.
    coid = client_order_id(asof, order.ticker, order.side)

    existing = store.conn.execute(
        "SELECT intended_order_id, alpaca_order_id, status FROM intended_orders "
        "WHERE asof_date = ? AND ticker = ? AND source = 'decide'",
        [asof, order.ticker],
    ).fetchone()
    if existing is not None and existing[1] is not None:
        logger.warning(
            "skip: %s already submitted earlier today (alpaca_order_id=%s, "
            "status=%s); preventing double-submit",
            order.ticker,
            existing[1],
            existing[2],
        )
        return "skipped"
    if existing is not None:
        # A prior attempt left a NULL alpaca_order_id: it either failed BEFORE
        # Alpaca accepted (network/constraint/OPG-window) OR crashed AFTER Alpaca
        # accepted but before the id was recorded. Ask Alpaca by client_order_id.
        # FAIL CLOSED on any lookup error — treating a lookup failure as "no
        # order" would resubmit an order that may already be live (double-submit).
        try:
            found = alpaca.get_order_by_client_order_id(coid)
        except Exception as e:
            logger.error(
                "recovery lookup failed for %s (coid=%s): %s; NOT resubmitting "
                "(fail closed to avoid a double-submit)",
                order.ticker, coid, e,
            )
            store.conn.execute(
                "UPDATE intended_orders SET status = 'recovery_failed', error = ? "
                "WHERE intended_order_id = ?",
                [f"recovery lookup failed: {e!r}", existing[0]],
            )
            return "failed"
        if found is not None:
            rec_id, rec_status = found
            if rec_status not in _DEAD_ORDER_STATES:
                # The order is live (or filled) at Alpaca — ADOPT it, never resubmit.
                logger.warning(
                    "recover: %s already at Alpaca (id=%s, status=%s, coid=%s); "
                    "adopting instead of resubmitting",
                    order.ticker, rec_id, rec_status, coid,
                )
                store.conn.execute(
                    "UPDATE intended_orders SET alpaca_order_id = ? WHERE intended_order_id = ?",
                    [rec_id, existing[0]],
                )
                return "submitted"
            # Prior order under this coid is dead (rejected/canceled/expired). Do
            # NOT resubmit today: the coid is spent and a fresh non-deterministic
            # id would break crash-idempotency. Record it; the next scheduled
            # decide (new asof → new coid) re-evaluates.
            logger.warning(
                "recover: %s prior order terminal at broker (status=%s, coid=%s); "
                "not resubmitting today",
                order.ticker, rec_status, coid,
            )
            store.conn.execute(
                "UPDATE intended_orders SET status = 'submission_failed', error = ? "
                "WHERE intended_order_id = ?",
                [f"prior order terminal at broker: {rec_status}", existing[0]],
            )
            return "failed"
        # Broker confirms no such order → safe to replace the row and (re-)submit
        # with the SAME deterministic coid.
        store.conn.execute(
            "DELETE FROM intended_orders WHERE asof_date = ? AND ticker = ? AND source = 'decide'",
            [asof, order.ticker],
        )

    intended_id = str(uuid.uuid4())
    target_weight = _weight_for_ticker(order.ticker, decisions)
    store.conn.execute(
        """
        INSERT INTO intended_orders
        (intended_order_id, asof_date, ticker, side, target_shares,
         target_weight, last_price, source, status, run_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, 'decide', 'submitted', ?)
    """,
        [
            intended_id,
            asof,
            order.ticker,
            order.side,
            order.shares,
            target_weight,
            order.last_price,
            run_id,
        ],
    )

    try:
        if order.side == "BUY":
            # Always a DAY MARKET order, NOT OPG. Opening-auction-only (OPG)
            # orders barely fill in Alpaca's PAPER engine — the 2026-06-08 batch
            # EXPIRED with ~0 fills, so the bot sold but couldn't buy (stuck near
            # 82% cash). A DAY market order queues to the next session and fills
            # in continuous trading near the open. (OPG could return for LIVE,
            # where the opening auction actually executes — gate on paper-vs-live.)
            aid = alpaca.submit_day_market_buy(
                order.ticker, order.shares, client_order_id=coid
            )
        else:
            aid = alpaca.submit_day_sell(
                order.ticker, order.shares, client_order_id=coid
            )
        store.conn.execute(
            "UPDATE intended_orders SET alpaca_order_id = ? WHERE intended_order_id = ?",
            [aid, intended_id],
        )
        return "submitted"
    except Exception as e:
        logger.exception("submit failed for %s: %s", order.ticker, e)
        store.conn.execute(
            "UPDATE intended_orders SET status = 'submission_failed', error = ? "
            "WHERE intended_order_id = ?",
            [str(e), intended_id],
        )
        return "failed"


def _weight_for_ticker(ticker: str, decisions) -> float | None:
    for d in decisions:
        if d.ticker == ticker:
            return float(d.target_weight)
    return None


def _latest_buy_dates(*, store, tickers: set[str]) -> dict[str, date]:
    """Return ticker → most-recent BUY fill date from paper_fills, in ET.

    `paper_fills.filled_at` is a naïve UTC TIMESTAMP. A naïve `CAST AS DATE`
    would mis-attribute a late-day ET fill (e.g. 23:30 ET / 03:30 UTC next
    day) to the wrong trading day. Convert UTC → America/New_York before
    truncating, so the date matches the trading session the operator means.

    Tickers without any BUY fill are absent from the result (caller treats
    that as "no entry date → no min-hold protection"). Used by translate()
    to enforce rails.min_hold_days.
    """
    if not tickers:
        return {}
    # DuckDB: `AT TIME ZONE 'UTC'` interprets a naïve timestamp as UTC,
    # then `AT TIME ZONE 'America/New_York'` rotates to ET; cast to DATE
    # gives the ET calendar day.
    rows = store.conn.execute(
        """
        SELECT ticker,
               MAX(
                 CAST(
                   (filled_at AT TIME ZONE 'UTC') AT TIME ZONE 'America/New_York'
                   AS DATE
                 )
               )
        FROM paper_fills
        WHERE side = 'BUY' AND ticker = ANY(?) AND filled_shares > 0
        GROUP BY ticker
        """,
        [list(tickers)],
    ).fetchall()
    return {r[0]: r[1] for r in rows if r[1] is not None}


def _load_prices(store, universe: list[str], start: date, end: date) -> pd.DataFrame:
    """Load prices via the existing Store connection (avoids DuckDB config-conflict)."""
    df = store.conn.execute(
        """
        SELECT ticker, date, open, high, low, close, adj_close, volume
        FROM (
            SELECT *,
                   ROW_NUMBER() OVER (
                       PARTITION BY ticker, date
                       ORDER BY CASE source WHEN 'yfinance' THEN 0
                                             WHEN 'alpaca' THEN 1
                                             ELSE 2 END
                   ) AS rn
            FROM prices
            WHERE ticker = ANY($tickers)
              AND date BETWEEN $start_date AND $end_date
              AND adj_close IS NOT NULL
        ) t
        WHERE rn = 1
        ORDER BY ticker, date
    """,
        {"tickers": list(universe), "start_date": start, "end_date": end},
    ).df()
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def _load_upcoming_earnings_from_store(
    *,
    store,
    start: date,
    end: date,
) -> dict[str, list[date]]:
    """Same as sma.risk.earnings_blackout.load_upcoming_earnings but via existing Store."""
    rows = store.conn.execute(
        "SELECT ticker, report_date FROM earnings "
        "WHERE report_date BETWEEN ? AND ? ORDER BY ticker, report_date",
        [start, end],
    ).fetchall()
    out: dict[str, list[date]] = {}
    for ticker, report_date in rows:
        out.setdefault(ticker, []).append(report_date)
    return out


def _last_prices_per_ticker(
    prices: pd.DataFrame,
    universe: list[str],
) -> dict[str, tuple[float, date]]:
    """Most-recent (price, price_date) per ticker."""
    out: dict[str, tuple[float, date]] = {}
    if prices.empty:
        return out
    for ticker in universe:
        df = prices[prices["ticker"] == ticker]
        if df.empty:
            continue
        latest = df.sort_values("date").iloc[-1]
        out[ticker] = (float(latest["adj_close"]), latest["date"])
    return out


def _to_dollars(positions: dict, last_prices: dict) -> dict[str, float]:
    """Convert {ticker: {shares, cost_basis}} → {ticker: dollars-current-value}."""
    out: dict[str, float] = {}
    for ticker, info in positions.items():
        if ticker in last_prices:
            price = last_prices[ticker][0]
            out[ticker] = price * info["shares"]
        else:
            # Fallback to cost_basis if no current price (e.g., halted ticker).
            out[ticker] = info["cost_basis"] * info["shares"]
    return out


def _sector_exposure(
    positions: dict,
    last_prices: dict,
    equity: float,
    sector_for: Callable[[str], str],
) -> dict[str, float]:
    """Compute sector → fraction-of-equity from current positions."""
    if equity <= 0:
        return {}
    dollars = _to_dollars(positions, last_prices)
    out: dict[str, float] = {}
    for ticker, dollar_value in dollars.items():
        sector = sector_for(ticker)
        out[sector] = out.get(sector, 0.0) + (dollar_value / equity)
    return out
