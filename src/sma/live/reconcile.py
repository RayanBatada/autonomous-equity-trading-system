"""The 16:30 ET end-of-day reconcile job.

`asof` here is the *decide* date — the day decide submitted OPG orders to
Alpaca. Reconcile is invoked the following session: it queries Alpaca for
the orders submitted that day (which fill at the next session's 09:30 ET
open auction), persists fills to paper_fills under the same decide date,
snapshots account state to account_snapshots, then runs three classes of
drift detection:

  1. No-fill drift (intended order → no matching fill):
     - Single missed BUY (DAY_OPG): log only (normal noise)
     - >20% of buys in batch missed: notify (systemic miss)
     - Any missed SELL (DAY): notify always (sells should fill)

  2. Partial-fill drift:
     - filled_shares < target_shares * 0.9: notify

  3. Catastrophic-loss drift:
     - equity dropped >10% from yesterday: notify (manual review BEFORE next decide)

10% threshold rationale: worst single-day drawdown observed in val backtest
was ~2.4%; 10% is ≈4× worst-observed, won't trip on normal market noise.
Two-tier scheme: 10% triggers a notify; 30% (= equity < 0.70 × yesterday)
blocks the next decide via preflight step 5.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from sma.live.alpaca_client import AlpacaClient
from sma.live.orders import client_order_id

logger = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")
PARTIAL_FILL_THRESHOLD = 0.90       # filled < 90% of target → notify
BUY_MISS_BATCH_THRESHOLD = 0.20     # >20% of buys missed → notify
CATASTROPHIC_LOSS_THRESHOLD = 0.10  # equity dropped >10% from yesterday


@dataclass
class DriftAlert:
    kind: str         # 'buy_miss_systemic' | 'sell_miss' | 'partial_fill' |
                      # 'catastrophic_loss' | 'snapshot_failed' |
                      # 'ledger_position_drift'
    detail: str


@dataclass
class ReconcileResult:
    fills_recorded: int
    snapshot_written: bool
    alerts: list[DriftAlert]
    # True when `now >= _next_session_open(asof)` — i.e. the asof batch has had
    # its chance to fill at an open auction. Callers should defer the
    # reconcile-sentinel write until this is True; otherwise a Monday holiday
    # reconcile would phantom-sentinel Friday's still-unfilled OPG batch, and
    # the next reconcile would skip Friday forever. False also when the
    # calendar lookup itself failed (we can't tell — caller decides).
    order_drift_open: bool = False
    # Distinguishes "calendar said open hasn't happened" from "calendar
    # failed". Callers must DEFER batch completion on calendar failure (the
    # daily 16:30 retry is bounded; the .ran liveness sentinel keeps the
    # watchdog quiet) — completing blind can strand unfilled orders' fills.
    calendar_lookup_ok: bool = False
    # Orders whose broker fetch/parse failed this run (503s, malformed
    # payloads). >0 must also DEFER batch completion so tomorrow's run
    # re-records the missing fills (Codex module review 2026-06-11 HIGH).
    fetch_failures: int = 0


def reconcile(
    *,
    asof: date,
    store,                # sma.ingest.store.Store (already connected)
    alpaca: AlpacaClient,
    notify_fn=None,       # callable(message: str) -> None; injectable for tests
    now: datetime | None = None,  # injectable wall clock for tests
    snapshot_date: date | None = None,  # date to tag account_snapshot under; defaults to now.date()
) -> ReconcileResult:
    """Match Alpaca fills against intended_orders for the given decide date,
    persist fills + account snapshot, then run drift detection.

    `asof` is the decide date being reconciled (used for paper_fills and
    drift queries). `snapshot_date` is the date the account_snapshot row is
    keyed under — by default today (when reconcile actually runs), NOT the
    asof date. Keeping these separate prevents reconcile-on-day-N+1 from
    overwriting day-N's account_snapshot with day-N+1's live equity.
    """
    if notify_fn is None:
        notify_fn = _default_notify
    if now is None:
        now = datetime.now(ET)
    if snapshot_date is None:
        snapshot_date = now.astimezone(ET).date()

    fills_recorded, fetch_failures = _record_fills(asof=asof, store=store, alpaca=alpaca)
    snapshot_written, book_positions = _write_account_snapshot(
        snapshot_date=snapshot_date, store=store, alpaca=alpaca,
    )

    alerts: list[DriftAlert] = []
    # Order-level drift (no-fill, partial-fill) is only meaningful AFTER the
    # next session's open auction has had a chance to run. Before that, OPG
    # orders are still queued and flagging them as missed is a false positive.
    # If the calendar lookup fails (e.g. Alpaca unreachable, empty calendar)
    # we conservatively skip order-drift rather than abort reconcile entirely.
    # Catastrophic-loss is equity-based and always runs.
    order_drift_open = False
    calendar_lookup_ok = False
    try:
        next_open = _next_session_open(asof=asof, alpaca=alpaca)
        # Sanity bound mirrors preflight's 14-day cap. If Alpaca's calendar is
        # degraded and persistently returns a far-future date, treat as a
        # lookup failure so the caller's fallback-to-write-sentinel kicks in
        # and we don't defer forever (codex MED finding). 14 days covers every
        # historical US closure (9/11 = 6 days, Hurricane Sandy = ~3) with
        # margin.
        if next_open - now > timedelta(days=14):
            logger.warning(
                "next_session for asof=%s is %s (>%s days out); treating as "
                "calendar failure to avoid indefinite sentinel deferral",
                asof, next_open, 14,
            )
        else:
            order_drift_open = now >= next_open
            calendar_lookup_ok = True
    except Exception:  # noqa: BLE001 — any calendar failure should not abort reconcile
        logger.warning(
            "next_session calendar lookup failed for asof=%s; "
            "skipping order-drift detection. Catastrophic-loss still runs.",
            asof, exc_info=True,
        )
    if order_drift_open:
        alerts.extend(_detect_no_fill_drift(asof=asof, store=store))
        alerts.extend(_detect_partial_fill_drift(asof=asof, store=store))
    if snapshot_written:
        alerts.extend(_detect_catastrophic_loss(snapshot_date=snapshot_date, store=store))
    else:
        # Catastrophic-loss detection reads the account snapshot we just failed to
        # write — running it on stale/missing data is worse than not running it.
        # ALERT so a human knows the safety check didn't run (2026-06-05 audit).
        alerts.append(DriftAlert(
            kind="snapshot_failed",
            detail=(
                f"account snapshot write FAILED for {snapshot_date}; "
                "catastrophic-loss detection did NOT run — review the account/DB "
                "before the next decide"
            ),
        ))

    # Ledger-vs-broker drift (2026-06-24 audit): the paper_fills ledger must
    # net to the live Alpaca book per name. Reuses the snapshot's position read
    # (no second broker call; the same book the snapshot saw); skipped when the
    # snapshot had no book to share.
    if book_positions is not None:
        try:
            alerts.extend(
                _detect_ledger_position_drift(store=store, book=book_positions)
            )
        except Exception:  # noqa: BLE001 — advisory data-integrity check
            logger.warning("ledger-position drift check failed; skipped", exc_info=True)

    for alert in alerts:
        notify_fn(f"[{alert.kind}] {alert.detail}")
        logger.warning("drift alert %s: %s", alert.kind, alert.detail)

    return ReconcileResult(
        fills_recorded=fills_recorded,
        fetch_failures=fetch_failures,
        snapshot_written=snapshot_written,
        alerts=alerts,
        order_drift_open=order_drift_open,
        calendar_lookup_ok=calendar_lookup_ok,
    )


def _next_session_open(*, asof: date, alpaca: AlpacaClient) -> datetime:
    """The 09:30 ET open-auction moment of the next trading session strictly
    after `asof`. Drift detection waits until this moment because earlier
    runs would flag still-queued OPG orders as missed."""
    next_session = alpaca.next_session_date(today=asof)
    return datetime.combine(next_session, time(9, 30), tzinfo=ET)


def _status_str(order_status) -> str:
    """Normalize an Alpaca order status to a lowercase string.

    Alpaca's TradingClient returns OrderStatus enum members. They are
    string-enum (str subclass) so equality with raw strings works, but
    `str(OrderStatus.FILLED)` returns the enum repr 'OrderStatus.FILLED',
    not 'filled'. Use .value for enums and fall back to str() for raw
    strings (test mocks).
    """
    value = getattr(order_status, "value", order_status)
    return str(value).lower()


def _side_str(order_side) -> str:
    """Same enum-vs-string normalization for OrderSide."""
    value = getattr(order_side, "value", order_side)
    return str(value).upper()


# Terminal broker statuses that did NOT fully fill, mapped to the intended_orders
# lifecycle status. A partial fill before a terminal state is recorded as
# partially_filled instead of the terminal reason.
_TERMINAL_NO_FULL_FILL = {
    "expired": "expired",
    "canceled": "canceled",
    "cancelled": "canceled",
    "rejected": "rejected",
    "done_for_day": "expired",
}


def _reconciled_status(broker_status: str, filled_qty: int) -> str | None:
    """Map a broker order status to an intended_orders lifecycle status, or None
    to leave it unchanged (order still in-flight: accepted/new/pending/held —
    reconcile runs EOD so this is rare)."""
    if broker_status == "filled":
        return "filled"
    if broker_status == "partially_filled":
        return "partially_filled"
    if broker_status in _TERMINAL_NO_FULL_FILL:
        return "partially_filled" if filled_qty > 0 else _TERMINAL_NO_FULL_FILL[broker_status]
    return None


def _backfill_null_order_ids(*, asof: date, store, alpaca: AlpacaClient) -> int:
    """Crash-after-accept recovery: a decide row can be status='submitted' with
    alpaca_order_id=NULL (the broker accepted the order but the process died
    before the id was written). _record_fills only looks at rows WITH an id, so
    such a fill would be lost forever. Recover the id via the deterministic
    client_order_id and backfill it so the fill is recorded. Returns count
    recovered."""
    rows = store.conn.execute(
        "SELECT intended_order_id, ticker, side FROM intended_orders "
        "WHERE asof_date = ? AND source = 'decide' AND alpaca_order_id IS NULL "
        # recovery_failed = decide's own transient-error fail-closed marker;
        # the broker may hold a live (even filled) order for it. It must stay
        # retryable here or the batch's fills are lost forever (Codex
        # post-audit review, 2026-06-09).
        "AND status IN ('submitted', 'recovery_failed')",
        [asof],
    ).fetchall()
    recovered = 0
    for intended_id, ticker, side in rows:
        coid = client_order_id(asof, ticker, side)
        try:
            found = alpaca.get_order_by_client_order_id(coid)
        except Exception as e:  # fail-closed lookups re-raise; log + skip this row
            logger.warning(
                "reconcile: client_order_id recovery failed for %s: %s", coid, e
            )
            continue
        if found is None:
            continue  # broker confirms no such order — nothing to recover
        order_id, _broker_status = found
        store.conn.execute(
            "UPDATE intended_orders SET alpaca_order_id = ? WHERE intended_order_id = ?",
            [order_id, intended_id],
        )
        logger.info(
            "reconcile: recovered alpaca_order_id %s for %s via client_order_id",
            order_id, coid,
        )
        recovered += 1
    return recovered


def _record_fills(*, asof: date, store, alpaca: AlpacaClient) -> tuple[int, int]:
    """Record fills for every decide order on `asof`, matched by the order's
    alpaca_order_id via its intended_orders row — NOT a submission-date window.

    OPG-queued / catch-up / weekend orders submit on a DIFFERENT calendar day
    than their decide asof, so `get_orders_for_date(asof)` returns the wrong
    orders and misses these (confirmed live 2026-06-08: the 6/05 fills were
    attributed to 6/08 and never recorded). Driving from intended_orders also
    means manual/unrelated broker activity is never considered (no contamination).
    """
    # First recover any crash-after-accept rows (status='submitted', id=NULL) so
    # their fills aren't lost; then match every placed order by id.
    _backfill_null_order_ids(asof=asof, store=store, alpaca=alpaca)
    rows = store.conn.execute(
        "SELECT intended_order_id, alpaca_order_id FROM intended_orders "
        "WHERE asof_date = ? AND source = 'decide' AND alpaca_order_id IS NOT NULL",
        [asof],
    ).fetchall()
    rid = store.allocate_run_id()
    recorded = 0
    fetch_failures = 0
    for intended_id, alpaca_order_id in rows:
        # Fetch + parse all broker-derived values inside the try: a 404,
        # transient error, OR a malformed payload (filled_qty=None/"NaN", bad
        # price) must skip THIS order with a logged warning, not abort the whole
        # reconcile and lose every other fill for the batch (Codex review). The
        # paper_fills upsert stays OUTSIDE the try — a failure there is our own
        # schema bug and must fail loud.
        try:
            order = alpaca.get_order_by_id(alpaca_order_id)
            status = _status_str(order.status)
            filled_qty = int(float(order.filled_qty or 0))
            fill_price = float(order.filled_avg_price or 0)
            commission = float(getattr(order, "commission", 0) or 0)
            fees = float(getattr(order, "fees", 0) or 0)
            symbol, side = order.symbol, _side_str(order.side)
            order_id, submitted_at, filled_at = order.id, order.submitted_at, order.filled_at
        except Exception as e:
            logger.warning(
                "reconcile: skipping order %s for intended_order %s (fetch/parse): %s",
                alpaca_order_id, intended_id, e,
            )
            fetch_failures += 1
            continue
        # Update the intended order's lifecycle status from the broker (filled/
        # partially_filled/expired/canceled/rejected) so it no longer reads
        # 'submitted' forever. Drift detection keys off alpaca_order_id +
        # paper_fills, NOT this column, so a terminal status here can't silence
        # a missed-buy alert.
        new_status = _reconciled_status(status, filled_qty)
        if new_status is not None:
            store.conn.execute(
                "UPDATE intended_orders SET status = ? WHERE intended_order_id = ?",
                [new_status, intended_id],
            )
        # Record any order with a REAL fill, even if its terminal status is
        # canceled/expired (a partial fill before cancel still moved the book).
        if status not in ("filled", "partially_filled") and filled_qty <= 0:
            continue
        store.conn.execute("""
            INSERT INTO paper_fills
            (alpaca_order_id, intended_order_id, asof_date, ticker, side,
             filled_shares, fill_price, commission, fees, status,
             submitted_at, filled_at, run_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (alpaca_order_id) DO UPDATE SET
                filled_shares = EXCLUDED.filled_shares,
                fill_price = EXCLUDED.fill_price,
                commission = EXCLUDED.commission,
                fees = EXCLUDED.fees,
                status = EXCLUDED.status,
                filled_at = EXCLUDED.filled_at,
                run_id = EXCLUDED.run_id
        """, [
            order_id, intended_id, asof, symbol, side,
            filled_qty, fill_price, commission, fees, status,
            submitted_at, filled_at, rid,
        ])
        recorded += 1
    return recorded, fetch_failures


def _write_account_snapshot(
    *, snapshot_date: date, store, alpaca: AlpacaClient
) -> tuple[bool, dict | None]:
    """Snapshot today's account state under `snapshot_date` (the date the
    snapshot was taken, NOT the reconcile asof). Idempotent on asof_date PK.

    The schema column is named `asof_date` for historical reasons, but the
    semantic is "date of the snapshot itself", not "decide date being
    reconciled". Reconcile-on-day-N+1 writes snapshot under N+1, leaving
    day-N's row untouched.
    """
    account = alpaca.get_account()
    positions = alpaca.get_positions()
    rid = store.allocate_run_id()
    try:
        store.conn.execute("""
            INSERT INTO account_snapshots
            (asof_date, equity, cash, buying_power, long_market_value,
             position_count, total_unrealized_pnl, run_id)
            VALUES (?, ?, ?, ?, ?, ?, NULL, ?)
            ON CONFLICT (asof_date) DO UPDATE SET
                equity = EXCLUDED.equity,
                cash = EXCLUDED.cash,
                buying_power = EXCLUDED.buying_power,
                long_market_value = EXCLUDED.long_market_value,
                position_count = EXCLUDED.position_count
        """, [
            snapshot_date, account["equity"], account["cash"], account["buying_power"],
            account["long_market_value"], len(positions), rid,
        ])
        return True, positions
    except Exception:
        logger.exception("failed to write account_snapshots")
        return False, positions


def _detect_no_fill_drift(*, asof: date, store) -> list[DriftAlert]:
    """Per spec §7 Layer 3: classify no-fill drift by side."""
    rows = store.conn.execute("""
        SELECT i.ticker, i.side
        FROM intended_orders i
        LEFT JOIN paper_fills f ON i.alpaca_order_id = f.alpaca_order_id
        WHERE i.asof_date = ?
          AND i.source = 'decide'
          AND i.alpaca_order_id IS NOT NULL
          AND f.alpaca_order_id IS NULL
    """, [asof]).fetchall()

    if not rows:
        return []

    missed_buys = [r[0] for r in rows if r[1] == "BUY"]
    missed_sells = [r[0] for r in rows if r[1] == "SELL"]

    total_buys = store.conn.execute(
        "SELECT COUNT(*) FROM intended_orders "
        "WHERE asof_date = ? AND side = 'BUY' AND source = 'decide' "
        "AND alpaca_order_id IS NOT NULL",
        [asof],
    ).fetchone()[0] or 0

    alerts: list[DriftAlert] = []

    if total_buys > 0:
        miss_pct = len(missed_buys) / total_buys
        if miss_pct > BUY_MISS_BATCH_THRESHOLD:
            alerts.append(DriftAlert(
                kind="buy_miss_systemic",
                detail=(f"{len(missed_buys)}/{total_buys} buys "
                        f"({miss_pct:.0%}) failed to cross at open: "
                        f"{', '.join(missed_buys)}"),
            ))

    if missed_sells:
        alerts.append(DriftAlert(
            kind="sell_miss",
            detail=(f"{len(missed_sells)} DAY sells did not fill "
                    f"(anomalous): {', '.join(missed_sells)}"),
        ))

    return alerts


def _detect_partial_fill_drift(*, asof: date, store) -> list[DriftAlert]:
    rows = store.conn.execute("""
        SELECT i.ticker, i.target_shares, f.filled_shares
        FROM intended_orders i
        JOIN paper_fills f ON i.alpaca_order_id = f.alpaca_order_id
        WHERE i.asof_date = ?
          AND f.filled_shares < i.target_shares * ?
    """, [asof, PARTIAL_FILL_THRESHOLD]).fetchall()

    return [
        DriftAlert(
            kind="partial_fill",
            detail=(f"{ticker}: filled {filled}/{target} "
                    f"({filled / target:.0%}) — significant partial"),
        )
        for ticker, target, filled in rows
    ]


def _detect_catastrophic_loss(*, snapshot_date: date, store) -> list[DriftAlert]:
    """Compare equity at snapshot_date (today) to the prior snapshot row."""
    rows = store.conn.execute("""
        SELECT today.equity AS today_eq, yesterday.equity AS yesterday_eq
        FROM account_snapshots today
        JOIN account_snapshots yesterday
          ON yesterday.asof_date = (
              SELECT MAX(asof_date) FROM account_snapshots WHERE asof_date < ?
          )
        WHERE today.asof_date = ?
    """, [snapshot_date, snapshot_date]).fetchone()

    if rows is None:
        return []
    today_eq, yesterday_eq = float(rows[0]), float(rows[1])
    if yesterday_eq <= 0:
        return []
    drop = (yesterday_eq - today_eq) / yesterday_eq
    if drop > CATASTROPHIC_LOSS_THRESHOLD:
        return [DriftAlert(
            kind="catastrophic_loss",
            detail=(f"equity dropped {drop:.1%} (yesterday ${yesterday_eq:,.0f}, "
                    f"today ${today_eq:,.0f}); manual review BEFORE next decide"),
        )]
    return []


def _detect_ledger_position_drift(*, store, book: dict) -> list[DriftAlert]:
    """Assert the paper_fills ledger nets to the live Alpaca book per name.

    `book` is the live position map {symbol: {"shares": int, ...}} already
    fetched for the account snapshot — reused here so there is no second broker
    call and the comparison is against the exact book the snapshot saw.

    Ledger net = SUM(BUY filled_shares) - SUM(SELL filled_shares) per ticker;
    in steady state it must equal the live broker share count for every name.
    A persistent mismatch means fills were mis-recorded (the 2026-06-04/05
    submission-window misattribution left COIN/CRWD/HOOD adrift): decisions and
    equity read the live book directly, so no trade is wrong, but every
    per-name P&L / cost-basis / hit-rate computed off paper_fills is unreliable
    until the ledger is backfilled. A long-only book can never net negative, so
    a negative ledger is an unambiguous recording bug.

    Returns a single alert listing all mismatches (or none). A transient
    same-day timing gap (a fill not yet recorded vs already in the book)
    self-resolves on the next reconcile, so this is advisory (notify, not block).

    Coverage caveat (Codex 2026-06-24): only fills written to paper_fills are
    counted. Decide fills are recorded by _record_fills; stop-loss sells are NOT
    (moot while stop_loss_pct=0). If stop-loss is re-enabled, record its fills
    too, otherwise this will — correctly — flag the resulting ledger gap.
    """
    ledger = {
        ticker: int(net or 0)
        for ticker, net in store.conn.execute(
            "SELECT ticker, "
            "SUM(CASE WHEN UPPER(side) = 'BUY' THEN filled_shares "
            "ELSE -filled_shares END) "
            "FROM paper_fills GROUP BY ticker"
        ).fetchall()
    }
    book_shares = {sym: int(pos["shares"]) for sym, pos in book.items()}

    mismatches = [
        (ticker, ledger.get(ticker, 0), book_shares.get(ticker, 0))
        for ticker in sorted(set(ledger) | set(book_shares))
        if ledger.get(ticker, 0) != book_shares.get(ticker, 0)
    ]
    if not mismatches:
        return []

    detail = "; ".join(
        f"{ticker}: ledger {led} vs book {bk} (Δ{led - bk:+d})"
        for ticker, led, bk in mismatches
    )
    return [DriftAlert(
        kind="ledger_position_drift",
        detail=(
            f"paper_fills ledger != live Alpaca book for {len(mismatches)} "
            f"name(s): {detail}. Per-name P&L is unreliable until the ledger "
            f"is backfilled to match the broker."
        ),
    )]


def _default_notify(message: str) -> None:
    """Default notify writes to logger.warning. Production callers should pass
    a real notify_fn that hits sma.ingest.notify.send_alert or similar.
    """
    logger.warning("DRIFT ALERT: %s", message)
