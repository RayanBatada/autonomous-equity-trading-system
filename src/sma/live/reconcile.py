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
from sma.live.quantity import as_qty, fmt_qty, qty_eq
from sma.live.trade_push import TRADE_PUSH_SENTINEL_LABEL
from sma.sentinels import read_sentinel

logger = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")
PARTIAL_FILL_THRESHOLD = 0.90       # filled < 90% of target → notify
BUY_MISS_BATCH_THRESHOLD = 0.20     # >20% of buys missed → notify
CATASTROPHIC_LOSS_THRESHOLD = 0.10  # equity dropped >10% from yesterday
# Relative tolerance for the trade-push verification: |realized_pct -
# pushed_pct| / |pushed_pct|. See _detect_trade_push_drift.
TRADE_PUSH_PCT_TOLERANCE = 0.20

# Order sources whose fills reconcile records into paper_fills. Intraday
# session orders (sma.live.session) use source 'session-<name>' and
# 'session-<name>-retry'; none exist unless a session is enabled, so for the
# open path this is exactly the historical ('decide', 'stop-loss') set.
RECORDED_SOURCES_SQL = "(source IN ('decide', 'stop-loss') OR source LIKE 'session-%')"


@dataclass(frozen=True)
class DriftThresholds:
    """The three reconcile drift-alert thresholds, defaulting to the historical
    module constants. reconcile() accepts one so config.yaml's live.drift.* keys
    actually drive the detectors — before, they were read into config and then
    ignored, a safety rail that looked tunable but wasn't (review 2026-07-04)."""

    buy_miss: float = BUY_MISS_BATCH_THRESHOLD
    partial_fill: float = PARTIAL_FILL_THRESHOLD
    catastrophic_loss: float = CATASTROPHIC_LOSS_THRESHOLD
    trade_push_pct_tolerance: float = TRADE_PUSH_PCT_TOLERANCE

    @classmethod
    def from_config(cls, drift_cfg) -> DriftThresholds:
        """Build from a config LiveDrift (config.yaml's live.drift.* block)."""
        return cls(
            buy_miss=float(drift_cfg.buy_miss_alert_threshold_pct),
            partial_fill=float(drift_cfg.partial_fill_alert_threshold_pct),
            catastrophic_loss=float(drift_cfg.catastrophic_loss_alert_pct),
            # getattr fallback: a minimal test-stub LiveDrift (or a config.yaml
            # written before 2026-09-04) may not carry this field yet.
            trade_push_pct_tolerance=float(
                getattr(drift_cfg, "trade_push_pct_tolerance_pct", TRADE_PUSH_PCT_TOLERANCE)
            ),
        )


@dataclass
class DriftAlert:
    kind: str         # 'buy_miss_systemic' | 'sell_miss' | 'partial_fill' |
                      # 'catastrophic_loss' | 'snapshot_failed' |
                      # 'ledger_position_drift' | 'trade_push_mismatch'
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
    thresholds: DriftThresholds | None = None,  # drift-alert cutoffs; defaults to module constants
    check_ledger_drift: bool = True,  # False lets a multi-batch drain run it ONCE post-drain
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
    if thresholds is None:
        thresholds = DriftThresholds()

    fills_recorded, fetch_failures = _record_fills(asof=asof, store=store, alpaca=alpaca)
    # Do not write a same-day snapshot row before snapshot_date's session has
    # actually closed (or on a non-session day) -- see _pre_close_skip_reason.
    # This is a deliberate skip, not a failure: gap-fill / the official-close
    # healer create/repair the row once the real close is available, so it
    # must NOT trip the "snapshot write FAILED" alert below.
    snapshot_skip_reason = _pre_close_skip_reason(
        snapshot_date=snapshot_date, now=now, alpaca=alpaca,
    )
    if snapshot_skip_reason is not None:
        logger.info(
            "reconcile: skipping account_snapshots write for %s: %s",
            snapshot_date, snapshot_skip_reason,
        )
        snapshot_written, book_positions = False, None
    else:
        snapshot_written, book_positions = _write_account_snapshot(
            snapshot_date=snapshot_date, store=store, alpaca=alpaca, now=now,
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
        alerts.extend(_detect_no_fill_drift(
            asof=asof, store=store, threshold=thresholds.buy_miss))
        alerts.extend(_detect_partial_fill_drift(
            asof=asof, store=store, threshold=thresholds.partial_fill))
        # `asof` here IS the correct push to verify -- do NOT "fix" this to
        # asof - 1 session. `asof` is the DECIDE date (see module docstring),
        # not today. record_trade_push(asof=asof, ...) and this same-run's
        # _record_fills(asof=asof) above are written by the SAME decide_once
        # call and the SAME reconcile call respectively, so push(asof) and
        # paper_fills(asof_date=asof) always describe the identical batch --
        # by construction, never off by a session. Investigated 2026-09-11
        # after a bug report claimed this needed asof-1: reproduced against
        # prod data (push 2026-09-04, push 2026-09-10) and the existing
        # test_reconcile_fires_trade_push_mismatch_after_open (reconcile
        # called with asof=decide-day, `now` past the FOLLOWING session's
        # open) -- both confirm same-asof matching is correct. Shifting to
        # asof-1 would have made both real pushes silently skip (no push
        # exists for the session before either).
        alerts.extend(_detect_trade_push_drift(
            asof=asof, store=store, threshold=thresholds.trade_push_pct_tolerance))
    if snapshot_written:
        alerts.extend(_detect_catastrophic_loss(
            snapshot_date=snapshot_date, store=store,
            threshold=thresholds.catastrophic_loss))
    elif snapshot_skip_reason is None:
        # Catastrophic-loss detection reads the account snapshot we just failed to
        # write — running it on stale/missing data is worse than not running it.
        # ALERT so a human knows the safety check didn't run (2026-06-05 audit).
        # Only for a GENUINE failure: an intentional pre-close skip (see above)
        # is expected daily behavior, not an error, and must not page.
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
    # snapshot had no book to share. `check_ledger_drift=False` lets a multi-batch
    # drain defer this to a SINGLE post-drain check — a fill recorded in a later
    # batch must net an earlier batch's apparent gap, else it pages mid-drain
    # (the same-day stop-exit / backlog-drain false page; review 2026-07-04).
    if book_positions is not None and check_ledger_drift:
        try:
            alerts.extend(
                _detect_ledger_position_drift(store=store, book=book_positions)
            )
        except Exception:  # noqa: BLE001 — advisory data-integrity check
            logger.warning("ledger-position drift check failed; skipped", exc_info=True)

    for alert in alerts:
        notify_fn(f"[{alert.kind}] {alert.detail}")
        logger.warning("drift alert %s: %s", alert.kind, alert.detail)

    # Measurement only (2026-09-26): the fill day's open and close next to
    # every fill. Idempotent and advisory; it never affects the batch outcome.
    try:
        record_fill_counterfactuals(store=store)
    except Exception:  # noqa: BLE001 — a measurement must never fail reconcile
        logger.warning("fill counterfactuals skipped", exc_info=True)

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


def _reconciled_status(broker_status: str, filled_qty: float) -> str | None:
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
    client_order_id and backfill it so the fill is recorded. Returns
    (recovered, lookup_failures) — lookup_failures MUST flow into the caller's
    fetch_failures so the completion gate DEFERS the batch: a transient 5xx on
    the coid lookup used to be swallowed, letting the batch complete with a
    still-NULL id whose live fill was then lost forever (review 2026-07-02)."""
    rows = store.conn.execute(
        "SELECT intended_order_id, ticker, side, source FROM intended_orders "
        # 'stop-loss' too: a price-exit sell can also crash after the broker
        # accepts it; its fill must be recovered + recorded like a decide order,
        # or the ledger never nets the exit (review 2026-07-04). Session orders
        # likewise (their coid embeds the source, like stop-loss).
        f"WHERE asof_date = ? AND {RECORDED_SOURCES_SQL} "
        "AND alpaca_order_id IS NULL "
        # recovery_failed = decide's own transient-error fail-closed marker;
        # the broker may hold a live (even filled) order for it. It must stay
        # retryable here or the batch's fills are lost forever (Codex
        # post-audit review, 2026-06-09).
        # submission_failed belongs here too: a submit that TIMED OUT on the
        # response can still have been accepted by the broker (a live order
        # that fills at the open, tracked by nobody). The coid lookup below is
        # authoritative either way — 404 means genuinely never accepted, found
        # means adopt the id (review 2026-07-01, MEDIUM).
        "AND status IN ('submitted', 'recovery_failed', 'submission_failed')",
        [asof],
    ).fetchall()
    recovered = 0
    lookup_failures = 0
    for intended_id, ticker, side, src in rows:
        # The stop-loss coid embeds its source (orders.py); decide's does not.
        coid = client_order_id(asof, ticker, side, source=None if src == "decide" else src)
        try:
            found = alpaca.get_order_by_client_order_id(coid)
        except Exception as e:  # transient failure: defer the batch, retry tomorrow
            lookup_failures += 1
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
    return recovered, lookup_failures


def _record_fills(*, asof: date, store, alpaca: AlpacaClient) -> tuple[int, int]:
    """Record fills for every decide order on `asof`, matched by the order's
    alpaca_order_id via its intended_orders row — NOT a submission-date window.

    OPG-queued / catch-up / weekend orders submit on a DIFFERENT calendar day
    than their decide asof, so `get_orders_for_date(asof)` returns the wrong
    orders and misses these (confirmed live 2026-06-08: the 6/05 fills were
    attributed to 6/08 and never recorded). Driving from intended_orders also
    means manual/unrelated broker activity is never considered (no contamination).

    Timestamp convention (`submitted_at`/`filled_at`, both plain TIMESTAMP
    columns): `order.submitted_at`/`order.filled_at` from alpaca-py are
    tz-aware UTC, but duckdb's Python client converts a tz-aware datetime to
    THIS HOST'S LOCAL TIMEZONE before storing it into a naive TIMESTAMP
    column -- and this host runs on America/New_York. So both columns land
    as naive ET WALL CLOCK, not naive UTC (verified 2026-09-27 against the
    Alpaca orders endpoint: a fill stored here as 09:32:29 is 13:32:29Z at
    the broker). Readers (`sma.live.decide._latest_buy_dates` /
    `_current_holding_entry_dates`) rely on this: `CAST(filled_at AS DATE)`
    with NO timezone conversion already gives the ET trading day.
    """
    # First recover any crash-after-accept rows (status='submitted', id=NULL) so
    # their fills aren't lost; then match every placed order by id. Backfill
    # lookup failures count as fetch failures: the batch must DEFER, not
    # complete with an unresolved possibly-live order (review 2026-07-02).
    _, backfill_failures = _backfill_null_order_ids(asof=asof, store=store, alpaca=alpaca)
    rows = store.conn.execute(
        "SELECT intended_order_id, alpaca_order_id, status, session, order_type "
        "FROM intended_orders "
        # 'stop-loss' too: a price-exit sell's fill must reach paper_fills so the
        # ledger nets the exit and the trailing-stop peak window resets (#7/#8,
        # review 2026-07-04). The rest of this function is source-agnostic.
        f"WHERE asof_date = ? AND {RECORDED_SOURCES_SQL} "
        "AND alpaca_order_id IS NOT NULL",
        [asof],
    ).fetchall()
    rid = store.allocate_run_id()
    recorded = 0
    fetch_failures = 0
    for intended_id, alpaca_order_id, cur_status, session, order_type in rows:
        # Fetch + parse all broker-derived values inside the try: a 404,
        # transient error, OR a malformed payload (filled_qty=None/"NaN", bad
        # price) must skip THIS order with a logged warning, not abort the whole
        # reconcile and lose every other fill for the batch (Codex review). The
        # paper_fills upsert stays OUTSIDE the try — a failure there is our own
        # schema bug and must fail loud.
        try:
            order = alpaca.get_order_by_id(alpaca_order_id)
            status = _status_str(order.status)
            filled_qty = as_qty(order.filled_qty or 0)
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
        # Preserve the sweep's 'canceled_by_stop' marker: the broker reports the
        # order as plain 'canceled' (we canceled it), but overwriting the marker
        # would let the no-fill drift detector misread this deliberate exit as a
        # missed buy. A real partial fill is still recorded below regardless.
        new_status = _reconciled_status(status, filled_qty)
        if new_status is not None and cur_status != "canceled_by_stop":
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
             submitted_at, filled_at, run_id, session, order_type)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            submitted_at, filled_at, rid, session, order_type,
        ])
        recorded += 1
    return recorded, fetch_failures + backfill_failures


# After this ET time, get_account().equity is priced off after-hours quotes
# rather than the 16:00 close, so the snapshot must source equity from
# portfolio-history instead. 16:10 leaves the normal 16:30 reconcile on the
# portfolio-history path while giving Alpaca a few minutes past the bell.
_AFTER_HOURS_EQUITY_CUTOFF_ET = time(16, 10)


def _snapshot_equity(
    *, snapshot_date: date, account: dict, alpaca: AlpacaClient, now: datetime
) -> tuple[float, float, str]:
    """Resolve (equity, long_market_value, source) for the snapshot row.

    Before 16:10 ET the live get_account read IS the session mark, so use it.
    After 16:10 that read prices the book off after-hours quotes: on 2026-07-29
    it was off by thousands, and on 2026-08-12 a boot-catch-up reconcile at
    20:16 stored 102,516.39 against a true close of 102,343.20 (0.17% high).

    cash is not mark-dependent (no fills after the bell), so the correction is
    absorbed entirely into long_market_value, preserving the row's own
    equity == cash + long_market_value identity.
    """
    equity = account["equity"]
    lmv = account["long_market_value"]
    now_et = now.astimezone(ET)
    if now_et.date() != snapshot_date or now_et.timetz().replace(
        tzinfo=None
    ) < _AFTER_HOURS_EQUITY_CUTOFF_ET:
        return equity, lmv, "get_account"

    try:
        close = alpaca.session_close_equity(day=snapshot_date)
    except Exception:  # noqa: BLE001 — a history failure must not lose the snapshot
        logger.warning(
            "portfolio-history lookup failed for %s; storing the live "
            "get_account equity, which after 16:10 ET is an AFTER-HOURS mark",
            snapshot_date, exc_info=True,
        )
        return equity, lmv, "get_account_after_hours"

    if close is None:
        # Alpaca has not published enough of the session yet (the 16:10-18:00
        # window, and empirically often later). Keep the live read but LABEL it
        # so a stale-looking row is explainable rather than mysterious.
        logger.warning(
            "no portfolio-history close for %s yet; storing the live get_account "
            "equity %.2f, which is an AFTER-HOURS mark", snapshot_date, equity,
        )
        return equity, lmv, "get_account_after_hours"

    close_equity, source = close
    drift = abs(close_equity - equity)
    if drift > 0:
        logger.info(
            "snapshot equity for %s: using %s close %.2f instead of the live "
            "after-hours get_account read %.2f (diff %.2f, %.4f%%)",
            snapshot_date, source, close_equity, equity, drift,
            drift / close_equity * 100 if close_equity else 0.0,
        )
    return close_equity, round(close_equity - account["cash"], 2), source


# Used only when the trading-calendar lookup itself fails (Alpaca unreachable):
# we cannot ask session_window whether `snapshot_date` has closed, so fall
# back to treating the ordinary 16:00 ET close as the boundary — the same
# fail-closed posture _sweep_skip_reason's fallback band uses, rather than
# either always-skip or always-write on a calendar blip.
_CALENDAR_FAILURE_FALLBACK_CLOSE_ET = time(16, 0)


def _pre_close_skip_reason(
    *, snapshot_date: date, now: datetime, alpaca: AlpacaClient
) -> str | None:
    """Return why a same-day account_snapshots row for `snapshot_date` must NOT
    be written yet, or None when it is safe to write.

    A snapshot row is meant to record that session's CLOSE. Before the session
    has actually closed -- or on a day that is not a trading session at all --
    get_account() prices off whatever mark happens to be live, which before the
    open is literally the PRIOR session's after-hours mark. Writing that under
    asof_date=snapshot_date creates a phantom "close" for a session that has not
    happened yet.

    2026-08-20: the Mac booted at 05:23 and a launchd catch-up ran reconcile at
    ~05:36 -- before 2026-08-20's session had even OPENED -- and wrote a
    2026-08-20 row sourced from Wednesday night's after-hours mark
    ($121,930.99 vs live $121,534). backfill_official_closes self-heals it at
    the next post-close reconcile, but a machine that stays dark until 16:30
    leaves the phantom standing as "today's close" all day, including on the
    dashboard.

    backfill_missing_snapshots (gap-fill) and backfill_official_closes (the
    official-close healer) already create/repair `snapshot_date`'s row once the
    real close is available, so skipping here is safe: the row for
    `snapshot_date` gets written correctly by the next post-close reconcile or
    by gap-fill.

    Half-days close at 13:00, not 16:00 -- `session_window` supplies the real
    close so this never hardcodes the wrong hour on those days.
    """
    try:
        window = alpaca.session_window(day=snapshot_date)
    except Exception:  # noqa: BLE001 — a calendar blip must not lose the snapshot outright
        logger.warning(
            "session_window lookup failed for %s; falling back to a fixed "
            "16:00 ET close to decide whether it is safe to write today's "
            "account_snapshots row", snapshot_date, exc_info=True,
        )
        close = datetime.combine(
            snapshot_date, _CALENDAR_FAILURE_FALLBACK_CLOSE_ET, tzinfo=ET
        )
    else:
        if window is None:
            return f"{snapshot_date} is not an NYSE trading session"
        _, close = window
    now_et = now.astimezone(ET)
    if now_et < close:
        return (
            f"{snapshot_date}'s session has not closed yet (close "
            f"{close:%H:%M} ET, now {now_et:%H:%M} ET)"
        )
    return None


def _write_account_snapshot(
    *, snapshot_date: date, store, alpaca: AlpacaClient, now: datetime | None = None
) -> tuple[bool, dict | None]:
    """Snapshot today's account state under `snapshot_date` (the date the
    snapshot was taken, NOT the reconcile asof). Idempotent on asof_date PK.

    The schema column is named `asof_date` for historical reasons, but the
    semantic is "date of the snapshot itself", not "decide date being
    reconciled". Reconcile-on-day-N+1 writes snapshot under N+1, leaving
    day-N's row untouched.

    Equity is sourced per _snapshot_equity: the live read before 16:10 ET, the
    portfolio-history close after it. equity_source records which.
    """
    if now is None:
        now = datetime.now(ET)
    account = alpaca.get_account()
    positions = alpaca.get_positions()
    equity, lmv, equity_source = _snapshot_equity(
        snapshot_date=snapshot_date, account=account, alpaca=alpaca, now=now,
    )
    rid = store.allocate_run_id()
    try:
        store.conn.execute("""
            INSERT INTO account_snapshots
            (asof_date, equity, cash, buying_power, long_market_value,
             position_count, total_unrealized_pnl, run_id, equity_source)
            VALUES (?, ?, ?, ?, ?, ?, NULL, ?, ?)
            ON CONFLICT (asof_date) DO UPDATE SET
                equity = EXCLUDED.equity,
                cash = EXCLUDED.cash,
                buying_power = EXCLUDED.buying_power,
                long_market_value = EXCLUDED.long_market_value,
                position_count = EXCLUDED.position_count,
                equity_source = EXCLUDED.equity_source
        """, [
            snapshot_date, equity, account["cash"], account["buying_power"],
            lmv, len(positions), rid, equity_source,
        ])
        return True, positions
    except Exception:
        logger.exception("failed to write account_snapshots")
        return False, positions


# 2026-08-13: portfolio_history_1min_close (validated ~0.01% error vs the
# after-hours mark, 2026-08-12) is a proxy, not the record of truth. Alpaca
# eventually publishes the authoritative 1D bar for a session, sometimes
# hours or days later (measured ABSENT at 21:20 ET the same evening — see
# session_close_equity). This self-heals any snapshot still sitting on the
# proxy once the official close lands, so a human doesn't have to remember to
# backfill it (7081d81's 8/12 correction was a one-time manual fix; this
# makes it automatic on every future reconcile run).
BACKFILL_LOOKBACK_TRADING_DAYS = 5


def backfill_official_closes(*, store, alpaca: AlpacaClient) -> int:
    """Self-heal: replace a NON-OFFICIAL snapshot equity with Alpaca's OFFICIAL
    daily close once it publishes, for any snapshot row in the last
    BACKFILL_LOOKBACK_TRADING_DAYS trading days. Call this at the START of
    every reconcile run, under the same writer_lock/connection reconcile
    itself uses.

    Candidate = any row not already sourced from the OFFICIAL daily bar
    ('portfolio_history_daily' when the snapshot itself got it,
    'portfolio_history_daily_close' when this backfill wrote it). The original
    filter only healed 'portfolio_history_1min_close' rows, which excluded
    exactly the WRONG ones (2026-08-16 review):

      - 'get_account_after_hours' — the label _snapshot_equity writes when the
        portfolio-history lookup failed or hadn't published, i.e. the read that
        was off by THOUSANDS on 2026-07-29. That row is the least trustworthy
        in the table and was the one row the heal refused to touch.
      - NULL — legacy rows written before equity_source existed; provenance
        unknown, so no reason to trust them over the official bar.
      - 'get_account' — a pre-16:10 ET read. An honest INTRADAY mark, but still
        not the close.

    Expressed as "not already official" rather than an allowlist of proxies so
    a future source label can't silently opt itself out of healing again.

    Idempotent: a corrected row's equity_source becomes
    'portfolio_history_daily_close', so it stops matching and this function
    never touches it again. A row whose official close still isn't published
    (or whose lookup fails) is left as-is and re-checked next reconcile run.

    cash is mark-independent (no fills after the bell), so the correction is
    absorbed entirely into long_market_value — same invariant
    _snapshot_equity keeps: equity == cash + long_market_value on the stored
    row.

    Returns the number of rows corrected (also emits one INFO log line per
    correction).
    """
    rows = store.conn.execute(
        """
        WITH recent AS (
            SELECT asof_date FROM account_snapshots
            ORDER BY asof_date DESC LIMIT ?
        )
        SELECT asof_date, equity, cash FROM account_snapshots
        WHERE (equity_source IS NULL
               OR equity_source NOT IN (
                   'portfolio_history_daily', 'portfolio_history_daily_close'
               ))
          AND asof_date IN (SELECT asof_date FROM recent)
        ORDER BY asof_date DESC
        """,
        [BACKFILL_LOOKBACK_TRADING_DAYS],
    ).fetchall()

    corrected = 0
    for snap_date, old_equity, cash in rows:
        try:
            close = alpaca.session_close_equity(day=snap_date)
        except Exception:  # noqa: BLE001 — one bad lookup must not stop the rest
            logger.warning(
                "backfill: official-close lookup failed for %s; leaving the "
                "1Min-proxy row in place", snap_date, exc_info=True,
            )
            continue
        if close is None:
            continue  # nothing published yet — normal, retried next reconcile
        close_equity, source = close
        if source != "portfolio_history_daily":
            # session_close_equity fell back to the 1Min proxy again — the
            # OFFICIAL bar still hasn't landed for this session.
            continue
        if close_equity == old_equity:
            continue  # stored value and official already agree exactly
        new_lmv = round(close_equity - cash, 2)
        store.conn.execute(
            "UPDATE account_snapshots SET equity = ?, long_market_value = ?, "
            "equity_source = 'portfolio_history_daily_close' "
            "WHERE asof_date = ?",
            [close_equity, new_lmv, snap_date],
        )
        logger.info(
            "backfill: corrected %s equity %.2f -> %.2f (official daily "
            "close replacing the 1Min proxy)", snap_date, old_equity, close_equity,
        )
        corrected += 1
    return corrected


# 2026-08-20: the Mac was dark through the entire 2026-08-19 session, so
# reconcile itself never ran and NO account_snapshots row exists for that date
# at all -- backfill_official_closes (above) only UPGRADES an existing row's
# equity source, it cannot CREATE a missing one. This bounds how many trading
# sessions back backfill_missing_snapshots looks for a missing row.
GAPFILL_LOOKBACK_SESSIONS = 5
# Calendar days searched backward to find GAPFILL_LOOKBACK_SESSIONS trading
# sessions. 14 comfortably covers the longest realistic US holiday gap --
# mirrors _next_session_open's identical 14-day sanity bound.
GAPFILL_CALENDAR_LOOKBACK_DAYS = 14


def backfill_missing_snapshots(
    *, store, alpaca: AlpacaClient, today: date | None = None
) -> int:
    """Self-heal: INSERT a missing account_snapshots row for any of the last
    GAPFILL_LOOKBACK_SESSIONS trading sessions that has none, once Alpaca's
    OFFICIAL daily close is available for that date. Call this at the START
    of every reconcile run (alongside backfill_official_closes), under the
    same writer_lock/connection reconcile itself uses.

    Unlike backfill_official_closes (which corrects an existing row's equity
    source), this CREATES a row that never existed -- the case where
    reconcile never ran at all for a session (a dead host through the whole
    day). Session dates come from the Alpaca calendar rather than existing
    table rows: a missing day by definition has no row of its own to anchor a
    "last N rows" lookback the way backfill_official_closes does.

    Only the OFFICIAL portfolio-history daily bar is accepted -- never the
    1Min proxy backfill_official_closes tolerates for corrections. An
    approximate equity for a day whose cash/positions can never be
    reconstructed is not worth the false precision; better to wait one more
    day for the real bar. cash/buying_power/long_market_value/position_count
    are stored NULL (migration 8): they are genuinely unknowable for a day
    nothing ran, not merely unrecorded.

    Idempotent: a session with a row (any equity_source) is never a candidate
    again. A session whose official close still isn't published (or whose
    lookup fails) is left missing and re-checked on the next reconcile run.

    Returns the number of rows inserted (also emits one INFO log line per
    insert).
    """
    if today is None:
        today = datetime.now(ET).date()

    try:
        sessions = alpaca.sessions_between(
            start=today - timedelta(days=GAPFILL_CALENDAR_LOOKBACK_DAYS), end=today,
        )
    except Exception:  # noqa: BLE001 — a calendar failure must not abort reconcile
        logger.warning(
            "gap-fill: calendar lookup failed; skipping this reconcile run",
            exc_info=True,
        )
        return 0
    recent_sessions = sessions[-GAPFILL_LOOKBACK_SESSIONS:]
    if not recent_sessions:
        return 0

    existing = {
        row[0]
        for row in store.conn.execute(
            "SELECT asof_date FROM account_snapshots WHERE asof_date IN "
            f"({', '.join(['?'] * len(recent_sessions))})",
            recent_sessions,
        ).fetchall()
    }
    missing = sorted(d for d in recent_sessions if d not in existing)

    inserted = 0
    for snap_date in missing:
        try:
            close = alpaca.session_close_equity(day=snap_date)
        except Exception:  # noqa: BLE001 — one bad lookup must not stop the rest
            logger.warning(
                "gap-fill: official-close lookup failed for %s; no row inserted, "
                "will retry next reconcile run", snap_date, exc_info=True,
            )
            continue
        if close is None:
            continue  # nothing published yet — normal, retried next reconcile
        close_equity, source = close
        if source != "portfolio_history_daily":
            # Only the 1Min proxy is available so far — not worth inserting a
            # gap row on an approximate mark; wait for the official bar.
            continue
        rid = store.allocate_run_id()
        store.conn.execute(
            """
            INSERT INTO account_snapshots
            (asof_date, equity, cash, buying_power, long_market_value,
             position_count, total_unrealized_pnl, run_id, equity_source)
            VALUES (?, ?, NULL, NULL, NULL, NULL, NULL, ?, 'portfolio_history_daily_close')
            ON CONFLICT (asof_date) DO NOTHING
            """,
            [snap_date, close_equity, rid],
        )
        logger.info(
            "gap-fill: inserted missing account_snapshots row for %s "
            "(equity=%.2f, source=portfolio_history_daily_close); cash/positions "
            "unknown for the missed day", snap_date, close_equity,
        )
        inserted += 1
    return inserted


# Trading sessions elapsed since a snapshot beyond which it is stale enough to
# warrant a caveat wherever equity is surfaced (dashboard, `sma.live status`).
# >1 means at least one full session's gap exists between the snapshot and
# the latest known session -- the 2026-08-19 case: the account moved +19%
# (MRNA earnings) while status/dashboard kept quoting the 2026-08-18 close.
STALE_SNAPSHOT_SESSIONS_THRESHOLD = 1


def snapshot_staleness_sessions(*, conn, snapshot_date: date) -> int:
    """Trading sessions elapsed strictly after `snapshot_date`, proxied by
    SPY's ingested price dates in the `prices` table rather than an Alpaca
    calendar call.

    Needed by read-only, no-broker-call consumers (the dashboard's Paper tab,
    `sma.live status`) that must not spend an API call just to caveat a
    number. `prices` is populated every trading evening by ingest regardless
    of whether reconcile ran, so it is a reliable trading-calendar proxy that
    is always already in the DB.

    0 means `snapshot_date` IS the latest known trading session (fresh).
    """
    row = conn.execute(
        "SELECT COUNT(DISTINCT date) FROM prices WHERE ticker = 'SPY' AND date > ?",
        [snapshot_date],
    ).fetchone()
    return int(row[0] or 0)


def snapshot_staleness_message(*, sessions: int, snapshot_date: date) -> str | None:
    """Shared caveat text for a stale account_snapshots read, or None when
    `sessions` (from snapshot_staleness_sessions) is fresh enough not to need
    one. Both the dashboard banner and `sma.live status` import this so they
    say the exact same thing about the exact same condition."""
    if sessions <= STALE_SNAPSHOT_SESSIONS_THRESHOLD:
        return None
    return (
        f"Latest snapshot is {sessions} sessions old (asof {snapshot_date}) "
        "— live account may differ."
    )


def _detect_no_fill_drift(
    *, asof: date, store, threshold: float = BUY_MISS_BATCH_THRESHOLD
) -> list[DriftAlert]:
    """Per spec §7 Layer 3: classify no-fill drift by side."""
    # A 'canceled_by_stop' decide order was pulled ON PURPOSE by the 09:25 sweep
    # (it exited the name); that is not a failed open-auction cross, so exclude it
    # from BOTH the missed set and the total denominator or it inflates
    # buy_miss_systemic and pages spuriously (#9/#10, review 2026-07-04). A generic
    # broker/manual 'canceled' is NOT excluded — that genuinely didn't execute and
    # is worth surfacing. _record_fills preserves the 'canceled_by_stop' marker.
    rows = store.conn.execute("""
        SELECT i.ticker, i.side
        FROM intended_orders i
        LEFT JOIN paper_fills f ON i.alpaca_order_id = f.alpaca_order_id
        WHERE i.asof_date = ?
          AND i.source = 'decide'
          AND i.alpaca_order_id IS NOT NULL
          AND i.status != 'canceled_by_stop'
          AND f.alpaca_order_id IS NULL
    """, [asof]).fetchall()

    if not rows:
        return []

    missed_buys = [r[0] for r in rows if r[1] == "BUY"]
    missed_sells = [r[0] for r in rows if r[1] == "SELL"]

    total_buys = store.conn.execute(
        "SELECT COUNT(*) FROM intended_orders "
        "WHERE asof_date = ? AND side = 'BUY' AND source = 'decide' "
        "AND alpaca_order_id IS NOT NULL "
        "AND status != 'canceled_by_stop'",
        [asof],
    ).fetchone()[0] or 0

    alerts: list[DriftAlert] = []

    if total_buys > 0:
        miss_pct = len(missed_buys) / total_buys
        if miss_pct > threshold:
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


def _detect_partial_fill_drift(
    *, asof: date, store, threshold: float = PARTIAL_FILL_THRESHOLD
) -> list[DriftAlert]:
    rows = store.conn.execute("""
        SELECT i.ticker, i.target_shares, f.filled_shares
        FROM intended_orders i
        JOIN paper_fills f ON i.alpaca_order_id = f.alpaca_order_id
        WHERE i.asof_date = ?
          AND f.filled_shares < i.target_shares * ?
    """, [asof, threshold]).fetchall()

    return [
        DriftAlert(
            kind="partial_fill",
            detail=(f"{ticker}: filled {fmt_qty(filled)}/{fmt_qty(target)} "
                    f"({filled / target:.0%}) — significant partial"),
        )
        for ticker, target, filled in rows
    ]


def _detect_trade_push_drift(
    *, asof: date, store, threshold: float = TRADE_PUSH_PCT_TOLERANCE,
) -> list[DriftAlert]:
    """Verify tonight's ntfy trade push against the fills that actually
    booked (2026-09-04: a decide push overstated an order's size 5x -- see
    sma.live.trade_push's module docstring -- and was caught only by a human
    eyeballing the push against Alpaca fills; ntfy's own cache expires in
    12h, so without this nothing lets a push be checked after the fact).

    Reads the com.sma.trade-push-<asof> sentinel `sma.live.trade_push.
    record_trade_push` writes at send time. Missing sentinel (a push from
    before this feature shipped, or the persist itself failed) -> silent
    skip: this is a best-effort accuracy check, not a required gate. An
    empty orders list (a "no trades tonight" push) -> nothing to verify.

    Per pushed order: match to its paper_fills row by (ticker, side) via
    intended_orders (source='decide') for this asof. Shares must match
    EXACTLY (qty_eq) -- any mismatch is worth a page regardless of dollar
    size, since it means the push claimed a different trade than what was
    actually placed. Given matching shares, the realized fill pct
    (fill notional / pushed equity) must be within `threshold` RELATIVE
    tolerance of the pushed pct -- generous enough for an ordinary overnight
    price move between the decide-time mark and the next session's fill,
    tight enough to catch a wrong-by-multiples bug like the LLY one (pushed
    5.7%, realized 1.0%: a 470% relative diff, nowhere near the 20% default).
    """
    push = read_sentinel(label=TRADE_PUSH_SENTINEL_LABEL, asof=asof)
    if push is None:
        return []
    orders = push.get("orders") or []
    if not orders:
        return []
    push_equity = push.get("equity")

    alerts: list[DriftAlert] = []
    matched = 0
    for pushed in orders:
        try:
            ticker = pushed["ticker"]
            side = pushed["side"]
            pushed_shares = pushed.get("shares")
            pushed_pct = pushed.get("order_pct_of_equity")
        except (KeyError, TypeError):
            logger.warning(
                "trade-push verify: malformed pushed order %r for asof %s; skipping",
                pushed, asof,
            )
            continue

        row = store.conn.execute(
            """
            SELECT COALESCE(SUM(f.filled_shares), 0),
                   COALESCE(SUM(f.filled_shares * f.fill_price), 0)
            FROM intended_orders i
            LEFT JOIN paper_fills f ON i.alpaca_order_id = f.alpaca_order_id
            WHERE i.asof_date = ? AND i.source = 'decide'
              AND i.ticker = ? AND i.side = ?
            """,
            [asof, ticker, side],
        ).fetchone()
        realized_shares = float(row[0]) if row else 0.0
        realized_notional = float(row[1]) if row else 0.0

        try:
            shares_match = (
                pushed_shares is not None
                and qty_eq(float(pushed_shares), realized_shares)
            )
        except (TypeError, ValueError):
            logger.warning(
                "trade-push verify: unusable pushed shares %r for %s %s "
                "(asof %s); skipping",
                pushed_shares, side, ticker, asof,
            )
            continue
        if not shares_match:
            alerts.append(DriftAlert(
                kind="trade_push_mismatch",
                detail=(
                    f"{side} {ticker}: pushed {fmt_qty(pushed_shares or 0)} shares, "
                    f"realized {fmt_qty(realized_shares)} shares (asof {asof.isoformat()})"
                ),
            ))
            continue

        if pushed_pct is None or not push_equity:
            matched += 1  # shares matched; nothing more to compare
            continue
        try:
            pushed_pct = float(pushed_pct)
        except (TypeError, ValueError):
            matched += 1  # shares matched; pct unusable, don't false-page on it
            continue

        realized_pct = realized_notional / push_equity
        if pushed_pct == 0:
            rel_diff = 0.0 if realized_pct == 0 else float("inf")
        else:
            rel_diff = abs(realized_pct - pushed_pct) / abs(pushed_pct)

        if rel_diff > threshold:
            alerts.append(DriftAlert(
                kind="trade_push_mismatch",
                detail=(
                    f"{side} {ticker}: pushed {pushed_pct:.1%} of equity, "
                    f"realized {realized_pct:.1%} ({rel_diff:.0%} relative diff, "
                    f"tolerance {threshold:.0%}; asof {asof.isoformat()})"
                ),
            ))
        else:
            matched += 1

    if matched and not alerts:
        logger.info(
            "trade-push verify: %d/%d pushed order(s) matched booked fills for %s",
            matched, len(orders), asof.isoformat(),
        )
    return alerts


def _detect_catastrophic_loss(
    *, snapshot_date: date, store, threshold: float = CATASTROPHIC_LOSS_THRESHOLD
) -> list[DriftAlert]:
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
    if drop > threshold:
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

    Coverage (Codex 2026-06-24; resolved 2026-07-04): only fills written to
    paper_fills are counted. _record_fills records BOTH decide and stop-loss fills
    (source IN ('decide','stop-loss')); _resolve_reconcile_asofs discovers a
    stop-only day (decide placed nothing); and reconcile_cmd runs THIS check ONCE
    after the whole drain (check_ledger_drift=False per batch). Together those mean
    a fired price exit's SELL is recorded the same afternoon and the ledger nets to
    the broker book, rather than paging a spurious gap mid-drain or one day late.
    """
    ledger = {
        ticker: as_qty(net or 0)
        for ticker, net in store.conn.execute(
            "SELECT ticker, "
            "SUM(CASE WHEN UPPER(side) = 'BUY' THEN filled_shares "
            "ELSE -filled_shares END) "
            "FROM paper_fills GROUP BY ticker"
        ).fetchall()
    }
    book_shares = {sym: as_qty(pos["shares"]) for sym, pos in book.items()}

    # qty_eq, not `!=`: once quantities can be fractional, comparing a SUM of
    # recorded fills against the broker's own float with bare `!=` pages every
    # single afternoon on a 1e-16 rounding residue. A real divergence is orders
    # of magnitude larger than QTY_EPS.
    mismatches = [
        (ticker, ledger.get(ticker, 0), book_shares.get(ticker, 0))
        for ticker in sorted(set(ledger) | set(book_shares))
        if not qty_eq(ledger.get(ticker, 0), book_shares.get(ticker, 0))
    ]
    if not mismatches:
        return []

    detail = "; ".join(
        f"{ticker}: ledger {fmt_qty(led)} vs book {fmt_qty(bk)} "
        f"(Δ{'+' if led >= bk else '-'}{fmt_qty(abs(led - bk))})"
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


# ---- fill quality (2026-09-26) ----------------------------------------------
#
# Measurement only. Nothing below changes an order, a fill row or a drift
# alert. Timestamps: paper_fills.filled_at is the broker's ET wall clock,
# stored naive (verified 2026-09-26 against the Alpaca orders endpoint: DB
# 09:32:29 == broker 13:32:29Z), so CAST(filled_at AS DATE) is the ET session.

# |bp| above this is a benchmark on a different share scale (a split that
# yfinance restated and the raw fill did not: CRWD 4:1 in June), not a cost.
# Kept as a backstop even though print selection (below) now picks the
# same-scale source first; see _pick_print_prices.
FILL_QUALITY_MAX_ABS_BP = 2000.0


def _signed_cost_bp(side: str, fill: float, bench: float | None) -> float | None:
    if bench is None or bench <= 0 or fill is None or fill <= 0:
        return None
    sign = 1.0 if str(side).upper() == "BUY" else -1.0
    return sign * (fill - bench) / bench * 10_000.0


# Print selection, shared by the counterfactuals table and the fill-quality
# report (2026-10-01 fix). The first version of each took yfinance first, but
# yfinance history is SPLIT-ADJUSTED while fills are on the share scale of the
# day they happened: CRWD's 4:1 split (2026-07-02) turned May/June CRWD buys
# at 650-690 into "+30,000bp" against a restated 160-175 open, and FDX's
# restated history did the same at +-2,400bp. Alpaca IEX bars are RAW (never
# restated), i.e. the same scale as the fill by construction, so they go
# first. yfinance is only a fallback, and only when its close for the fill
# date is within _CF_YF_FALLBACK_MAX_REL of the fill price (same scale);
# otherwise the prints are NULL with print_source='unavailable' rather than a
# number on the wrong scale.
_CF_YF_FALLBACK_MAX_REL = 0.05
CF_UNAVAILABLE = "unavailable"


def _pick_print_prices(fill_px, a_open, a_close, y_open, y_close):
    """(open_print, close_print, print_source) for one fill: alpaca (raw,
    same scale as the fill) first, yfinance only as a fallback and only when
    it's within _CF_YF_FALLBACK_MAX_REL of the fill price (same scale), else
    unavailable. Used by both record_fill_counterfactuals and
    fill_quality_by_session so the two never disagree on which print a fill
    is benchmarked against."""
    if a_close is not None and a_close > 0:
        return a_open, a_close, "alpaca"
    if (
        y_close is not None and y_close > 0 and fill_px
        and abs(y_close / fill_px - 1.0) <= _CF_YF_FALLBACK_MAX_REL
    ):
        return y_open, y_close, "yfinance"
    return None, None, CF_UNAVAILABLE


def record_fill_counterfactuals(*, store, recompute: bool = False) -> int:
    """Upsert fill_counterfactuals for every paper fill that has no row yet,
    or whose row still lacks a print (the fill day's prices land at the 18:30
    ingest, after the 16:30 reconcile, so day D completes at D+1's run).
    Complete rows are frozen, unless `recompute` (reconcile
    --recompute-counterfactuals) rewrites every row. Returns rows written."""
    rows = store.conn.execute("""
        SELECT f.alpaca_order_id, f.ticker, UPPER(f.side), CAST(f.filled_at AS DATE),
               f.fill_price, f.filled_shares, a.open, a.close, y.open, y.close
        FROM paper_fills f
        LEFT JOIN fill_counterfactuals c ON c.alpaca_order_id = f.alpaca_order_id
        LEFT JOIN prices a ON a.ticker = f.ticker AND a.date = CAST(f.filled_at AS DATE)
                          AND a.source = 'alpaca'
        LEFT JOIN prices y ON y.ticker = f.ticker AND y.date = CAST(f.filled_at AS DATE)
                          AND y.source = 'yfinance'
        WHERE f.filled_at IS NOT NULL AND f.filled_shares > 0 AND f.fill_price > 0
          AND (? OR c.alpaca_order_id IS NULL OR c.open_print IS NULL
               OR c.close_print IS NULL)
    """, [bool(recompute)]).fetchall()
    if not rows:
        return 0
    rid = store.allocate_run_id()
    now = datetime.utcnow()
    out = []
    for oid, ticker, side, fdate, px, qty, a_o, a_c, y_o, y_c in rows:
        o, c, src = _pick_print_prices(px, a_o, a_c, y_o, y_c)
        out.append((
            oid, ticker, side, fdate, px, qty, o, c, src,
            _signed_cost_bp(side, px, o), _signed_cost_bp(side, px, c), now, rid,
        ))
    store.conn.executemany("""
        INSERT OR REPLACE INTO fill_counterfactuals
        (alpaca_order_id, ticker, side, fill_date, fill_price, filled_shares,
         open_print, close_print, print_source, open_bp, close_bp, computed_at, run_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, out)
    return len(out)


def counterfactual_stats(conn) -> dict:
    """All-fills summary of fill_counterfactuals (signed cost bp, + = cost)."""
    n, o_mean, o_med, c_mean, c_med, o_null, c_null = conn.execute("""
        SELECT COUNT(*), AVG(open_bp), MEDIAN(open_bp), AVG(close_bp), MEDIAN(close_bp),
               COUNT(*) FILTER (WHERE open_bp IS NULL),
               COUNT(*) FILTER (WHERE close_bp IS NULL)
        FROM fill_counterfactuals
    """).fetchone()
    by_source = dict(conn.execute(
        "SELECT COALESCE(print_source, 'NULL'), COUNT(*) FROM fill_counterfactuals GROUP BY 1"
    ).fetchall())
    return {
        "n": n, "open_bp_mean": o_mean, "open_bp_median": o_med,
        "close_bp_mean": c_mean, "close_bp_median": c_med,
        "open_bp_null": o_null, "close_bp_null": c_null, "by_source": by_source,
    }


def fill_quality_by_session(
    conn, *, since: date | None = None, max_abs_bp: float = FILL_QUALITY_MAX_ABS_BP
) -> list[dict]:
    """Realized cost per fill in signed bp (+ = cost), grouped by
    (session, order_type). NULL session/order_type is the open path and is
    labelled 'open'/'opg' (historical label: decide has sent DAY market
    orders, not OPG, since 2026-06-08).

    Benchmarks: open-path fills against the fill day's open print, picked by
    the same alpaca-first/yfinance-fallback logic as the fill_counterfactuals
    table (_pick_print_prices: alpaca raw bars first, since they're on the
    same share scale as the fill; yfinance only when its close is within
    _CF_YF_FALLBACK_MAX_REL of the fill price; otherwise the fill has no
    print and is excluded rather than benchmarked against a restated,
    wrong-scale price). Intraday-session fills are benchmarked against their
    arrival price, the quote mid the session priced off
    (intended_orders.last_price) — unaffected by print source. Each group
    reports n, equal-weighted mean and median, the notional-weighted mean
    (the form the 16.6bp headline used), how many fills had no benchmark at
    all (missing arrival price) and how many were excluded, either as a
    split-scale mismatch (no usable print) or a residual outlier
    (|bp| > max_abs_bp).

    `close_nw` is the counterfactual: the same fills against the fill day's
    CLOSE print (same selection logic), notional-weighted. It answers "would
    the close have been cheaper" without changing a single order.
    """
    rows = conn.execute("""
        SELECT COALESCE(f.session, i.session, 'open') AS session,
               COALESCE(f.order_type, i.order_type, 'opg') AS order_type,
               f.side, f.fill_price, f.filled_shares, i.last_price,
               a.open, a.close, y.open, y.close
        FROM paper_fills f
        LEFT JOIN intended_orders i ON i.intended_order_id = f.intended_order_id
        LEFT JOIN prices a ON a.ticker = f.ticker AND a.date = CAST(f.filled_at AS DATE)
                          AND a.source = 'alpaca'
        LEFT JOIN prices y ON y.ticker = f.ticker AND y.date = CAST(f.filled_at AS DATE)
                          AND y.source = 'yfinance'
        WHERE f.filled_shares > 0 AND f.fill_price > 0 AND f.filled_at IS NOT NULL
          AND (? IS NULL OR CAST(f.filled_at AS DATE) >= ?)
    """, [since, since]).fetchall()

    groups: dict[tuple[str, str], dict] = {}
    for session, otype, side, px, qty, last_price, a_o, a_c, y_o, y_c in rows:
        g = groups.setdefault((session, otype), {"bps": [], "w": [], "cbps": [], "cw": [],
                                                 "excluded": 0, "no_bench": 0})
        open_print, close_print, print_source = _pick_print_prices(px, a_o, a_c, y_o, y_c)
        notional = float(px) * float(qty)
        if session == "open":
            if print_source == CF_UNAVAILABLE:
                g["excluded"] += 1
                continue
            bench = open_print
        else:
            bench = last_price
        bp = _signed_cost_bp(side, px, bench)
        if bp is None:
            g["no_bench"] += 1
        elif abs(bp) > max_abs_bp:
            g["excluded"] += 1
            continue
        else:
            g["bps"].append(bp)
            g["w"].append(notional)
        cbp = _signed_cost_bp(side, px, close_print)
        if cbp is not None and abs(cbp) <= max_abs_bp:
            g["cbps"].append(cbp)
            g["cw"].append(notional)

    def _nw(vals, w):
        tot = sum(w)
        return sum(v * x for v, x in zip(vals, w, strict=True)) / tot if tot else None

    out = []
    for (session, otype), g in sorted(groups.items()):
        bps = sorted(g["bps"])
        n = len(bps)
        med = None
        if n:
            med = bps[n // 2] if n % 2 else (bps[n // 2 - 1] + bps[n // 2]) / 2.0
        out.append({
            "session": session,
            "order_type": otype,
            "n": n,
            "mean_bp": sum(bps) / n if n else None,
            "median_bp": med,
            "nw_mean_bp": _nw(g["bps"], g["w"]),
            "close_nw_bp": _nw(g["cbps"], g["cw"]),
            "excluded": g["excluded"],
            "no_benchmark": g["no_bench"],
        })
    return out


def format_fill_quality(rows: list[dict]) -> list[str]:
    """Small fixed-width table for `python -m sma.live status`."""
    def f(v):
        return "   n/a" if v is None else f"{v:+6.1f}"

    lines = [
        "Fill quality (bp/side, + = cost; open vs day's open print, "
        "sessions vs arrival mid):",
        f"  {'session':<8} {'type':<7} {'n':>4} {'mean':>6} {'median':>6} "
        f"{'nw':>6} {'vsClose':>7}",
    ]
    for r in rows:
        lines.append(
            f"  {r['session']:<8} {r['order_type']:<7} {r['n']:>4} {f(r['mean_bp'])} "
            f"{f(r['median_bp'])} {f(r['nw_mean_bp'])} {f(r['close_nw_bp']):>7}"
        )
    if len(lines) == 2:
        lines.append("  (no fills)")
    return lines
