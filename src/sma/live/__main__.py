"""CLI: python -m sma.live <subcommand>

Subcommands:
  decide              — 18:35 ET daily job: strategy → orders → Alpaca
  stop-loss-sweep     — 09:25 ET morning sweep: liquidate stop-loss triggers
  reconcile           — 16:30 ET end-of-day: fills + snapshot + drift detection
  status              — quick visibility (last fires + open positions; no API calls)
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from datetime import date as date_cls
from datetime import time as time_cls
from pathlib import Path
from zoneinfo import ZoneInfo

import click
from loguru import logger

from sma.config import load_settings
from sma.ingest.notify import notify_failure, send_ntfy
from sma.ingest.store import Store
from sma.ingest.universe import load_universe
from sma.live.alpaca_client import AlpacaClient
from sma.live.decide import decide_once
from sma.live.preflight import (
    PreflightHolidaySkipped,
    UpstreamMissedDeadline,
    run_preflight,
)
from sma.live.preopen_guard import run_preopen_guard
from sma.live.quantity import QTY_EPS
from sma.live.real_money import (
    RealMoneyRefusedError,
    build_alpaca_client,
    build_gate,
    force_dry_run,
    preflight_real_money,
)
from sma.live.reconcile import (
    RECORDED_SOURCES_SQL,
    backfill_missing_snapshots,
    backfill_official_closes,
    fill_quality_by_session,
    format_fill_quality,
    snapshot_staleness_message,
    snapshot_staleness_sessions,
)
from sma.live.reconcile import reconcile as run_reconcile
from sma.live.retry import _retry_transient
from sma.live.sizing import SizingPolicy, build_sizing_policy
from sma.live.stop_loss import stop_loss_sweep
from sma.live.trade_push import record_trade_push
from sma.locks import writer_lock
from sma.risk.rails import RiskRails
from sma.sched_adapter import get_adapter
from sma.sectors import sector_for
from sma.sentinels import write_sentinel

DEFAULT_DB = "data/sma.duckdb"
DEFAULT_CONFIG = "config.yaml"
DEFAULT_UNIVERSE = "src/sma/universe.yaml"


@click.group()
def cli() -> None:
    """Phase 5 Alpaca paper-trading live module."""


@cli.command("decide")
@click.option("--asof-date", default=None, help="ISO date (default: today)")
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Print orders WITHOUT submitting; writes nothing to DB.",
)
@click.option(
    "--canary",
    default=None,
    type=str,
    help="Filter decisions to a single ticker (emergency debug only).",
)
@click.option(
    "--use-theses",
    is_flag=True,
    default=False,
    help="Include Phase 4 LLM theses in strategy decisions.",
)
@click.option("--db", default=DEFAULT_DB, type=click.Path())
@click.option("--config", default=DEFAULT_CONFIG, type=click.Path(exists=True))
@click.option("--universe", default=DEFAULT_UNIVERSE, type=click.Path(exists=True))
def decide(
    asof_date,
    dry_run,
    canary,
    use_theses,
    db,
    config,
    universe,
):
    """Daily decide job — runs at 18:35 ET, fires DAY_OPG buys / DAY sells."""
    asof = date_cls.fromisoformat(asof_date) if asof_date else date_cls.today()
    try:
        _decide_impl(asof, dry_run, canary, use_theses, db, config, universe)
    except PreflightHolidaySkipped as e:
        # Market holiday: quiet no-op (exit 0, no sentinel, no page). The
        # watchdog and monitoring have their own holiday guards.
        click.echo(f"decide: skipped — {e} (market holiday)")
        return
    except UpstreamMissedDeadline as e:
        # 2026-08-05 post-mortem (8/4 no-trade outage): this is the exact
        # failure a battery-slowed Mac produces — predict never wrote a
        # sentinel and decide correctly refused to trade on stale/missing
        # predictions. The bare "decide failed" page used to leave 1am-Rayan
        # guessing what to run; name the missing upstream and give the two
        # copy-pasteable commands that heal it (no need to open a laptop and
        # think at 1am).
        if not dry_run:
            notify_failure(
                title="SMA decide FAILED — bot did NOT trade",
                message=f"{asof.isoformat()}: {e}\n{_rekick_hint(e.label)}",
            )
        raise
    except Exception as e:
        # A real decide failure is a NO-TRADE night: page a human immediately
        # rather than waiting for the 22:30 monitoring sweep (2026-06-09: the
        # preflight refusal exited 1 with only a log line). Manual dry-runs
        # must not page anyone.
        if not dry_run:
            notify_failure(
                title="SMA decide FAILED — bot did NOT trade",
                message=f"{asof.isoformat()}: {e}",
            )
        raise


def _rekick_hint(upstream_label: str | None) -> str:
    """One-line action hint for a missed-upstream-deadline page: names the
    stalled upstream in plain English and gives the exact recovery commands
    to re-kick it and then decide, so the notification alone is actionable
    (2026-08-05 post-mortem — see `decide`'s UpstreamMissedDeadline handler).

    Commands come from `sma.sched_adapter.get_adapter()` so this reads
    launchctl on macOS (today, byte-identical) and systemctl on a systemd
    host (host-migration-runbook.md Section 2c) without branching here.
    """
    if not upstream_label:
        return (
            "upstream job never produced a sentinel; if the Mac slept, plug "
            "in and check `python -m sma.live status`."
        )
    # "com.sma.model.predict.daily" -> "predict"; "com.sma.ingest.daily" ->
    # "ingest" — the segment right before the frequency suffix.
    short = upstream_label.split(".")[-2]
    adapter = get_adapter()
    return (
        f"upstream {short} never produced a sentinel; if the Mac slept, plug "
        f"in and run: {adapter.rekick_hint(upstream_label)} && "
        f"{adapter.rekick_hint('com.sma.live.decide.daily')}"
    )


def _decide_impl(asof, dry_run, canary, use_theses, db, config, universe):
    settings = load_settings(config_path=Path(config))
    # live.real_money.dry_run forces dry-run regardless of the CLI flag, and can
    # only ever ADD safety (never clears an explicit --dry-run). It is the knob
    # you leave on for the first week of a real-money account: the job
    # authenticates against the live endpoint, reads the real account, sizes real
    # orders, logs exactly what it would submit, and submits nothing.
    _rm_gate = build_gate(settings)
    _effective_dry_run = force_dry_run(dry_run, _rm_gate)
    if _effective_dry_run and not dry_run:
        logger.warning(
            "live.real_money.dry_run is set — forcing dry-run; no orders will "
            "be submitted at any endpoint"
        )
    dry_run = _effective_dry_run
    universe_list = load_universe(Path(universe))
    alpaca = _build_alpaca(settings)
    rails = _build_rails(settings)

    # Preflight: reads sentinels only; MUST run before acquiring writer_lock
    # or opening the writable Store. Running preflight while holding a writable
    # DuckDB connection was the source of the writer-blocks-self deadlock
    # (codex round-1 finding, 2026-04-30).
    run_preflight(asof=asof, db_path=Path(db), alpaca=alpaca)

    # Preflight passed: now safe to acquire writer_lock and open writable Store.
    # Evening-chain patience (2026-06-12): agents ran long on the first
    # 267-name night and held the lock past the default 30s — decide DIED
    # instead of queueing. 15 min covers a slow upstream while still failing
    # loud (with a page) on a true deadlock.
    with writer_lock(label="decide", timeout_s=900.0):
        store = Store(path=db).connect()
        try:
            strategy = _build_decide_strategy(
                universe_list, use_theses=use_theses, db=db, store=store,
                settings=settings,
            )
            import contextlib
            abort_pct = 0.30
            with contextlib.suppress(AttributeError):
                abort_pct = float(settings.live.drift.catastrophic_loss_abort_pct)
            peak_abort_pct = 0.25
            with contextlib.suppress(AttributeError):
                peak_abort_pct = float(
                    settings.live.drift.catastrophic_peak_drawdown_abort_pct
                )
            result = decide_once(
                asof=asof,
                store=store,
                alpaca=alpaca,
                universe=universe_list,
                strategy=strategy,
                sector_for=sector_for,
                rails=rails,
                catastrophic_loss_abort_pct=abort_pct,
                catastrophic_peak_drawdown_abort_pct=peak_abort_pct,
                canary=canary,
                dry_run=dry_run,
                sizing=_build_sizing(settings),
            )
            if not result.dry_run:
                _record_sleeve_attribution(store=store, asof=asof, sleeve_book=strategy)
            failed_rows = []
            if result.failed:
                failed_rows = store.conn.execute(
                    "SELECT ticker, side FROM intended_orders "
                    "WHERE asof_date = ? AND source = 'decide' "
                    "AND status IN ('submission_failed', 'recovery_failed') "
                    "ORDER BY side, ticker",
                    [asof],
                ).fetchall()
        finally:
            store.conn.close()
        # Sentinel write: INSIDE writer_lock (after store closed) so that
        # the ordering contract is satisfied: sentinel writes are serialized
        # by the lock (codex round-3 ordering rule).
        write_sentinel(
            label="com.sma.live.decide.daily",
            asof=asof,
            payload={
                "label": "com.sma.live.decide.daily",
                "asof": asof.isoformat(),
                "completed_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                "submitted_count": result.submitted,
                "failed_count": result.failed,
                "skipped_count": result.skipped,
                "dry_run": result.dry_run,
                "canary_ticker": canary,
                "decisions_total": result.decisions_after_rails,
            },
        )
        if result.failed and not result.dry_run:
            # Partial/total submit failure used to be SILENT: exceptions are
            # converted to status='submission_failed' rows, the CLI exits 0,
            # monitoring only checks sentinel existence, and next-day reconcile
            # can't see id-less rows — so a failed force-SELL silently skipped
            # a full day (review 2026-07-01, HIGH). Page a human, loudest for
            # sells (a skipped sell holds a position the model exited).
            detail = ", ".join(f"{s} {t}" for t, s in failed_rows) or f"{result.failed} orders"
            sells = sum(1 for _, s in failed_rows if str(s).upper() == "SELL")
            notify_failure(
                title=(
                    "SMA decide: SELL submit FAILED" if sells
                    else "SMA decide: order submits failed"
                ),
                message=(
                    f"{asof.isoformat()}: {result.failed} submit(s) failed "
                    f"({detail}). Not retried until tomorrow's decide — "
                    f"review the book now."
                ),
            )

    click.echo(
        f"decide: {result.decisions_after_rails} decisions after rails → "
        f"{result.submitted} submitted, {result.failed} failed "
        f"(dry_run={result.dry_run})"
    )

    # 2026-08-31: nightly trade push (Rayan's ask) -- last thing decide does,
    # deliberately OUTSIDE writer_lock (already released above) so a slow/
    # hung ntfy.sh request never extends how long the DB write lock is held
    # against reconcile/stop-loss-sweep. Bit-identical trading regardless of
    # notify.trade_pushes: this reads result.trade_push_message, which was
    # already fully computed inside decide_once before any orders here run
    # or don't -- flipping the flag only changes whether send_ntfy is
    # called, never what got submitted. dry_run leaves trade_push_message
    # None (nothing was actually submitted), so it's a no-op push. Wrapped
    # exactly like notify_failure's own sends: a push failure must never be
    # allowed to turn a successful decide run into a nonzero exit.
    if (
        not result.dry_run
        and settings.notify.trade_pushes
        and result.trade_push_message is not None
    ):
        delivered = False
        try:
            delivered = send_ntfy(result.trade_push_message, title=result.trade_push_title)
            if not delivered:
                logger.warning(
                    "trade push: send_ntfy returned False "
                    "(topic unset or send failed); message not delivered"
                )
        except Exception as e:  # a push failure must never fail a decide run
            logger.warning("trade push: send raised {}: {}", type(e).__name__, e)
        # 2026-09-04: persist what was pushed (title/message/structured orders
        # payload, delivered or not) so reconcile can later verify the numbers
        # against booked fills -- see sma.live.trade_push.record_trade_push and
        # sma.live.reconcile._detect_trade_push_drift. ntfy's own cache expires
        # in 12h, so without this nothing lets a bad push be caught after the
        # fact except a human eyeballing it (which is exactly how the LLY
        # 5x-overstated push was caught). Wrapped the same as the send above --
        # a persist failure must never fail a decide run.
        try:
            record_trade_push(
                asof=asof,
                title=result.trade_push_title,
                message=result.trade_push_message,
                orders=result.trade_push_orders or [],
                equity=result.equity or 0.0,
                delivered=bool(delivered),
            )
        except Exception as e:
            logger.warning("trade push: persist raised {}: {}", type(e).__name__, e)


@cli.command("replay")
@click.option("--asof-date", required=True, help="ISO date to replay decide's selection for.")
@click.option(
    "--book",
    type=click.Choice(["current", "asof", "empty"]),
    default="current",
    help=(
        "current: real live positions via AlpacaClient (read-only). "
        "asof: reconstructed from paper_fills up to --asof-date (see "
        "sma.live.replay.reconstruct_book_asof for documented limits). "
        "empty: a fresh, all-cash book."
    ),
)
@click.option(
    "--use-theses/--no-use-theses",
    default=True,
    help="Match production (com.sma.live.decide.daily runs with --use-theses).",
)
@click.option("--db", default=DEFAULT_DB, type=click.Path())
@click.option("--config", default=DEFAULT_CONFIG, type=click.Path(exists=True))
@click.option("--universe", default=DEFAULT_UNIVERSE, type=click.Path(exists=True))
def replay_cmd(asof_date, book, use_theses, db, config, universe):
    """"What would decide have done on --asof-date?" -- READ-ONLY.

    Runs the exact decide selection + rails pipeline (same strategy code,
    same config) against predictions/theses already stored for that date.
    Never submits an order, never writes a sentinel, never writes to the DB,
    and never takes the writer_lock -- see sma.live.replay's module
    docstring for the guarantees and how they're enforced.
    """
    from sma.live.replay import replay_once

    asof = date_cls.fromisoformat(asof_date)
    settings = load_settings(config_path=Path(config))
    universe_list = load_universe(Path(universe))
    rails = _build_rails(settings)
    sizing = _build_sizing(settings)

    alpaca = None
    if book == "current":
        alpaca = _build_alpaca(settings)

    result = replay_once(
        asof=asof,
        db_path=Path(db),
        universe=universe_list,
        rails=rails,
        sizing=sizing,
        settings=settings,
        book=book,
        use_theses=use_theses,
        alpaca=alpaca,
    )
    if result.aborted:
        click.echo(f"replay: {result.aborted_reason}")
        return
    for d in result.decisions:
        prior = f"{d.prior_weight:.2%}" if d.prior_weight is not None else "—"
        target = f"{d.target_weight:.2%}" if d.target_weight is not None else "—"
        order_str = f" [{d.side} {d.shares}]" if d.side else ""
        click.echo(
            f"  {d.action:14s} {d.ticker:6s} {prior:>7s} -> {target:>7s}"
            f"  ({d.rail}){order_str}"
        )
    click.echo(result.summary)


@cli.command("stop-loss-sweep")
@click.option("--asof-date", default=None)
@click.option("--db", default=DEFAULT_DB, type=click.Path())
@click.option("--config", default=DEFAULT_CONFIG, type=click.Path(exists=True))
def stop_loss_sweep_cmd(asof_date, db, config):
    """Morning stop-loss sweep — runs at 09:25 ET, force-sells triggered positions.

    NOOP under shipping config (RiskRails(stop_loss_pct=0)) per the rail diagnostic.
    """
    asof = date_cls.fromisoformat(asof_date) if asof_date else date_cls.today()
    try:
        _stop_loss_sweep_impl(asof, db, config)
    except Exception as e:
        # 2026-06-09: a 09:25 DNS failure killed the sweep silently — the bot
        # ran without stop-loss protection all day and nobody was told.
        notify_failure(
            title="SMA stop-loss sweep FAILED",
            message=f"{asof.isoformat()}: {e}",
        )
        raise


# Sane window for the morning sweep (scheduled 09:25 ET, 5 min before the open).
# launchd re-fires a MISSED StartCalendarInterval at the next boot or wake, and
# that catch-up is NOT suppressible from the plist — RunAtLoad was already false
# on every scheduled job when the 2026-08-12 reboot fired the whole manifest at
# 20:11. The sweep therefore ran at 20:15, 10.8h late, against after-hours marks,
# while the watchdog had explicitly declined to kick it (kicked=0, "too late to
# safely kick"). The guard has to live in the job.
#
# Defense in depth: NO kick mechanism (launchd catch-up, watchdog kickstart, a
# manual launchctl kickstart) may make this job read positions or submit orders
# outside the window. Mirrors preflight's deadline+6h abort for decide.
#
# The window is derived from the ACTUAL session (2026-08-16), not a flat clock.
# The original 09:20-16:05 band was wrong twice over:
#   - half-days (2026-11-27, 2026-12-24) close at 13:00 ET, so a 13:00-16:05
#     invocation would sweep POST-CLOSE marks and call them live;
#   - stop_loss.py's module contract is that the sweep PROXIES THE OPEN (it
#     uses the pre-market/last-close price as a stand-in for the auction
#     print), yet a watchdog late-kick as late as 15:30 passed the band and ran
#     an "open" proxy six hours after the open.
# Allowed: (session open - 10min) .. (session open + 15min). 09:25 sits in the
# middle of it; anything else is skipped with the existing skip-reason
# machinery.
_SWEEP_LEAD_MINUTES = 10
_SWEEP_TRAIL_MINUTES = 15

# Fallback band, used ONLY when the trading calendar is unreachable. A calendar
# blip must not BYPASS the guard (that is how the 20:15 sweep happened), so the
# same lead/trail band is applied to the standard 09:30 ET open instead.
_SWEEP_FALLBACK_OPEN_ET = time_cls(9, 20)
_SWEEP_FALLBACK_CLOSE_ET = time_cls(9, 45)


def _now_et() -> datetime:
    """Current wall clock in America/New_York. Indirection so tests can pin it."""
    return datetime.now(ZoneInfo("America/New_York"))


def _sweep_skip_reason(*, now: datetime, alpaca) -> tuple[str, str] | None:
    """Return (kind, detail) if the sweep must NOT run now, else None.

    kind is "holiday" (quiet skip) or "out_of_hours" (skip + page).

    The allowed band is anchored to `now`'s ACTUAL session open, so half-days
    and any future calendar change are handled by the calendar rather than by
    a constant that has to be remembered.
    """
    # Holiday first: a non-session day is a quieter, more fundamental reason to
    # skip than the clock, and paging every market holiday is pure noise (decide
    # treats a holiday as a quiet no-op for the same reason). A calendar lookup
    # failure falls back to the FIXED band — same defensive posture as before,
    # never "assume it's fine".
    try:
        session = alpaca.session_window(day=now.date())
    except Exception:  # noqa: BLE001 — calendar unreachable; fall back to the fixed band
        logger.warning(
            "stop-loss: trading-calendar lookup failed for {}; assuming a trading "
            "day and relying on the fixed fallback window", now.date(),
        )
        t = now.timetz().replace(tzinfo=None)
        if t < _SWEEP_FALLBACK_OPEN_ET or t > _SWEEP_FALLBACK_CLOSE_ET:
            return (
                "out_of_hours",
                f"invoked at {now:%Y-%m-%d %H:%M %Z}, outside the fallback "
                f"{_SWEEP_FALLBACK_OPEN_ET:%H:%M}-{_SWEEP_FALLBACK_CLOSE_ET:%H:%M} "
                f"ET window (trading calendar unreachable)",
            )
        return None

    if session is None:
        return ("holiday", f"{now.date().isoformat()} is not a NYSE trading day")

    session_open, session_close = session
    start = session_open - timedelta(minutes=_SWEEP_LEAD_MINUTES)
    end = session_open + timedelta(minutes=_SWEEP_TRAIL_MINUTES)
    if now < start or now > end:
        return (
            "out_of_hours",
            f"invoked at {now:%Y-%m-%d %H:%M %Z}, outside the "
            f"{start:%H:%M}-{end:%H:%M} ET window around the "
            f"{session_open:%H:%M} open (session closes {session_close:%H:%M})",
        )
    return None


def _record_skipped_sweep(*, asof, kind: str, detail: str) -> None:
    """Write a flagged sentinel for a sweep that deliberately did no work.

    Terminating cleanly matters: with NO sentinel the watchdog re-kicks the job
    at every checkpoint inside its late-kick window and pages at every checkpoint
    past it. The sentinel deliberately carries NO run_id, so
    sentinels._is_strictly_newer refuses to let it overwrite a real run's
    sentinel — a genuine 09:25 sweep followed by a stray catch-up keeps the
    real record.

    That invariant only actually held once the SUCCESS payload started carrying
    a run_id (2026-08-16). Before that, both payloads were run_id-less, so
    _is_strictly_newer fell through to the completed_at comparison, which the
    LATER skip always won — a 20:15 launchd catch-up could overwrite the real
    morning run and erase halted_preopen_divergence / preopen_oversell_cancelled
    from the record. _stop_loss_sweep_impl now allocates the run_id.
    """
    payload = {
        "label": "com.sma.live.stop-loss.weekday",
        "asof": asof.isoformat(),
        "completed_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "positions_checked": 0,
        "positions_triggered": 0,
        "sells_submitted": 0,
        "halted_preopen_divergence": False,
        "preopen_oversell_cancelled": 0,
        "skip_detail": detail,
    }
    payload["holiday_skipped" if kind == "holiday" else "skipped_out_of_hours"] = True
    with writer_lock(label="stop_loss_skip"):
        write_sentinel(
            label="com.sma.live.stop-loss.weekday", asof=asof, payload=payload,
        )


def _stop_loss_sweep_impl(asof, db, config):
    settings = load_settings(config_path=Path(config))
    alpaca = _build_alpaca(settings)
    rails = _build_rails(settings)

    # Market-hours guard BEFORE any broker read: an out-of-hours sweep must not
    # even look at the book, let alone submit against after-hours marks.
    skip = _sweep_skip_reason(now=_now_et(), alpaca=alpaca)
    if skip is not None:
        kind, detail = skip
        logger.warning("stop-loss: SKIPPED ({}) — {}", kind, detail)
        _record_skipped_sweep(asof=asof, kind=kind, detail=detail)
        if kind == "out_of_hours":
            # A genuinely anomalous invocation: the morning sweep did not run
            # today and a human should know once (not once per watchdog pass).
            notify_failure(
                title="SMA stop-loss sweep SKIPPED (out of hours)",
                message=(
                    f"{asof.isoformat()}: the 09:25 sweep was {detail}. No positions "
                    "were evaluated and no orders were submitted. This is usually a "
                    "launchd catch-up after a reboot/wake — verify the book manually "
                    "if the market was open today."
                ),
            )
        click.echo(f"stop-loss-sweep: skipped — out of hours ({detail})"
                   if kind == "out_of_hours"
                   else f"stop-loss-sweep: skipped — {detail} (market holiday)")
        return

    # Fetch positions count before entering writer_lock (read-only Alpaca
    # call). Retried: a transient DNS/network blip must not cost the day's sweep.
    positions = _retry_transient(
        alpaca.get_positions, label="stop-loss get_positions"
    )
    positions_checked = len(positions)

    guard_cfg = settings.live.preopen_guard
    with writer_lock(label="stop_loss"):
        store = Store(path=db).connect()
        halted = False
        divergence = None
        run_id = None
        try:
            # Allocate the run_id HERE, not inside stop_loss_sweep: the success
            # sentinel below must carry one on EVERY path (halted, NOOP rails,
            # or a real sweep), and it must be the same id the sweep's rows
            # carry. Without it, _record_skipped_sweep's run_id-less skip
            # payload could win the completed_at fallback in
            # sentinels._is_strictly_newer and overwrite a real morning run —
            # erasing halted_preopen_divergence and the guard's oversell count
            # from the forensic record (2026-08-16 review).
            run_id = store.allocate_run_id()
            # Pre-open safety FIRST: if the live broker book has lost a large
            # fraction of the positions our ledger holds (a broker-side wipe/
            # glitch — incident 2026-07-07), CANCEL the day's queued orders + PAGE
            # and do NOT trade on the untrustworthy book. Runs regardless of the
            # (default-OFF) price exits.
            if guard_cfg.enabled:
                try:
                    divergence = run_preopen_guard(
                        store=store,
                        broker_positions=positions,
                        alpaca=alpaca,
                        notify_fn=notify_failure,
                        min_missing_fraction=guard_cfg.min_missing_fraction,
                        min_ledger_positions=guard_cfg.min_ledger_positions,
                        equity_crash_fraction=guard_cfg.equity_crash_fraction,
                    )
                except Exception as e:  # noqa: BLE001 — a guard read error must not
                    # take down the whole sweep job / block the sentinel. Page a
                    # degraded warning and proceed (the sweep is a NOOP under
                    # shipping config anyway).
                    notify_failure(
                        title="SMA pre-open guard degraded (skipped)",
                        message=f"{asof.isoformat()}: pre-open guard errored ({e!r}); "
                                "the divergence check did NOT run this morning.",
                    )
            # A per-ticker oversell cancel does NOT halt the day (the harmful
            # order is already pulled); only a corroborated whole-book wipe does.
            if divergence is not None and divergence.wipe_halt:
                halted = True
                result = None
            else:
                result = stop_loss_sweep(
                    asof=asof,
                    store=store,
                    alpaca=alpaca,
                    rails=rails,
                    run_id=run_id,
                )
        finally:
            store.conn.close()
        # Sentinel write: INSIDE writer_lock (after store closed) so that the
        # ordering contract is satisfied: sentinel writes are serialized by the
        # lock (same pattern as decide, reconcile).
        payload = {
            "label": "com.sma.live.stop-loss.weekday",
            "asof": asof.isoformat(),
            "completed_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            # run_id is what makes this record un-clobberable: the skip
            # sentinel carries none, and sentinels._is_strictly_newer refuses
            # to let a run_id-less payload replace one that has a run_id.
            "run_id": run_id,
            "positions_checked": positions_checked,
            "positions_triggered": 0 if halted else result.triggered,
            "sells_submitted": 0 if halted else result.submitted,
            "halted_preopen_divergence": halted,
            "preopen_oversell_cancelled": (
                len(divergence.oversell_cancelled) if divergence is not None else 0
            ),
        }
        if halted:
            payload["missing_names"] = divergence.divergence.missing_names
        write_sentinel(
            label="com.sma.live.stop-loss.weekday", asof=asof, payload=payload,
        )

    if halted:
        click.echo(
            f"stop-loss-sweep: HALTED (pre-open wipe: {divergence.divergence.detail})"
        )
    else:
        click.echo(
            f"stop-loss-sweep: {result.triggered} triggered, "
            f"{result.submitted} submitted, {result.failed} failed"
        )


def _today_et() -> date_cls:
    """Today's date in America/New_York. Indirection so tests can pin it."""
    from zoneinfo import ZoneInfo
    return datetime.now(ZoneInfo("America/New_York")).date()


def _resolve_reconcile_asofs(asof_date: str | None, store) -> list[date_cls]:
    """Resolve the decide batches reconcile should process this run.

    Explicit date wins (rejected if in the future). Otherwise return ALL placed
    decide batches without a reconcile sentinel, OLDEST first — never just the
    newest. The old newest-first single-batch pick permanently stranded any
    older deferred batch: each 16:30 run found yesterday's fresh batch first,
    so a batch deferred on a 503/calendar failure was shadowed forever (fills
    never recorded, missed-SELL drift never checked, min-hold entry dates
    lost), and a multi-day outage backlog drained at 0/day — one new batch
    born per weekday, one reconciled (review 2026-07-01, HIGH).

    Returns [] when no unreconciled batches exist. Callers MUST handle that by
    skipping fill recording AND skipping the reconcile-sentinel write — writing
    a sentinel for "today" before today's 20:00 decide submits is the
    phantom-sentinel bug that strands future reconciles. The account snapshot
    itself should still be written under today's date.
    """
    if asof_date:
        parsed = date_cls.fromisoformat(asof_date)
        today = _today_et()
        if parsed > today:
            raise click.ClickException(
                f"--asof-date {parsed.isoformat()} is in the future "
                f"(today ET is {today.isoformat()}); refusing to write "
                f"future-dated reconcile state."
            )
        return [parsed]
    from sma.sentinels import read_sentinel
    # Candidate batches = any decide order that was PLACED at the broker
    # (alpaca_order_id IS NOT NULL) OR is still 'submitted' (covers crash-after-
    # accept rows whose id was never written). NOT status='submitted' alone:
    # reconcile updates statuses to terminal, so if it crashes/defers before
    # writing the sentinel, a status-only filter would make the batch invisible
    # and strand it (no sentinel + not re-discoverable). The sentinel below
    # remains the authoritative "done" gate.
    rows = store.conn.execute(
        "SELECT DISTINCT asof_date FROM intended_orders "
        # 'stop-loss' too: a day a price exit fired but decide placed no orders
        # (all deltas 0) has only a source='stop-loss' row; a 'decide'-only
        # resolver never surfaces it, orphaning the stop SELL fill forever and
        # pinning a permanent ledger_position_drift page (adversarial review
        # 2026-07-04). The recording queries were widened to match; this is the
        # discovery half.
        # Intraday session orders too (RECORDED_SOURCES_SQL, sma.live.session).
        f"WHERE {RECORDED_SOURCES_SQL} "
        "AND (alpaca_order_id IS NOT NULL "
        # submission_failed included: a submit that timed out on the RESPONSE
        # can still be live at the broker (accepted-then-timeout). The batch
        # must be reconciled so the coid backfill can authoritatively adopt or
        # clear it (review 2026-07-01) — a 404 there just records zero fills
        # and completes the batch.
        "     OR status IN ('submitted', 'recovery_failed', 'submission_failed')) "
        "ORDER BY asof_date ASC"
    ).fetchall()
    return [
        cand for (cand,) in rows
        if read_sentinel(label="com.sma.live.reconcile.daily", asof=cand) is None
    ]


def _recompute_counterfactuals(db) -> None:
    """reconcile --recompute-counterfactuals (2026-10-01): rewrite every
    fill_counterfactuals row with the alpaca-first print rule. A job: writer_lock."""
    from sma.live.reconcile import counterfactual_stats, record_fill_counterfactuals

    def _fmt(st):
        def f(x):
            return "-" if x is None else f"{x:.2f}"
        return (
            f"n={st['n']} open_bp mean={f(st['open_bp_mean'])} "
            f"median={f(st['open_bp_median'])} close_bp mean={f(st['close_bp_mean'])} "
            f"median={f(st['close_bp_median'])} null_open={st['open_bp_null']} "
            f"null_close={st['close_bp_null']} by_source={st['by_source']}"
        )

    with writer_lock(label="reconcile-recompute-counterfactuals"):
        store = Store(path=db).connect()
        try:
            before = counterfactual_stats(store.conn)
            n = record_fill_counterfactuals(store=store, recompute=True)
            after = counterfactual_stats(store.conn)
        finally:
            store.close()
    click.echo(f"counterfactuals before: {_fmt(before)}")
    click.echo(f"counterfactuals after:  {_fmt(after)}")
    click.echo(f"rows rewritten: {n}")


@cli.command("reconcile")
@click.option("--asof-date", default=None)
@click.option("--db", default=DEFAULT_DB, type=click.Path())
@click.option("--config", default=DEFAULT_CONFIG, type=click.Path(exists=True))
@click.option(
    "--recompute-counterfactuals", is_flag=True, default=False,
    help="ONLY rewrite every fill_counterfactuals row (alpaca prints first) under "
         "the writer_lock, print before/after stats, and exit. No broker calls, "
         "no snapshot, no sentinel.",
)
def reconcile_cmd(asof_date, db, config, recompute_counterfactuals):
    """End-of-day reconcile — runs at 16:30 ET, persists fills + detects drift.

    Two paths:
      1. Unreconciled batch exists → reconcile it (record fills, snapshot today,
         detect drift, write sentinel for that batch's asof).
      2. No unreconciled batches → write today's account_snapshot only.
         Crucially, DO NOT write a reconcile sentinel for today. Today's batch
         won't be submitted until decide@20:00 ET; writing a today-sentinel at
         16:30 would phantom-block tomorrow's reconcile from picking it up.
    """
    if recompute_counterfactuals:
        _recompute_counterfactuals(db)
        return
    settings = load_settings(config_path=Path(config))
    alpaca = _build_alpaca(settings)
    # Make config.yaml's live.drift.* alert thresholds authoritative (they were
    # loaded then ignored; reconcile used hardcoded constants — review 2026-07-04).
    # Real settings always carry live.drift (a Pydantic model); the guard falls
    # back to the module-default thresholds only for minimal test stubs.
    from sma.live.reconcile import DriftThresholds
    drift_cfg = getattr(getattr(settings, "live", None), "drift", None)
    drift_thresholds = (
        DriftThresholds.from_config(drift_cfg) if drift_cfg is not None else DriftThresholds()
    )

    with writer_lock(label="reconcile"):
        store = Store(path=db).connect()
        no_batches = False
        results: list = []  # (asof, result) per processed batch, oldest first
        try:
            # Self-heal (2026-08-20): before anything else, INSERT a row for
            # any recent trading session that has NO account_snapshots row at
            # all (a dead host through the whole day, e.g. 2026-08-19) --
            # unlike the correction below, there is no existing row to heal
            # here. Runs under this same writer_lock.
            backfill_missing_snapshots(store=store, alpaca=alpaca)
            # Self-heal (2026-08-13): replace any recent 1Min-proxy equity
            # with Alpaca's official daily close now that it may have
            # published. Runs under this same writer_lock.
            backfill_official_closes(store=store, alpaca=alpaca)

            asofs = _resolve_reconcile_asofs(asof_date, store)
            today = _today_et()

            if not asofs:
                # No unreconciled batches; snapshot today's state only — unless
                # today's session hasn't closed yet (or today isn't a session
                # at all), in which case get_account() prices off a mark that
                # is not today's close (before the open it's literally last
                # night's after-hours mark: the 2026-08-20 05:36 launchd
                # catch-up wrote a phantom "close" off Wednesday night's
                # price). Skip; the next post-close reconcile or gap-fill
                # (backfill_missing_snapshots, above) writes the row correctly.
                from sma.live.reconcile import (
                    _pre_close_skip_reason,
                    _write_account_snapshot,
                )
                now = _now_et()
                skip_reason = _pre_close_skip_reason(
                    snapshot_date=today, now=now, alpaca=alpaca,
                )
                if skip_reason is not None:
                    logger.info(
                        "reconcile: skipping account_snapshots write for {}: {}",
                        today, skip_reason,
                    )
                    snapshot_written = False
                else:
                    # _write_account_snapshot returns (written, positions). Binding
                    # the TUPLE to snapshot_written made the "snapshot_written=..."
                    # log line dump the whole position book, and made a FAILED
                    # write read as truthy (a non-empty tuple always is) on this
                    # path — the no-batches path is the one that runs most days.
                    snapshot_written, _snapshot_positions = _write_account_snapshot(
                        snapshot_date=today, store=store, alpaca=alpaca, now=now,
                    )
                no_batches = True
            else:
                # Drain the WHOLE backlog oldest-first, one batch at a time —
                # a deferred/outage batch must not be shadowed by newer ones.
                for asof in asofs:
                    results.append((asof, run_reconcile(
                        asof=asof,
                        store=store,
                        alpaca=alpaca,
                        notify_fn=_notify,
                        snapshot_date=today,
                        thresholds=drift_thresholds,
                        # Defer the ledger-drift check to ONE post-drain pass
                        # below: a fill recorded in a later batch must net an
                        # earlier batch's gap, else draining a backlog (or a
                        # same-day stop exit) pages spuriously (review 2026-07-04).
                        check_ledger_drift=False,
                    )))
                account = alpaca.get_account()
                positions = alpaca.get_positions()
                # Single ledger-vs-broker drift check AFTER every batch's fills
                # are recorded, against the current live book.
                from sma.live.reconcile import _detect_ledger_position_drift
                try:
                    for alert in _detect_ledger_position_drift(store=store, book=positions):
                        _notify(f"[{alert.kind}] {alert.detail}")
                except Exception:  # noqa: BLE001 — advisory data-integrity check
                    from loguru import logger as _logger
                    _logger.warning("post-drain ledger-drift check failed; skipped", exc_info=True)
            _score_sleeves(store)
        finally:
            store.conn.close()

        if no_batches:
            # Liveness sentinel (run-date): the watchdog checks THIS label —
            # the batch sentinel is keyed by the reconciled decide-date, never
            # today (phantom-sentinel protection), which made the watchdog
            # re-kick reconcile every hour, every day.
            write_sentinel(
                label="com.sma.live.reconcile.daily.ran",
                asof=today,
                payload={
                    "label": "com.sma.live.reconcile.daily.ran",
                    "asof": today.isoformat(),
                    "completed_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                    "reconciled_asof": None,
                },
            )
            click.echo(
                f"reconcile: no unreconciled batches; "
                f"snapshot_written={snapshot_written} (date={today.isoformat()})"
            )
            return
        # Defer each batch's sentinel until its next-session open has actually
        # occurred. Otherwise a Mon-after-holiday reconcile would
        # phantom-sentinel Friday's still-unfilled OPG batch with 0 fills,
        # and Tuesday's reconcile would skip the batch forever — same failure
        # mode as the 5/7-5/21 38-paper_fills-missing bug, just with a
        # holiday-shaped boundary instead of a same-day boundary.
        # Complete a batch ONLY when we are SURE: the next-session open has
        # occurred (calendar confirmed) AND every placed order was fetched
        # cleanly. A calendar failure or a 503'd order defers to tomorrow's
        # 16:30 run — which now retries it because the resolver returns ALL
        # sentinel-less batches oldest-first, not just the newest (the old
        # single-batch pick made "will retry next cycle" a lie: the newer
        # batch shadowed the deferred one forever; review 2026-07-01 HIGH).
        for asof, result in results:
            complete = (
                result.order_drift_open
                and result.calendar_lookup_ok
                and result.fetch_failures == 0
            )
            if complete:
                # Sentinel write: INSIDE writer_lock (after store closed) so
                # the ordering contract is satisfied: sentinel writes are
                # serialized by the lock (same pattern as decide/predict).
                write_sentinel(
                    label="com.sma.live.reconcile.daily",
                    asof=asof,
                    payload={
                        "label": "com.sma.live.reconcile.daily",
                        "asof": asof.isoformat(),
                        "completed_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                        "fills_persisted": result.fills_recorded,
                        "account_equity": (
                            account.get("equity") if isinstance(account, dict) else None
                        ),
                        "positions_count": len(positions) if isinstance(positions, dict) else None,
                        "calendar_lookup_ok": result.calendar_lookup_ok,
                    },
                )
            else:
                reason = (
                    f"calendar_lookup_ok={result.calendar_lookup_ok} "
                    f"fetch_failures={result.fetch_failures} "
                    f"open_occurred={result.order_drift_open}"
                )
                click.echo(
                    f"reconcile: deferred sentinel for asof={asof.isoformat()} — "
                    f"{reason}; will retry next cycle"
                )
                if not result.calendar_lookup_ok or result.fetch_failures:
                    notify_failure(
                        title="SMA reconcile incomplete — batch deferred",
                        message=f"{asof.isoformat()}: {reason}; retrying at next 16:30 run",
                    )

        # Liveness sentinel (run-date) for the watchdog — see no-batches path.
        write_sentinel(
            label="com.sma.live.reconcile.daily.ran",
            asof=today,
            payload={
                "label": "com.sma.live.reconcile.daily.ran",
                "asof": today.isoformat(),
                "completed_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                # single date (newest) — the dashboard feeds this into
                # WHERE asof_date = ?, so a comma-joined string silently
                # matches nothing (review 2026-07-02 M1). Full list alongside.
                "reconciled_asof": results[-1][0].isoformat() if results else None,
                "reconciled_asofs": [a.isoformat() for a, _ in results],
            },
        )

    for asof, result in results:
        click.echo(
            f"reconcile[{asof.isoformat()}]: {result.fills_recorded} fills, "
            f"snapshot={result.snapshot_written}, {len(result.alerts)} alerts"
        )
        for alert in result.alerts:
            click.echo(f"  [{alert.kind}] {alert.detail}")


@cli.command("status")
@click.option("--db", default=DEFAULT_DB, type=click.Path())
@click.option("--config", default=DEFAULT_CONFIG, type=click.Path())
def status(db, config):
    """Print last-fire times for each job + open positions count + today's spend.

    Pure DB read; no Alpaca API calls. Use to confirm the system is alive.
    """
    store = Store(path=db).connect(read_only=True)

    last_decide = _last_fire(store, source="decide")
    last_stop_loss = _last_fire(store, source="stop-loss")
    last_reconcile = _last_snapshot_time(store)

    open_positions = _open_positions_count(store)
    today_spend = _today_spend_usd(store)

    click.echo(f"Last decide fire:        {_fmt_iso(last_decide['ts'])} ({last_decide['summary']})")
    click.echo(
        f"Last stop-loss fire:     {_fmt_iso(last_stop_loss['ts'])} ({last_stop_loss['summary']})"
    )
    click.echo(
        f"Last reconcile fire:     {_fmt_iso(last_reconcile['ts'])} ({last_reconcile['summary']})"
    )
    # Staleness caveat (2026-08-20): a dead host can leave the latest snapshot
    # multiple sessions behind the live account (2026-08-19: +19% overnight on
    # MRNA earnings while status kept quoting 8/18). DB-only, no Alpaca call --
    # same wording as the dashboard's Equity History banner.
    if last_reconcile["asof_date"] is not None:
        staleness = snapshot_staleness_sessions(
            conn=store.conn, snapshot_date=last_reconcile["asof_date"]
        )
        staleness_msg = snapshot_staleness_message(
            sessions=staleness, snapshot_date=last_reconcile["asof_date"]
        )
        if staleness_msg:
            click.echo(f"  WARNING: {staleness_msg}")
    click.echo(f"Open positions (from fills): {open_positions}")
    click.echo(f"Today's API spend:       ${today_spend:.4f}")
    try:  # read-only; a DB not yet migrated to v11 has no session columns
        for line in format_fill_quality(fill_quality_by_session(store.conn)):
            click.echo(line)
    except Exception as e:  # noqa: BLE001 - status must never crash on a measurement
        click.echo(f"Fill quality: unavailable ({type(e).__name__})")
    for line in _sleeve_status_lines(store, config):
        click.echo(line)


# ---- builders + helpers ---------------------------------------------------


def _build_alpaca(settings) -> AlpacaClient:
    """Broker client for every live job.

    Routes to PAPER unless `live.real_money` is fully armed — see
    sma.live.real_money, which owns the gates. Unarmed config (the default, and
    what the paper bot runs on) takes the identical path it always has.
    """
    try:
        alpaca = build_alpaca_client(settings)
    except ValueError as e:
        raise click.ClickException(str(e)) from e
    if not alpaca.paper:
        gate = build_gate(settings)
        try:
            preflight_real_money(alpaca=alpaca, gate=gate)
        except RealMoneyRefusedError as e:
            raise click.ClickException(str(e)) from e
    return alpaca


def _build_sizing(settings) -> SizingPolicy:
    """Capital-scale sizing policy from `live.sizing`. All-default config =
    whole shares, no order-size floor, no ADV cap = today's exact behaviour."""
    return build_sizing_policy(settings)


def _build_rails(settings) -> RiskRails:
    """Construct RiskRails from config.yaml `live.rails.*` if present, else defaults.

    Phase 5 ships with stop_loss_pct=0 (rail disabled) per the rail diagnostic.
    """
    live = getattr(settings, "live", None)
    if live and getattr(live, "rails", None):
        r = live.rails
        return RiskRails(
            stop_loss_pct=getattr(r, "stop_loss_pct", 0.0),
            # Smarter price exits (default OFF). Mirrored in the sim + live sweep.
            trailing_stop_pct=getattr(r, "trailing_stop_pct", 0.0),
            take_profit_pct=getattr(r, "take_profit_pct", 0.0),
            cash_floor_pct=getattr(r, "cash_floor_pct", 0.05),
            max_sector_pct=getattr(r, "max_sector_pct", 0.25),
            max_drawdown_pct=getattr(r, "max_drawdown_pct", 0.15),
            max_position_pct=getattr(r, "max_position_pct", 0.05),
            # Default 1 to match LiveRails default — same-day churn rail.
            min_hold_days=getattr(r, "min_hold_days", 1),
            drawdown_derisk_start=getattr(r, "drawdown_derisk_start", 0.05),
            drawdown_derisk_slope=getattr(r, "drawdown_derisk_slope", 0.0),
            drawdown_derisk_cap=getattr(r, "drawdown_derisk_cap", 0.60),
            # Default 0.10 — skip rebalances <10% of current position size.
            rebalance_dead_zone_pct=getattr(r, "rebalance_dead_zone_pct", 0.10),
            # Live-only safety buffer on estimated same-day sell proceeds.
            sell_proceeds_haircut=getattr(r, "sell_proceeds_haircut", 1.0),
        )
    # Default for Phase 5: stop_loss disabled, all other rails at default
    return RiskRails(stop_loss_pct=0.0)


def _build_decide_strategy(
    universe: list[str], *, use_theses: bool, db: str, store, settings=None, predictor=None,
):
    """What decide (and replay) hand to decide_once: a SleeveBook over the
    enabled open-session sleeves in `strategies.sleeves`. The default config
    (xgb_momentum alone, live, 1.0) yields exactly the incumbent's decisions;
    see tests/integration/live/test_sleeve_golden.py."""
    from sma.config import StrategiesSettings
    from sma.strategies.allocator import build_sleeve_book
    from sma.strategies.registry import SleeveBuild

    # Real Settings always carry a StrategiesSettings; minimal test stubs
    # (MagicMock settings) fall back to the default, i.e. the incumbent alone.
    strategies_cfg = getattr(settings, "strategies", None)
    if not isinstance(strategies_cfg, StrategiesSettings):
        strategies_cfg = StrategiesSettings()
    return build_sleeve_book(
        strategies_cfg,
        SleeveBuild(
            universe=universe, use_theses=use_theses, db=db, store=store,
            settings=settings, predictor=predictor,
            # Looked up at call time, so patching _build_strategy (tests)
            # still controls what the xgb_momentum sleeve runs.
            incumbent_factory=_build_strategy,
        ),
        session="open",
    )


def _record_sleeve_attribution(*, store, asof, sleeve_book) -> None:
    """Persist every sleeve's proposed book (live and shadow) and score any
    pending sleeve returns. Attribution only: it runs after orders are
    already submitted and a failure here is logged, never raised, so it can
    never cost a trade night."""
    proposals = getattr(sleeve_book, "proposals", None)
    if proposals is None:
        return
    try:
        from sma.strategies.attribution import persist_sleeve_targets

        run_id = store.allocate_run_id()
        persist_sleeve_targets(
            store.conn, asof=asof, session=getattr(sleeve_book, "session", "open"),
            proposals=proposals, run_id=run_id,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("sleeve attribution: persist failed: {}: {}", type(e).__name__, e)
    _score_sleeves(store)


def _score_sleeves(store) -> None:
    try:
        from sma.strategies.attribution import score_pending

        n = score_pending(store.conn, run_id=store.allocate_run_id())
        if n:
            logger.info("sleeve attribution: scored {} sleeve-day(s)", n)
    except Exception as e:  # noqa: BLE001
        logger.warning("sleeve attribution: scoring failed: {}: {}", type(e).__name__, e)


def _sleeve_status_lines(store, config) -> list[str]:
    """`status` must never need API secrets (load_settings would), so read
    just the strategies block from the YAML, tolerantly."""
    try:
        import yaml

        from sma.config import StrategiesSettings
        from sma.strategies.attribution import format_sleeve_status, sleeve_status

        raw = {}
        if Path(config).exists():
            raw = (yaml.safe_load(Path(config).read_text()) or {}).get("strategies") or {}
        cfg = StrategiesSettings(**raw)
        return format_sleeve_status(sleeve_status(store.conn, cfg.sleeves))
    except Exception as e:  # noqa: BLE001
        return [f"Sleeves: unavailable ({type(e).__name__}: {e})"]


def _build_strategy(
    universe: list[str], *, use_theses: bool, db: str, store, settings=None, predictor=None,
):
    """The incumbent XGBoostTopKStrategy, un-sleeved. Kept for callers that
    want the raw strategy (and as the golden test's reference path); the
    builder itself lives in sma.strategies.xgb_momentum so the sleeve and
    this function can never drift."""
    from sma.strategies.xgb_momentum import build_incumbent_strategy

    return build_incumbent_strategy(
        universe, use_theses=use_theses, db=db, store=store, settings=settings,
        predictor=predictor,
    )


def _notify(message: str) -> None:
    """Wire reconcile drift alerts into the existing notify pipeline."""
    try:
        notify_failure(title="sma.live drift", message=message)
    except Exception:
        # Notify failure must never break the calling job.
        click.echo(f"[notify failed] {message}", err=True)


def _last_fire(store, *, source: str) -> dict:
    row = store.conn.execute(
        "SELECT MAX(created_at), COUNT(*) FROM intended_orders WHERE source = ?",
        [source],
    ).fetchone()
    if row is None or row[0] is None:
        return {"ts": None, "summary": "never"}
    return {"ts": row[0], "summary": f"{row[1]} total intended orders"}


def _last_snapshot_time(store) -> dict:
    """Summarize the MOST RECENT account snapshot.

    Read the latest ROW — never three independent aggregates. The previous
    `MAX(asof_date), MAX(created_at), MAX(equity)` paired the newest date with
    the highest equity EVER recorded, so during a drawdown `status` printed the
    high-water mark as if it were current: live on 2026-07-29 it claimed
    `asof=2026-07-28, equity=$111,202.45` (the 7/7 peak) while the account was
    at $96,585.61. A health check must not hide the condition it exists to show.
    """
    row = store.conn.execute(
        "SELECT asof_date, created_at, equity FROM account_snapshots "
        "ORDER BY asof_date DESC, created_at DESC LIMIT 1"
    ).fetchone()
    if row is None or row[0] is None:
        return {"ts": None, "summary": "never", "asof_date": None}
    return {
        "ts": row[1],
        "summary": f"asof={row[0]}, equity=${float(row[2] or 0):,.2f}",
        "asof_date": row[0],
    }


def _open_positions_count(store) -> int:
    """Count tickers held net-long per the paper_fills ledger.

    Regression fixed 2026-09-06: this used to be `COUNT(DISTINCT ticker)` over
    every BUY ever placed via decide, never offset by a SELL, despite its own
    docstring claiming otherwise -- a permanently-wrong gauge that printed 59
    while the real book held 11 names (the class of bug this project has a
    standing lesson about: fix the gauge or delete it).

    Net shares per ticker = SUM(BUY filled_shares) - SUM(SELL filled_shares),
    the same ledger math `_detect_ledger_position_drift` in reconcile.py uses
    to check the ledger against the live broker book. A ticker counts as open
    when its net clears QTY_EPS: a fully-exited position can leave float dust
    behind (e.g. BUY 0.1 + BUY 0.2 - SELL 0.3 nets to ~5.5e-17 in IEEE754
    double arithmetic, not exactly 0) that must not read as still-open.

    Falls back to the latest account_snapshots.position_count when
    paper_fills has no rows yet (cold DB, or before the first reconcile has
    run) so a legitimate book still reads as non-zero.

    Approximation; the source of truth for live positions is Alpaca itself.
    """
    nets = store.conn.execute("""
        SELECT ticker,
               SUM(CASE WHEN UPPER(side) = 'BUY' THEN filled_shares
                        ELSE -filled_shares END) AS net
        FROM paper_fills
        GROUP BY ticker
    """).fetchall()
    if not nets:
        row = store.conn.execute(
            "SELECT position_count FROM account_snapshots "
            "ORDER BY asof_date DESC, created_at DESC LIMIT 1"
        ).fetchone()
        return int(row[0]) if row and row[0] is not None else 0
    return sum(1 for _, net in nets if (net or 0) > QTY_EPS)


def _today_spend_usd(store) -> float:
    """Sum est_cost_usd from agent_calls where created_at::DATE = today."""
    row = store.conn.execute("""
        SELECT COALESCE(SUM(est_cost_usd), 0)
        FROM agent_calls
        WHERE created_at::DATE = CURRENT_DATE
    """).fetchone()
    return float(row[0] or 0)


def _fmt_iso(ts) -> str:
    """ISO-8601 with TZ. None → 'never'."""
    if ts is None:
        return "never".ljust(25)
    # DuckDB returns datetime; render in local timezone with offset.
    return ts.astimezone().isoformat(timespec="seconds")


# Intraday trade sessions (sma.live.session); off by default.
from sma.live.session import session_cmd  # noqa: E402

cli.add_command(session_cmd)

if __name__ == "__main__":
    cli()
