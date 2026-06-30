"""CLI: python -m sma.live <subcommand>

Subcommands:
  decide              — 18:35 ET daily job: strategy → orders → Alpaca
  stop-loss-sweep     — 09:25 ET morning sweep: liquidate stop-loss triggers
  reconcile           — 16:30 ET end-of-day: fills + snapshot + drift detection
  status              — quick visibility (last fires + open positions; no API calls)
"""

from __future__ import annotations

from datetime import UTC, datetime
from datetime import date as date_cls
from pathlib import Path

import click

from sma.config import load_settings
from sma.ingest.notify import notify_failure
from sma.ingest.store import Store
from sma.ingest.universe import load_universe
from sma.live.alpaca_client import AlpacaClient
from sma.live.decide import decide_once
from sma.live.preflight import PreflightHolidaySkipped, run_preflight
from sma.live.reconcile import reconcile as run_reconcile
from sma.live.retry import _retry_transient
from sma.live.stop_loss import stop_loss_sweep
from sma.locks import writer_lock
from sma.risk.rails import RiskRails
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


def _decide_impl(asof, dry_run, canary, use_theses, db, config, universe):
    settings = load_settings(config_path=Path(config))
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
            strategy = _build_strategy(
                universe_list, use_theses=use_theses, db=db, store=store,
                settings=settings,
            )
            import contextlib
            abort_pct = 0.30
            with contextlib.suppress(AttributeError):
                abort_pct = float(settings.live.drift.catastrophic_loss_abort_pct)
            result = decide_once(
                asof=asof,
                store=store,
                alpaca=alpaca,
                universe=universe_list,
                strategy=strategy,
                sector_for=sector_for,
                rails=rails,
                catastrophic_loss_abort_pct=abort_pct,
                canary=canary,
                dry_run=dry_run,
            )
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

    click.echo(
        f"decide: {result.decisions_after_rails} decisions after rails → "
        f"{result.submitted} submitted, {result.failed} failed "
        f"(dry_run={result.dry_run})"
    )


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


def _stop_loss_sweep_impl(asof, db, config):
    settings = load_settings(config_path=Path(config))
    alpaca = _build_alpaca(settings)
    rails = _build_rails(settings)

    # Fetch positions count before entering writer_lock (read-only Alpaca
    # call). Retried: a transient DNS/network blip must not cost the day's sweep.
    positions = _retry_transient(
        alpaca.get_positions, label="stop-loss get_positions"
    )
    positions_checked = len(positions)

    with writer_lock(label="stop_loss"):
        store = Store(path=db).connect()
        try:
            result = stop_loss_sweep(
                asof=asof,
                store=store,
                alpaca=alpaca,
                rails=rails,
            )
        finally:
            store.conn.close()
        # Sentinel write: INSIDE writer_lock (after store closed) so that the
        # ordering contract is satisfied: sentinel writes are serialized by the
        # lock (same pattern as decide, reconcile).
        write_sentinel(
            label="com.sma.live.stop-loss.weekday",
            asof=asof,
            payload={
                "label": "com.sma.live.stop-loss.weekday",
                "asof": asof.isoformat(),
                "completed_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                "positions_checked": positions_checked,
                "positions_triggered": result.triggered,
                "sells_submitted": result.submitted,
            },
        )

    click.echo(
        f"stop-loss-sweep: {result.triggered} triggered, "
        f"{result.submitted} submitted, {result.failed} failed"
    )


def _today_et() -> date_cls:
    """Today's date in America/New_York. Indirection so tests can pin it."""
    from zoneinfo import ZoneInfo
    return datetime.now(ZoneInfo("America/New_York")).date()


def _resolve_reconcile_asof(asof_date: str | None, store) -> date_cls | None:
    """Resolve reconcile's --asof-date.

    Explicit date wins (rejected if in the future). Otherwise scan PLACED decide
    orders newest-first and return the first decide_date that does NOT already
    have a reconcile sentinel — i.e. the most recent unreconciled batch.

    Returns None when no unreconciled batches exist. Callers MUST handle the
    None case by skipping fill recording AND skipping the reconcile-sentinel
    write — writing a sentinel for "today" before today's 20:00 decide submits
    is the phantom-sentinel bug that strands future reconciles (the next-day
    16:30 reconcile sees today's sentinel and skips today's batch forever).
    The account snapshot itself should still be written under today's date.
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
        return parsed
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
        "WHERE source = 'decide' "
        "AND (alpaca_order_id IS NOT NULL "
        "     OR status IN ('submitted', 'recovery_failed')) "
        "ORDER BY asof_date DESC"
    ).fetchall()
    for (cand,) in rows:
        sentinel = read_sentinel(label="com.sma.live.reconcile.daily", asof=cand)
        if sentinel is None:
            return cand
    return None


@cli.command("reconcile")
@click.option("--asof-date", default=None)
@click.option("--db", default=DEFAULT_DB, type=click.Path())
@click.option("--config", default=DEFAULT_CONFIG, type=click.Path(exists=True))
def reconcile_cmd(asof_date, db, config):
    """End-of-day reconcile — runs at 16:30 ET, persists fills + detects drift.

    Two paths:
      1. Unreconciled batch exists → reconcile it (record fills, snapshot today,
         detect drift, write sentinel for that batch's asof).
      2. No unreconciled batches → write today's account_snapshot only.
         Crucially, DO NOT write a reconcile sentinel for today. Today's batch
         won't be submitted until decide@20:00 ET; writing a today-sentinel at
         16:30 would phantom-block tomorrow's reconcile from picking it up.
    """
    settings = load_settings(config_path=Path(config))
    alpaca = _build_alpaca(settings)

    with writer_lock(label="reconcile"):
        store = Store(path=db).connect()
        no_batches = False
        try:
            asof = _resolve_reconcile_asof(asof_date, store)
            today = _today_et()

            if asof is None:
                # No unreconciled batches; snapshot today's state only.
                from sma.live.reconcile import _write_account_snapshot
                snapshot_written = _write_account_snapshot(
                    snapshot_date=today, store=store, alpaca=alpaca,
                )
                no_batches = True
            else:
                result = run_reconcile(
                    asof=asof,
                    store=store,
                    alpaca=alpaca,
                    notify_fn=_notify,
                    snapshot_date=today,
                )
                account = alpaca.get_account()
                positions = alpaca.get_positions()
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
        # Defer the sentinel write until the asof batch's next-session open
        # has actually occurred. Otherwise a Mon-after-holiday reconcile would
        # phantom-sentinel Friday's still-unfilled OPG batch with 0 fills,
        # and Tuesday's reconcile would skip the batch forever — same failure
        # mode as the 5/7-5/21 38-paper_fills-missing bug, just with a
        # holiday-shaped boundary instead of a same-day boundary. Calendar-
        # lookup failure falls through to a sentinel write to avoid a
        # forever-retry loop on persistent calendar breakage.
        # Complete the batch ONLY when we are SURE: the next-session open has
        # occurred (calendar confirmed) AND every placed order was fetched
        # cleanly. A calendar failure or a 503'd order defers to tomorrow's
        # 16:30 run — bounded retry, watchdog stays quiet via the .ran
        # liveness sentinel, and a human is paged. The old write-anyway
        # calendar fallback predates both mechanisms and could strand fills
        # (Codex module review 2026-06-11, 2 HIGH).
        complete = (
            result.order_drift_open
            and result.calendar_lookup_ok
            and result.fetch_failures == 0
        )
        if complete:
            # Sentinel write: INSIDE writer_lock (after store closed) so that
            # the ordering contract is satisfied: sentinel writes are
            # serialized by the lock (same pattern as decide, predict, agents).
            write_sentinel(
                label="com.sma.live.reconcile.daily",
                asof=asof,
                payload={
                    "label": "com.sma.live.reconcile.daily",
                    "asof": asof.isoformat(),
                    "completed_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                    "fills_persisted": result.fills_recorded,
                    "account_equity": account.get("equity") if isinstance(account, dict) else None,
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
                "reconciled_asof": asof.isoformat(),
            },
        )

    click.echo(
        f"reconcile: {result.fills_recorded} fills, "
        f"snapshot={result.snapshot_written}, {len(result.alerts)} alerts"
    )
    for alert in result.alerts:
        click.echo(f"  [{alert.kind}] {alert.detail}")


@cli.command("status")
@click.option("--db", default=DEFAULT_DB, type=click.Path())
def status(db):
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
    click.echo(f"Open intended positions: {open_positions}")
    click.echo(f"Today's API spend:       ${today_spend:.4f}")


# ---- builders + helpers ---------------------------------------------------


def _build_alpaca(settings) -> AlpacaClient:
    api_key = settings.secrets.alpaca_api_key
    secret_key = settings.secrets.alpaca_api_secret
    if not api_key or not secret_key:
        raise click.ClickException(
            "ALPACA_API_KEY / ALPACA_API_SECRET must be set in .env or environment."
        )
    return AlpacaClient.paper_from_env(api_key=api_key, secret_key=secret_key)


def _build_rails(settings) -> RiskRails:
    """Construct RiskRails from config.yaml `live.rails.*` if present, else defaults.

    Phase 5 ships with stop_loss_pct=0 (rail disabled) per the rail diagnostic.
    """
    live = getattr(settings, "live", None)
    if live and getattr(live, "rails", None):
        r = live.rails
        return RiskRails(
            stop_loss_pct=getattr(r, "stop_loss_pct", 0.0),
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
        )
    # Default for Phase 5: stop_loss disabled, all other rails at default
    return RiskRails(stop_loss_pct=0.0)


def _build_strategy(universe: list[str], *, use_theses: bool, db: str, store, settings=None):
    from sma.backtest.strategies.xgb_top_k import XGBoostTopKStrategy
    from sma.model.predictor import Predictor

    # Pass the live store's connection to the predictor so it doesn't open a
    # second (conflicting) connection on the same DB file.
    predictor = Predictor(
        models_dir=Path("models_artifacts"),
        db_path=Path(db),
        conn=store.conn,
    )
    # Theses lookups also reuse the live store conn (same single-conn rule).
    # 2026-05-29: k=15 (was strategy default k=10). After bumping sector cap
    # 0.25 → 0.35 on 5/26, IT exposure stabilized at 29.4% (~5.6% cap
    # headroom) but the top-10 had 8 IT names — every available IT pick
    # (PANW/ARM/ORCL/AVGO/DDOG) was sector-cap-blocked. Non-IT picks that
    # would diversify cleanly (BBY/FDX/DKNG, all positive predictions) were
    # at ranks 11-13, outside the K=10 ceiling. Bumping to K=15 brings
    # them into the buy pool without changing per-name risk (weight_tilt
    # still active; top names get target_weight=10%, bottom of K=15 gets
    # floor=4%). Expected effect: deployment 43% → ~75%.
    # Strategy params from config (2026-06-12 ship: sector_neutralize=1.0 +
    # hold_rank=30 + rails.min_hold_days=7 — corrected-eval val sharpe +1.60
    # vs -1.58 baseline; plateau-checked in scripts/ab_wave4_robustness.py).
    strat_cfg = getattr(getattr(settings, "live", None), "strategy", None)
    k = getattr(strat_cfg, "k", 15) if strat_cfg else 15
    return XGBoostTopKStrategy(
        predictor=predictor,
        universe=universe,
        k=k,
        hold_rank=getattr(strat_cfg, "hold_rank", None) if strat_cfg else None,
        sector_neutralize=(
            getattr(strat_cfg, "sector_neutralize", 0.0) if strat_cfg else 0.0
        ),
        use_theses=use_theses,
        store=store if use_theses else None,
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
    row = store.conn.execute(
        "SELECT MAX(asof_date), MAX(created_at), MAX(equity) FROM account_snapshots"
    ).fetchone()
    if row is None or row[0] is None:
        return {"ts": None, "summary": "never"}
    return {
        "ts": row[1],
        "summary": f"asof={row[0]}, equity=${float(row[2] or 0):,.2f}",
    }


def _open_positions_count(store) -> int:
    """Count distinct tickers with at least one BUY intended order not offset by SELL.

    Approximation; the source of truth for live positions is Alpaca itself.
    Using intended_orders gives an offline estimate without an API call.
    """
    # "Placed at the broker" = alpaca_order_id IS NOT NULL, NOT status='submitted'
    # (reconcile now updates status to terminal values; 'submitted' would only
    # match still-pending orders and undercount).
    row = store.conn.execute("""
        SELECT COUNT(DISTINCT ticker) FROM intended_orders
        WHERE side = 'BUY' AND source = 'decide' AND alpaca_order_id IS NOT NULL
    """).fetchone()
    return int(row[0] or 0)


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


if __name__ == "__main__":
    cli()
