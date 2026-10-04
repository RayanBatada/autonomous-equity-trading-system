"""Periodic watchdog: detect missed scheduled jobs and kickstart them.

Runs on an hourly OS-scheduler cadence (19-22 ET) plus at user login/boot.
For each job in `sma.schedule.SCHEDULE` that runs today, check whether
its sentinel exists by deadline. If past deadline AND no sentinel, query
the scheduler's job state first (via `sma.sched_adapter.get_adapter()` --
launchd on macOS, systemd --user on Linux): only kickstart if the service
is idle ("not running" or unknown). If the service is already running or
waiting, skip without kickstarting.

Decision logic per job:
1. Skip if sentinel exists for today.
2. Skip if now < deadline.
3. Skip if scheduler state is "running" or "waiting" (log + increment skipped).
4. Else kickstart; increment kicked on rc=0, failures on rc!=0.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

from loguru import logger

from sma import schedule as sched
from sma.ingest.notify import notify_failure
from sma.readiness import sentinel_lineage_stale
from sma.sched_adapter import get_adapter
from sma.sentinels import read_sentinel

# Tripwire: if this file exists, an autoresearch eval died mid-swap and the
# LIVE strategy file may be an unevaluated LLM proposal (auto-restored at the
# next autoresearch start; until then, PAGE on every watchdog pass).
_AUTORESEARCH_BAK_PATH = Path("src/sma/strategy/active.py.autoresearch_bak")


def _is_trading_day(today, notify_fn=notify_failure) -> bool:
    """Return True if `today` is a NYSE trading day per Alpaca's calendar.

    Loads settings + Alpaca credentials from the default config.yaml.
    Falls back to True on any error (keeps existing watchdog behaviour for
    non-holiday weekdays if Alpaca is unreachable).
    """
    try:
        from pathlib import Path

        from sma.config import load_settings
        from sma.live.alpaca_client import AlpacaClient

        settings = load_settings(config_path=Path("config.yaml"))
        s = settings.secrets
        if not s.alpaca_api_key or not s.alpaca_api_secret:
            return True
        alpaca = AlpacaClient.paper_from_env(
            api_key=s.alpaca_api_key, secret_key=s.alpaca_api_secret
        )
        sessions = alpaca.sessions_between(start=today, end=today)
        return len(sessions) > 0
    except Exception as e:
        logger.warning(
            "watchdog: alpaca calendar lookup failed for {}; assuming trading day",
            today,
        )
        notify_fn(
            title="sma: watchdog calendar lookup failed",
            message=(
                f"Could not validate the trading calendar for {today} ({e!r}). "
                "Assuming a trading day; verify the market was/wasn't open."
            ),
        )
        return True


def _quality_passed(sentinel: dict) -> bool:
    """The same predicate as sentinels.ingest_succeeded_today: a sentinel is a
    success only if its quality block says passed. A holiday-skip sentinel is
    written with passed=True, so it stays done."""
    quality = sentinel.get("quality")
    return isinstance(quality, dict) and bool(quality.get("passed", False))


def _now() -> datetime:
    return datetime.now(tz=sched.NY_TZ)


def check(*, notify_fn=notify_failure) -> int:
    """Returns 0 if all good, 1 if any kickstart failed."""
    now = _now()
    today = now.date()
    adapter = get_adapter()
    failures = 0
    kicked = 0
    skipped = 0

    if _AUTORESEARCH_BAK_PATH.exists():
        notify_fn(
            title="sma: stranded autoresearch swap of active.py",
            message=(
                "src/sma/strategy/active.py.autoresearch_bak exists — a prior "
                "autoresearch eval died mid-swap and LIVE active.py may be an "
                "unevaluated LLM proposal. Run autoresearch (auto-restores) or "
                "restore the backup manually."
            ),
        )

    # Holiday guard: jobs that require a live market session (ingest, predict,
    # agents, decide, stop-loss, reconcile) produce no useful work on NYSE
    # holidays — kicking them just fails preflight and wastes lock time, so they
    # are skipped per-job below. But retrain/autoresearch/backup/senate/house run
    # regardless of the market (Mon holidays, weekends), so a blanket early-return
    # SILENCED a missed holiday-Monday retrain (review 2026-07-04). Evaluate every
    # job; skip only the market-sensitive ones on non-trading days.
    is_trading_day = _is_trading_day(today, notify_fn=notify_fn)
    if not is_trading_day:
        logger.info(
            "watchdog: {} is not a trading day; evaluating non-market jobs only",
            today,
        )

    # 2026-05-18: don't re-kick jobs that are far past deadline — kicking them
    # risks colliding with later-scheduled jobs on writer_lock (decide.daily,
    # backup.daily). The window is PER JOB (JobSchedule.late_kick_max_hours,
    # default 6h): heavy market-independent jobs (retrain/autoresearch/senate/
    # house) get wider windows so a Mac that sleeps through their early slot and
    # wakes midday still recovers them (2026-07-13: retrain missed entirely and
    # the model went a week stale under the flat 6h cap).
    # OPTIONAL jobs (intraday sessions / intraday ingest) join the loop only
    # when installed in the OS scheduler: they ship disabled and must not page.
    optional = tuple(
        j for j in sched.OPTIONAL_SCHEDULE if adapter.installed(j.label) is True
    )
    for job in (*sched.SCHEDULE, *optional):
        if not sched.runs_today(job.label, asof=today):
            continue
        # On a non-trading day, skip market-sensitive jobs (they have no work);
        # non-market jobs (retrain/autoresearch/backup/senate/house) still run.
        if job.requires_market_data and not is_trading_day:
            continue
        deadline = sched.deadline(job.label, asof=today)
        if now < deadline:
            continue
        # Sentinel FIRST, then lateness. The too-late page used to fire before
        # the sentinel was consulted, so a job that ran FINE hours ago paged
        # "missed job" at every later checkpoint (review 2026-07-20 HIGH — the
        # cried-wolf pages that buried the one real 7/13 miss).
        # Liveness label: reconcile's batch sentinel is keyed by the decide-
        # date it reconciled (yesterday), never today — check its run-date
        # liveness sentinel instead (pre-fix: pointless re-kicks every hour).
        check_label = job.liveness_sentinel_label or job.label
        sentinel = read_sentinel(label=check_label, asof=today)
        if (
            sentinel is not None
            and job.rekick_on_failed_quality
            and not _quality_passed(sentinel)
        ):
            # A failed verdict is not "done" for ingest: the CLI re-runs on
            # it (ingest_succeeded_today), so fall through to the lateness,
            # state and kick checks below (flaw hunt 2026-10-01 A1).
            logger.warning(
                f"watchdog: {job.label} sentinel for {today} failed quality "
                f"({(sentinel.get('quality') or {}).get('blocking_failures')}); "
                "treating as not done"
            )
            sentinel = None
        if sentinel is not None:
            # Lineage: a sentinel built from an OLDER upstream run than the
            # upstream currently records is STALE — treat as not-done so the
            # healed upstream produces fresh output (2026-06-09: predict kept
            # stale-feature predictions after the ingest heal; decide's
            # preflight WAITs on lineage, so this kick un-blocks the night).
            if job.lineage_upstream is not None:
                upstream_sentinel = read_sentinel(label=job.lineage_upstream, asof=today)
                if sentinel_lineage_stale(consumer=sentinel, upstream=upstream_sentinel):
                    logger.warning(
                        f"watchdog: {job.label} sentinel lineage is stale "
                        f"(consumed {sentinel.get('ingest_run_id')}, current "
                        f"{(upstream_sentinel or {}).get('run_id')}); re-kicking"
                    )
                else:
                    continue
            else:
                continue
        # Dependency gate (review 2026-07-20 #4): don't kick a job whose upstream
        # hasn't completed today — autoresearch kicked alongside retrain races
        # the heavy lock and can invert artifact precedence. Silent skip (no
        # page): the upstream's own miss already pages, and the next checkpoint
        # retries once the upstream sentinel lands.
        if job.kick_requires_upstream_sentinels and any(
            read_sentinel(label=up, asof=today) is None for up in job.depends_on
        ):
            logger.info(
                f"watchdog: {job.label} upstream sentinel(s) missing; "
                "deferring kick to a later checkpoint"
            )
            skipped += 1
            continue
        late_by_h = (now - deadline).total_seconds() / 3600
        if late_by_h > job.late_kick_max_hours:
            logger.warning(
                f"watchdog: {job.label} {late_by_h:.1f}h past deadline; "
                f"too late to safely kick (max {job.late_kick_max_hours}h); "
                "skipping until next natural fire"
            )
            # Don't skip SILENTLY — a late job with NO sentinel is a likely-
            # missed run and must reach a human (the freeze went unnoticed for
            # a week).
            notify_fn(
                title="sma: missed job (too late to kick)",
                message=(
                    f"{job.label} is {late_by_h:.1f}h past its deadline for {today} "
                    "and was NOT kicked (too late). The run was likely missed — "
                    "check the job's .err.log."
                ),
            )
            skipped += 1
            continue
        state = adapter.state(job.label)
        if state in ("running", "waiting"):
            logger.info(
                f"watchdog: {job.label} past deadline but scheduler"
                f" state={state!r}; skipping kickstart"
            )
            skipped += 1
            continue
        logger.warning(
            f"watchdog: {job.label} past deadline ({deadline}); state={state!r}; kickstarting"
        )
        cp = adapter.kickstart(job.label)
        if cp.returncode != 0:
            logger.error(
                f"watchdog: kickstart for {job.label} failed rc={cp.returncode} stderr={cp.stderr}"
            )
            notify_fn(
                title="sma: watchdog kickstart failed",
                message=(
                    f"{job.label} kickstart failed (rc={cp.returncode}): "
                    f"{cp.stderr.strip()}. The job may not have restarted."
                ),
            )
            failures += 1
        else:
            logger.info(f"watchdog: kickstart for {job.label} ok pid={cp.stdout.strip()}")
            kicked += 1
    logger.debug(f"watchdog: done; kicked={kicked} skipped={skipped} failures={failures}")
    return 0 if failures == 0 else 1


# Dead-man limit for one pass (flaw hunt 2026-10-01 A2). A normal pass takes
# seconds; the 10/1 pass hung for hours on an Alpaca call, and launchd will
# not start a second copy, so every later checkpoint was lost. Alpaca calls
# now time out on their own; this is the backstop for anything else.
PASS_DEADLINE_S = 600


def _on_deadman(signum, frame):
    raise TimeoutError(
        f"watchdog pass exceeded {PASS_DEADLINE_S}s; exiting so the next checkpoint runs"
    )


def main() -> int:
    import signal

    signal.signal(signal.SIGALRM, _on_deadman)
    signal.alarm(PASS_DEADLINE_S)
    try:
        return check()
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    sys.exit(main())
