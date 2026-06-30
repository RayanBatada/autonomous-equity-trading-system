"""Periodic watchdog: detect missed scheduled jobs and kickstart them.

Runs on a launchd schedule (every hour 19-22 ET) plus at user login.
For each job in `sma.schedule.SCHEDULE` that runs today, check whether
its sentinel exists by deadline. If past deadline AND no sentinel,
query launchd state first: only kickstart if the service is idle
("not running" or unknown). If the service is already running or
waiting (launchd queued it), skip without kickstarting.

Decision logic per job:
1. Skip if sentinel exists for today.
2. Skip if now < deadline.
3. Skip if launchd state is "running" or "waiting" (log + increment skipped).
4. Else kickstart -p; increment kicked on rc=0, failures on rc!=0.
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from loguru import logger

from sma import schedule as sched
from sma.ingest.notify import notify_failure
from sma.readiness import sentinel_lineage_stale
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


def _now() -> datetime:
    return datetime.now(tz=sched.NY_TZ)


def _launchctl_state(label: str) -> str:
    """Return the launchd state of `label`: 'running', 'waiting', 'not running', or 'unknown'."""
    target = f"gui/{os.getuid()}/{label}"
    cp = subprocess.run(
        ["launchctl", "print", target],
        capture_output=True,
        text=True,
    )
    if cp.returncode != 0:
        return "unknown"
    for line in cp.stdout.splitlines():
        line = line.strip()
        if line.startswith("state ="):
            return line.split("=", 1)[1].strip()  # "running", "waiting", "not running"
    return "unknown"


def check(*, notify_fn=notify_failure) -> int:
    """Returns 0 if all good, 1 if any kickstart failed."""
    now = _now()
    today = now.date()
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

    # Holiday guard: jobs that require market data (ingest, predict, agents,
    # decide, stop-loss, reconcile) are expected NOT to produce sentinels on
    # NYSE holidays. Kicking them on a holiday would cause noise (failed
    # preflight) and waste lock time. The backup/monitoring/autoresearch jobs
    # run regardless of trading day, so we skip only per-job; but it is
    # simpler and correct to skip the entire watchdog on non-trading weekdays
    # since all market-sensitive jobs have no useful work to do.
    if not _is_trading_day(today, notify_fn=notify_fn):
        logger.info(
            "watchdog: {} is not a trading day; skipping all kickstart checks",
            today,
        )
        return 0

    # 2026-05-18: don't re-kick jobs that are MORE than 6 hours past
    # deadline. If a job has been missing all day, kicking it now risks
    # colliding with later-scheduled jobs on writer_lock (decide.daily,
    # backup.daily). Wait for next natural fire instead.
    late_kick_max_hours = 6
    for job in sched.SCHEDULE:
        if not sched.runs_today(job.label, asof=today):
            continue
        deadline = sched.deadline(job.label, asof=today)
        if now < deadline:
            continue
        late_by_h = (now - deadline).total_seconds() / 3600
        if late_by_h > late_kick_max_hours:
            logger.warning(
                f"watchdog: {job.label} {late_by_h:.1f}h past deadline; "
                f"too late to safely kick (max {late_kick_max_hours}h); "
                "skipping until next natural fire"
            )
            # Don't skip SILENTLY — a >6h-late job is a likely-missed run and
            # must reach a human (the freeze went unnoticed for a week).
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
        # Liveness label: reconcile's batch sentinel is keyed by the decide-
        # date it reconciled (yesterday), never today — check its run-date
        # liveness sentinel instead (pre-fix: pointless re-kicks every hour).
        check_label = job.liveness_sentinel_label or job.label
        sentinel = read_sentinel(label=check_label, asof=today)
        if sentinel is not None:
            # Lineage: a sentinel built from an OLDER upstream run than the
            # upstream currently records is STALE — re-kick so the healed
            # upstream produces fresh output (2026-06-09: predict kept
            # stale-feature predictions after the ingest heal; decide's
            # preflight WAITs on lineage, so this kick un-blocks the night).
            if job.lineage_upstream is not None:
                upstream_sentinel = read_sentinel(
                    label=job.lineage_upstream, asof=today
                )
                if sentinel_lineage_stale(
                    consumer=sentinel, upstream=upstream_sentinel
                ):
                    logger.warning(
                        f"watchdog: {job.label} sentinel lineage is stale "
                        f"(consumed {sentinel.get('ingest_run_id')}, current "
                        f"{(upstream_sentinel or {}).get('run_id')}); re-kicking"
                    )
                else:
                    continue
            else:
                continue
        state = _launchctl_state(job.label)
        if state in ("running", "waiting"):
            logger.info(
                f"watchdog: {job.label} past deadline but launchd"
                f" state={state!r}; skipping kickstart"
            )
            skipped += 1
            continue
        target = f"gui/{os.getuid()}/{job.label}"
        logger.warning(
            f"watchdog: {job.label} past deadline ({deadline});"
            f" state={state!r}; kickstarting {target}"
        )
        cp = subprocess.run(
            ["launchctl", "kickstart", "-p", target],
            capture_output=True,
            text=True,
        )
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


if __name__ == "__main__":
    sys.exit(check())
