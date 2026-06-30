"""Sentinel-based preflight for the decide job. Does NOT open DuckDB.

For each upstream job in `schedule.com.sma.live.decide.daily.depends_on`,
polls the sentinel until READY or until deadline. Then calls Alpaca's
get_next_session() to verify tomorrow is a trading day.

Replaces the legacy preflight that queried `ingest_log` via a read-only
DuckDB connection (the source of the writer-blocks-self deadlock that
codex round-1 surfaced 2026-04-30).
"""

from __future__ import annotations

import time as time_mod
from datetime import date, datetime, timedelta
from pathlib import Path

from loguru import logger

from sma import schedule as sched
from sma.live.retry import _retry_transient
from sma.readiness import (
    ReadinessResult,
    ReadinessState,
    live_readiness,
    sentinel_lineage_stale,
)
from sma.sentinels import read_sentinel


class PreflightError(Exception):
    pass


class PreflightAbortError(PreflightError):
    pass


# Legacy alias used in tests and old callers.
PreflightAbort = PreflightAbortError


class IngestNotCompleteError(PreflightError):
    pass


class UpstreamMissedDeadline(PreflightError):  # noqa: N818
    pass


class UpstreamReadinessFailed(PreflightError):  # noqa: N818
    pass


class PreflightHolidaySkipped(PreflightError):  # noqa: N818
    """Ingest marked the day as a market holiday — decide must skip QUIETLY
    (exit 0, no sentinel, no page). Launchd fires on weekday holidays; before
    this, the passed=True/run_id=None holiday sentinel sailed through
    readiness and decide would submit orders on a closed day (Codex branch
    review, 2026-06-09)."""


class NextSessionNotTomorrow(PreflightError):  # noqa: N818
    pass


# Alias kept for backward compat.
CalendarHolidayError = NextSessionNotTomorrow


def _next_session_with_retry(alpaca, asof: date) -> date:
    """Read-only Alpaca calendar lookup, retried through a transient DNS/network
    blip. decide-preflight runs shortly after the laptop wakes; a single
    unresolved DNS on the first attempt must not fail-close the whole decide
    (2026-06-24 audit; same failure mode as the 2026-06-09 stop-loss DNS kill).
    """
    return _retry_transient(
        lambda: alpaca.next_session_date(today=asof),
        label="preflight next_session",
    )


def run_preflight(
    *,
    asof: date,
    db_path: Path | None = None,
    alpaca,
    max_wait_s: float = 7200,
    poll_interval_s: float = 30,
    now_fn=None,
    sleep_fn_poll=None,
    # Legacy params: accepted but ignored so old call sites still compile.
    store=None,
    ingest_retry_seconds: int = 60,
    ingest_max_retries: int = 20,
    sleep_fn=None,
) -> None:
    """Block until all upstream sentinels are READY or raise. No DB access.

    Parameters
    ----------
    asof:
        Date for which decide is about to run.
    db_path:
        Accepted for API stability; not used. Preflight reads only sentinels.
    alpaca:
        Alpaca client used for the calendar check (get_next_session()).
    max_wait_s:
        Maximum total seconds to wait for upstream sentinels. Pass 0 for
        test-mode: raises immediately on any missing/non-ready sentinel.
    poll_interval_s:
        Seconds between readiness polls (ignored when max_wait_s=0).
    store, ingest_retry_seconds, ingest_max_retries, sleep_fn:
        Deprecated; accepted and ignored for backward compatibility with
        integration tests written against the legacy DuckDB-based preflight.
    """
    _now = now_fn or (lambda: datetime.now(tz=sched.NY_TZ))
    _sleep = sleep_fn_poll or time_mod.sleep

    decide_job = sched.get("com.sma.live.decide.daily")
    decide_deadline = sched.deadline("com.sma.live.decide.daily", asof=asof)
    start_now = _now()

    # Absolute cutoff: past deadline+6h (symmetric with the watchdog's
    # too-late-to-kick rule) a decide start is an anomaly (RunAtLoad recovery
    # at 3am, manual run) — fail closed instead of submitting very-late
    # orders (Codex branch review, 2026-06-09). max_wait_s=0 is explicit
    # test-mode (historical asofs) and is exempt.
    if max_wait_s != 0 and start_now > decide_deadline + timedelta(hours=6):
        raise PreflightAbortError(
            f"decide for {asof.isoformat()} started at {start_now} — more than "
            f"6h past its {decide_deadline} deadline; too late to trade safely"
        )

    for upstream_label in decide_job.depends_on:
        upstream_deadline = sched.deadline(upstream_label, asof=asof)
        # Wait until the tighter of decide's deadline OR upstream's deadline + 15 min headroom.
        wait_until = min(decide_deadline, upstream_deadline + timedelta(minutes=15))
        # Late-start grace: a watchdog-kicked decide (e.g. 22:00 after a healed
        # ingest, deadline 21:00) must still poll briefly — the watchdog kicks a
        # stale predict in the SAME pass, and dying instantly on deadline math
        # would strand the night seconds before predict lands (2026-06-09
        # recovery hole). Also gives a seconds-late upstream a window on normal
        # nights instead of failing decide at 20:00:01.
        wait_until = max(wait_until, start_now + timedelta(minutes=20))

        while True:
            sentinel = read_sentinel(label=upstream_label, asof=asof)
            r = live_readiness(
                label=upstream_label,
                sentinel=sentinel,
                waivers=decide_job.waivers,
            )
            if (
                upstream_label == "com.sma.ingest.daily"
                and sentinel is not None
                and (
                    sentinel.get("holiday_skipped")
                    or (r.state == ReadinessState.READY and sentinel.get("run_id") is None)
                )
            ):
                # Holiday-skip sentinel is passed=True with run_id=None: READY
                # by quality but there is no trading session — skip the day
                # quietly rather than trading stale features on a closed market.
                    raise PreflightHolidaySkipped(
                        f"ingest marked {asof.isoformat()} as a market holiday "
                        "(holiday_skipped sentinel); no trading day"
                    )
            if (
                r.state == ReadinessState.READY
                and upstream_label == "com.sma.model.predict.daily"
            ):
                # Lineage: predictions must come from the CURRENT ingest run.
                # A mismatch is WAIT (not FAIL): the watchdog re-kicks predict
                # on stale lineage, so a fresh sentinel may land within the
                # poll window (2026-06-09: predict silently kept stale-feature
                # predictions after ingest healed).
                ingest_sentinel = read_sentinel(
                    label="com.sma.ingest.daily", asof=asof
                )
                if sentinel_lineage_stale(
                    consumer=sentinel, upstream=ingest_sentinel
                ):
                    r = ReadinessResult(
                        state=ReadinessState.WAIT,
                        explanation=(
                            f"predict sentinel lineage stale: consumed ingest run "
                            f"{sentinel.get('ingest_run_id')} but current ingest run is "
                            f"{(ingest_sentinel or {}).get('run_id')}; waiting for re-predict"
                        ),
                    )
            if r.state == ReadinessState.READY:
                logger.info(f"preflight: {upstream_label} READY ({r.explanation})")
                break
            if r.state == ReadinessState.FAIL:
                raise UpstreamReadinessFailed(r.explanation)
            # WAIT
            if max_wait_s == 0:
                # Test mode: raise immediately without sleeping.
                if upstream_label == "com.sma.ingest.daily":
                    raise IngestNotCompleteError(
                        f"ingest sentinel not ready for {asof.isoformat()} "
                        f"(test mode): {r.explanation}"
                    )
                raise UpstreamMissedDeadline(
                    f"{upstream_label} sentinel not ready for {asof.isoformat()} "
                    f"(test mode): {r.explanation}"
                )
            now = _now()
            if now >= wait_until:
                if upstream_label == "com.sma.ingest.daily":
                    raise IngestNotCompleteError(
                        f"ingest sentinel not ready by deadline {wait_until}: "
                        f"{r.explanation}"
                    )
                raise UpstreamMissedDeadline(
                    f"{upstream_label} sentinel not ready by deadline {wait_until}: "
                    f"{r.explanation}"
                )
            _sleep(poll_interval_s)

    # Advisory dependencies (e.g. agents theses): check once, never block. By
    # the time decide runs, an advisory upstream has had its full window; if it
    # is not READY we log and proceed rather than freezing trading on an overlay
    # signal (regression guard for the 2026-06-04 wifi-outage freeze).
    for advisory_label in decide_job.advisory_deps:
        try:
            sentinel = read_sentinel(label=advisory_label, asof=asof)
        except Exception as e:  # corrupt JSON / read error must not block trading
            logger.warning(
                f"preflight: {advisory_label} sentinel unreadable "
                f"(advisory — proceeding anyway): {e!r}"
            )
            continue
        r = live_readiness(
            label=advisory_label,
            sentinel=sentinel,
            waivers=decide_job.waivers,
        )
        if r.is_ready():
            logger.info(f"preflight: {advisory_label} READY (advisory) ({r.explanation})")
        else:
            logger.warning(
                f"preflight: {advisory_label} NOT READY (advisory — proceeding anyway): "
                f"{r.explanation}"
            )

    # Calendar check: next trading session must be in the near future. Pre-fix
    # this required `next_session == calendar tomorrow (weekday-adjusted)`,
    # which broke decide on every pre-holiday day. Empirical 2026-05-22:
    # Friday before Memorial Day, alpaca correctly returned next session =
    # Tue 5/26, the check expected Mon 5/25, decide refused to fire and the
    # bot skipped a full trading-day's signal. OPG orders queued at Alpaca
    # transparently wait for the next session — a 1- to 4-day pre-holiday gap
    # is normal. The 14-day bound still catches truly broken calendar data
    # (e.g., SDK returning a stale or wildly-future date).
    next_session_date_val = _next_session_with_retry(alpaca, asof)
    max_gap_days = 14
    if not (asof < next_session_date_val <= asof + timedelta(days=max_gap_days)):
        raise NextSessionNotTomorrow(
            f"alpaca next session is {next_session_date_val}, expected within "
            f"{max_gap_days} days strictly after {asof}"
        )
