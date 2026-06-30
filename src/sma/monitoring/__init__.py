"""Daily post-pipeline monitoring: alert when critical jobs didn't fire.

Empirical 2026-05-22: Fri before Memorial Day, decide.daily was blocked by a
preflight bug. No alert surfaced; the bot silently skipped a trading day's
signal until a manual audit ~3 days later. This module exists so a missed
fire becomes visible the same evening.

Scope: per-day post-mortem at 22:30 ET. Checks the sentinel of each label
in CRITICAL_LABELS for today (ET); calls notify_failure for each one
missing. Skips entirely on non-trading days (US market holidays) so we
don't false-positive on a quiet Memorial Day.
"""

from __future__ import annotations

from datetime import date

from loguru import logger

from sma.ingest.notify import notify_failure
from sma.sentinels import read_sentinel

# The minimum set of labels whose absence on a trading day indicates the
# pipeline broke. Decide is the load-bearing one — if it fired, every
# upstream dependency also fired (preflight reads ingest + predict + agents
# sentinels and refuses to proceed without them).
CRITICAL_LABELS: tuple[str, ...] = (
    "com.sma.live.decide.daily",
)


def check_critical_jobs_fired(
    *,
    asof: date,
    alpaca,
    notify_fn=notify_failure,
) -> list[str]:
    """Return the list of CRITICAL_LABELS missing a sentinel for `asof`.

    Calls `notify_fn(title, message)` once per missing label (so the user
    gets a macOS notification each evening that something broke).

    Skips silently if `asof` is not a trading day per Alpaca's calendar —
    on holidays, decide is expected to be blocked by preflight (quality
    check fails on a no-prices day).
    """
    try:
        sessions = alpaca.sessions_between(start=asof, end=asof)
        is_trading_day = len(sessions) > 0
    except Exception as e:  # calendar/API/network failure must not crash before alerting
        logger.warning(
            f"monitoring: calendar lookup failed for {asof.isoformat()} ({e!r}); "
            "assuming a trading day and checking sentinels anyway"
        )
        notify_fn(
            title="sma: monitoring degraded",
            message=(
                f"Could not determine market status for {asof.isoformat()} ({e!r}). "
                "Assuming a trading day and checking critical sentinels."
            ),
        )
        is_trading_day = True

    if not is_trading_day:
        logger.info(
            f"monitoring: {asof.isoformat()} is not a trading day; skipping alerts"
        )
        return []

    missing: list[str] = []
    for label in CRITICAL_LABELS:
        if read_sentinel(label=label, asof=asof) is None:
            missing.append(label)

    if not missing:
        logger.info(
            f"monitoring: all {len(CRITICAL_LABELS)} critical jobs fired for "
            f"{asof.isoformat()}"
        )
        return []

    for label in missing:
        notify_fn(
            title="sma: critical job missed",
            message=(
                f"{label} has no sentinel for {asof.isoformat()}. "
                "Trading day with no decide fire — check live.decide.err.log."
            ),
        )
        logger.warning(
            f"monitoring: MISSING {label} for {asof.isoformat()}"
        )
    return missing
