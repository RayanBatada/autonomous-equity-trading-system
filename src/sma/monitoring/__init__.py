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
from pathlib import Path

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

def check_model_staleness(
    *,
    asof: date,
    models_dir=None,
    notify_fn=notify_failure,
) -> bool:
    """Page when the most recent SCHEDULED retrain did not run (its sentinel is
    missing), or when no model artifact exists at all. Returns True when stale.

    Sentinel-based, not artifact-age-based (review 2026-07-20 #5/#6): a retrain
    whose artifact the deploy gate quarantined is a DESIGNED state (the incumbent
    keeps serving) and must not page "retrain missed"; conversely a genuinely
    missed Monday retrain must page THAT evening, not two days later when an age
    threshold finally trips. Runs from the 22:30 monitoring job, so a miss nags
    every evening until fixed (2026-07-13: one watchdog page, then a week of
    silence while the model aged). Never raises.
    """
    from datetime import timedelta
    from pathlib import Path

    from sma import schedule as sched
    from sma.model.persistence import latest_model_for_date
    from sma.sentinels import read_sentinel

    if models_dir is None:
        models_dir = Path("models_artifacts")
    try:
        # Most recent scheduled retrain day on/before asof (weekly cadence →
        # always within the past 7 days).
        last_scheduled = None
        for back in range(8):
            d = asof - timedelta(days=back)
            if sched.runs_today("com.sma.model.retrain.weekly", asof=d):
                last_scheduled = d
                break
        if last_scheduled is not None and read_sentinel(
            label="com.sma.model.retrain.weekly", asof=last_scheduled
        ) is None:
            from sma.sched_adapter import get_adapter

            rekick = get_adapter().rekick_hint("com.sma.model.retrain.weekly")
            notify_fn(
                title="sma: model STALE (retrain missed)",
                message=(
                    f"The {last_scheduled.isoformat()} retrain has no sentinel as "
                    f"of {asof.isoformat()} — the model is serving stale weights. "
                    f"Likely the Mac was asleep at Mon 04:00. Run `{rekick}` before "
                    "15:00 ET, or leave the Mac awake next Monday."
                ),
            )
            logger.warning(
                f"monitoring: model STALE — no retrain sentinel for {last_scheduled}"
            )
            return True

        # Backstop: a sentinel without ANY serving artifact is still a page.
        latest_model_for_date(Path(models_dir), asof)
    except FileNotFoundError:
        notify_fn(
            title="sma: model STALE (no artifact found)",
            message=(
                f"No serving model artifact found in {models_dir} as of "
                f"{asof.isoformat()} — predict has nothing to load. Check the "
                "Monday retrain + models_artifacts/."
            ),
        )
        return True
    except Exception as e:  # noqa: BLE001 — never break monitoring on this check
        logger.warning(f"monitoring: model-staleness check errored: {e!r}")
        return False
    logger.info("monitoring: retrain sentinel + serving artifact present")
    return False


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


# ---------------------------------------------------------------------------
# Regime-turn crossing detector (2026-08-25)
#
# The dashboard's Model tab (dashboard/tabs/model.py, commit 3bd5f87) already
# computes a rolling live cross-sectional rank-IC and a trailing-21-decide-
# date "regime read" (mean + t-stat, |t|>=2 significance bar). That
# computation was extracted to sma.eval.live_ic so this module can reuse it
# directly -- check_regime_turn below calls the SAME trailing_ic_regime /
# model_edge_ic_df functions the dashboard renders, it does not reimplement
# the math.
#
# Unlike check_critical_jobs_fired (pages every trading day a job is
# missing) and check_model_staleness (nags every evening until fixed), this
# check must be SILENT most days -- "still positive" is not news. It only
# pages on a CROSSING: the regime level flipping into "positive" or
# "negative" significance. Last-known level is persisted in
# sma.monitoring.regime_state (one small file, overwritten every run).
# ---------------------------------------------------------------------------

REGIME_IC_HORIZON_DAYS = 10
REGIME_IC_WINDOW = 21
_SIGNIFICANT_REGIME_LEVELS = frozenset({"positive", "negative"})


def _load_live_ic_series(
    *,
    horizon_days: int,
    db_path: Path | None,
    universe_path: Path | None,
):
    """Live trailing IC series at `horizon_days`, wired straight to
    sma.eval.live_ic.model_edge_ic_df -- the exact same computation the
    dashboard's Model Edge chart renders. Returns None when there isn't
    enough live history yet (empty result), never an empty/NaN-filled
    series."""
    from sma.eval import live_ic

    resolved_db = db_path if db_path is not None else live_ic.DEFAULT_DB_PATH
    resolved_universe = (
        universe_path if universe_path is not None else live_ic.DEFAULT_UNIVERSE_PATH
    )
    ic_df = live_ic.model_edge_ic_df(
        horizon_days, db_path=resolved_db, universe_path=resolved_universe
    )
    if ic_df.empty:
        return None
    return ic_df.sort_values("asof_date").set_index("asof_date")["ic"]


def check_regime_turn(
    *,
    asof: date,
    notify_fn=notify_failure,
    ic_series=None,
    horizon_days: int = REGIME_IC_HORIZON_DAYS,
    window: int = REGIME_IC_WINDOW,
    db_path: Path | None = None,
    universe_path: Path | None = None,
) -> str | None:
    """Trailing-`window`-decide-date mean IC + t-stat at `horizon_days`
    (default 10d) -- reuses sma.eval.live_ic.trailing_ic_regime, the SAME
    function the dashboard's Model tab uses for its regime read.

    Fires a notification only on a CROSSING: the regime level (per
    trailing_ic_regime's |t|>=2 significance bar) moving INTO "positive"
    from anything else, or INTO "negative" from anything else. A steady
    "still positive" day, or a wobble into "neutral"/"insufficient", is
    silent -- paging every day the sign happens to still be positive would
    be noise, not signal. Last-known level is persisted via
    sma.monitoring.regime_state (one small file, overwritten every run --
    not a new dated sentinel per day).

    `ic_series`: injectable for tests (a synthetic trailing IC series --
    only order and values matter, index can be anything orderable). When
    None (the production default), computed live from the DB via
    _load_live_ic_series -> sma.eval.live_ic.model_edge_ic_df.

    Returns "positive"/"negative" when a crossing fired this call, else
    None (no signal yet, or a non-crossing/steady day). Never raises --
    matches this module's other checks (check_critical_jobs_fired,
    check_model_staleness): a broken regime check must not break the rest
    of the evening monitoring run.
    """
    from sma.eval.live_ic import trailing_ic_regime
    from sma.monitoring.regime_state import read_regime_state, write_regime_state

    try:
        series = (
            ic_series
            if ic_series is not None
            else _load_live_ic_series(
                horizon_days=horizon_days, db_path=db_path, universe_path=universe_path
            )
        )
        if series is None or series.dropna().empty:
            logger.info(
                "monitoring: regime-turn check has no live IC series yet; skipping"
            )
            return None

        regime = trailing_ic_regime(series, window=window)
        level = regime["level"]
        prior = read_regime_state()
        prior_level = prior.get("level") if prior else None

        fired: str | None = None
        if level in _SIGNIFICANT_REGIME_LEVELS and level != prior_level:
            fired = level
            notify_fn(
                title="sma: regime signal",
                message=(
                    f"Regime signal: trailing IC turned {level} "
                    f"(IC={regime['mean']:.2f}, t={regime['t_stat']:.1f})"
                ),
            )
            logger.warning(
                f"monitoring: regime-turn CROSSING -> {level} "
                f"(IC={regime['mean']:.4f}, t={regime['t_stat']:.2f}, "
                f"n={regime['n']}, prior={prior_level})"
            )
        else:
            logger.info(
                f"monitoring: regime-turn steady at level={level} (prior={prior_level})"
            )

        write_regime_state(
            {
                "level": level,
                "asof": asof.isoformat(),
                "mean": regime["mean"],
                "t_stat": regime["t_stat"],
                "n": regime["n"],
            }
        )
        return fired
    except Exception as e:  # noqa: BLE001 -- never break monitoring on this check
        logger.warning(f"monitoring: regime-turn check errored: {e!r}")
        return None
