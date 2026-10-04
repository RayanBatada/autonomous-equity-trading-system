"""Single source of truth for job schedules. Read by:
- ops/launchd/render_plists.py to render plists
- src/sma/live/preflight.py to compute waiting deadlines
- src/sma/watchdog.py to detect missed jobs
- ops/launchd/install.sh --check to verify drift
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from enum import IntEnum
from zoneinfo import ZoneInfo

NY_TZ = ZoneInfo("America/New_York")


class Day(IntEnum):
    """1=Mon ... 7=Sun (matches Python's isoweekday(); MON-FRI coincide with launchd Weekday 1-5).

    NOTE: launchd's StartCalendarInterval Weekday uses 0=Sunday, 1=Monday, ..., 6=Saturday.
    For Mon-Sat the integer values overlap, but Day.SUN = 7 does NOT match launchd's Weekday=0.
    Renderers (ops/launchd/render_plists.py) must convert Day.SUN to 0 when emitting plists.
    """

    MON = 1
    TUE = 2
    WED = 3
    THU = 4
    FRI = 5
    SAT = 6
    SUN = 7


@dataclass(frozen=True)
class JobSchedule:
    label: str
    days: tuple[Day, ...]
    fire_time_et: time
    wake_lead_minutes: int
    deadline_offset_minutes: int
    depends_on: tuple[str, ...]
    requires_writer_lock: bool
    waivers: frozenset[str]
    # Advisory (non-blocking) dependencies: checked once and logged, but a
    # not-ready advisory dep never aborts the consumer. Use for overlay signals
    # (e.g. agents theses) that inform but must not gate trading.
    advisory_deps: frozenset[str] = frozenset()
    # When set, this job's sentinel records `ingest_run_id` lineage from the
    # named upstream; the watchdog re-kicks the job if its sentinel was built
    # from an OLDER upstream run than the upstream sentinel currently records
    # (2026-06-09: stale predictions survived an ingest heal).
    lineage_upstream: str | None = None
    # When set, the watchdog checks THIS label for today instead of `label` —
    # for jobs whose primary sentinel is keyed by a different date (reconcile
    # keys its batch sentinel by the decide-date it reconciled, never today).
    liveness_sentinel_label: str | None = None
    # Whether this job depends on a live market session. Market-sensitive jobs
    # (ingest/predict/agents/decide/stop-loss/reconcile) produce no useful work
    # on NYSE holidays, so the watchdog skips them on non-trading days. Jobs that
    # run regardless of the market (retrain/autoresearch/backup/senate/house) set
    # this False so a miss on a holiday Monday / weekend is still kicked + paged
    # rather than silently suppressed by the holiday guard (review 2026-07-04).
    requires_market_data: bool = True
    # How long past the deadline the watchdog may still kick this job. The flat
    # 6h default protects the evening pipeline from writer-lock collisions, but
    # it stranded heavy market-independent jobs for a whole WEEK when the Mac
    # slept through their early-morning slots (retrain missed entirely Mon
    # 2026-07-13; the machine woke midday but 6h had passed). Widen per job
    # where a later kick is still safe — bounded so the run completes clear of
    # the 18:30 ET ingest / evening writer-lock traffic, INCLUDING when it
    # serializes behind another late-kicked heavy job (review 2026-07-20 #3).
    late_kick_max_hours: float = 6.0
    # When True, the watchdog kicks this job only if every depends_on upstream
    # has today's sentinel. Autoresearch enforces no dependency itself: kicked
    # alongside retrain it races the heavy lock, searches against week-old
    # weights, and its IC-gated winner loses to the later retrain artifact via
    # the created-at tie-break (the 2026-06-19 bug, reintroducible by kick
    # ordering — review 2026-07-20 #4).
    kick_requires_upstream_sentinels: bool = False
    # When True, a sentinel whose quality.passed is false does NOT count as
    # done: the watchdog re-kicks the job (past deadline, idle, inside the
    # late-kick window). Ingest only: its CLI already re-runs on a failed
    # sentinel, and a dead network at 18:30 that is back by 21:00 is the
    # commonest lost night (flaw hunt 2026-10-01 A1: 4 of 6 lost nights since
    # 9/9). Not for agents: a re-run cannot fix exhausted API credit and it
    # would hold the writer lock across decide.
    rekick_on_failed_quality: bool = False

    def __post_init__(self) -> None:
        if not self.days:
            raise ValueError(f"JobSchedule {self.label!r} must have at least one Day")


SCHEDULE: tuple[JobSchedule, ...] = (
    JobSchedule(
        label="com.sma.ingest.daily",
        days=(Day.MON, Day.TUE, Day.WED, Day.THU, Day.FRI),
        fire_time_et=time(18, 30),
        wake_lead_minutes=15,
        deadline_offset_minutes=120,
        depends_on=(),
        requires_writer_lock=True,
        waivers=frozenset({"newsapi_rate_limit", "theses_freshness"}),
        rekick_on_failed_quality=True,
    ),
    JobSchedule(
        label="com.sma.model.predict.daily",
        days=(Day.MON, Day.TUE, Day.WED, Day.THU, Day.FRI),
        fire_time_et=time(19, 30),
        wake_lead_minutes=0,
        deadline_offset_minutes=15,
        depends_on=("com.sma.ingest.daily",),
        requires_writer_lock=True,
        lineage_upstream="com.sma.ingest.daily",
        # Same overlay waivers as decide: predict needs PRICES; news/earnings/
        # theses/divergence failures must not stop fresh predictions (the
        # 2026-06-09 ingest gate would otherwise block predict on overlay-only
        # bad nights that decide itself would waive through).
        waivers=frozenset({
            "theses_freshness",
            "news_per_ticker_minimum",
            "earnings_coverage",
            "no_cross_source_price_divergence",
            # no_dead_or_frozen_tickers (2026-08-31) is blocking=False in code
            # already (a single dead name must never freeze the whole book —
            # see the EA/AVB corporate-action gap this check exists to close),
            # so this waiver is belt-and-suspenders: "code and config agree"
            # per the same Codex follow-up that added the divergence waiver
            # above.
            "no_dead_or_frozen_tickers",
        }),
    ),
    JobSchedule(
        label="com.sma.agents.daily",
        days=(Day.MON, Day.TUE, Day.WED, Day.THU, Day.FRI),
        fire_time_et=time(19, 45),
        wake_lead_minutes=0,
        deadline_offset_minutes=30,
        depends_on=("com.sma.model.predict.daily",),
        requires_writer_lock=True,
        waivers=frozenset({"budget_exhausted"}),
    ),
    JobSchedule(
        label="com.sma.live.decide.daily",
        days=(Day.MON, Day.TUE, Day.WED, Day.THU, Day.FRI),
        fire_time_et=time(20, 0),
        wake_lead_minutes=0,
        deadline_offset_minutes=60,
        depends_on=(
            "com.sma.ingest.daily",
            "com.sma.model.predict.daily",
        ),
        # Agents is an ADVISORY overlay, not a hard gate. On 2026-06-04 a wifi
        # outage made every agent call fail (all_tickers_failed); as a hard
        # dependency that froze ALL trading. The model predictions + risk rails
        # are the real trade drivers — a stale/failed thesis overlay must not
        # block them. decide reads whatever theses exist via --use-theses.
        advisory_deps=frozenset({"com.sma.agents.daily"}),
        requires_writer_lock=True,
        # 2026-05-14: universe expansion 81→146 added tickers whose news and
        # earnings haven't been backfilled yet. The first two below blocked
        # preflight on 5/12-5/14. Waiving them restored trading.
        # 2026-05-18: no_cross_source_price_divergence blocked decide on 5/15
        # — one or more of the new tickers had >1% price disagreement between
        # yfinance and alpaca (likely a stale row for an illiquid name). The
        # check is informational rather than money-critical, so waive it for
        # decide; ingest still surfaces it. Re-evaluate after the ingest
        # source-priority dedupe is more aggressive about stale rows.
        waivers=frozenset({
            "theses_freshness",
            "news_per_ticker_minimum",
            "earnings_coverage",
            "no_cross_source_price_divergence",
            # Belt-and-suspenders alongside blocking=False in code (see the
            # predict job's waiver above for the full rationale).
            "no_dead_or_frozen_tickers",
        }),
    ),
    JobSchedule(
        label="com.sma.live.stop-loss.weekday",
        days=(Day.MON, Day.TUE, Day.WED, Day.THU, Day.FRI),
        fire_time_et=time(9, 25),
        wake_lead_minutes=15,
        deadline_offset_minutes=5,
        depends_on=(),
        requires_writer_lock=True,
        waivers=frozenset(),
    ),
    JobSchedule(
        label="com.sma.live.reconcile.daily",
        days=(Day.MON, Day.TUE, Day.WED, Day.THU, Day.FRI),
        fire_time_et=time(16, 30),
        wake_lead_minutes=0,
        deadline_offset_minutes=15,
        depends_on=(),
        requires_writer_lock=True,
        waivers=frozenset(),
        # reconcile's batch sentinel is keyed by the decide-date it reconciled
        # (yesterday), NEVER today (phantom-sentinel protection). The watchdog
        # therefore checks this run-date liveness sentinel, written by the CLI
        # on every run — without it, reconcile got pointlessly re-kicked every
        # watchdog hour, every day.
        liveness_sentinel_label="com.sma.live.reconcile.daily.ran",
    ),
    JobSchedule(
        # 2026-05-18: moved Sat 02:00 ET → Mon 04:00 ET. Saturdays are the
        # weekend laptop-off window; missed retrains stranded the model on
        # stale data. Mon 04:00 ET runs before Mon 18:30 ET ingest, so the
        # 19:30 ET predict picks up the fresh artifact on the same day.
        label="com.sma.model.retrain.weekly",
        days=(Day.MON,),
        fire_time_et=time(4, 0),
        wake_lead_minutes=15,
        deadline_offset_minutes=180,
        depends_on=(),
        requires_writer_lock=True,
        waivers=frozenset(),
        requires_market_data=False,  # trains on stored history; runs on holidays
        # Kickable until 15:00 ET (deadline 07:00 + 8h): a ~2h retrain kicked at
        # the last moment finishes ~17:00, clear of the 18:30 ingest. A missed
        # retrain otherwise strands the model on week-old data (2026-07-13).
        late_kick_max_hours=8.0,
    ),
    JobSchedule(
        label="com.sma.backup.daily",
        days=(Day.MON, Day.TUE, Day.WED, Day.THU, Day.FRI, Day.SAT, Day.SUN),
        fire_time_et=time(22, 0),
        wake_lead_minutes=15,
        deadline_offset_minutes=30,
        depends_on=(),
        requires_writer_lock=True,
        waivers=frozenset(),
        requires_market_data=False,  # runs 7 days; a missed backup must page
    ),
    JobSchedule(
        # 2026-05-18: moved Sun 03:00 ET → Mon 07:00 ET. Sunday's laptop-off
        # weekend window made the Sun 03:00 fire miss its first scheduled run.
        # Mon 07:00 ET is after the Mon 04:00 retrain (so autoresearch evals
        # against the freshly-trained model) and before Mon 18:30 ET ingest
        # (so a long autoresearch run can't bleed into the trading-day
        # pipeline). Long deadline: the config search (CV-IC over ~6 configs ×
        # 5-fold walk-forward on ~460k rows) runs ~1.5-2h (measured 2026-06-19).
        label="com.sma.autoresearch.nightly",
        days=(Day.MON,),
        fire_time_et=time(7, 0),
        wake_lead_minutes=15,
        deadline_offset_minutes=180,
        depends_on=("com.sma.model.retrain.weekly",),
        requires_writer_lock=True,
        waivers=frozenset(),
        requires_market_data=False,  # config search on stored data; holiday-safe
        # Kickable until 13:00 ET (deadline 10:00 + 3h) — sized for the
        # SERIALIZED worst case: kicked at 13:00 behind a 13:00-kicked ~2h
        # retrain, the ~2h search runs 15:00-17:00, still clear of the 18:30
        # ingest. (5h looked safe per-job but the two jobs share the heavy
        # lock; review 2026-07-20 #3.)
        late_kick_max_hours=3.0,
        # Never kick before today's retrain sentinel exists (see field docs).
        kick_requires_upstream_sentinels=True,
    ),
    JobSchedule(
        # 2026-05-24: post-pipeline notifier. Fires after the watchdog's
        # 22:00 ET cycle would have caught up any 20:00 ET decide that
        # missed its slot. If decide STILL has no sentinel for today, alerts
        # via macOS notification. Empirically Fri 5/22's preflight failure
        # left decide silently un-fired for 3 days until manual audit.
        label="com.sma.monitoring.daily",
        days=(Day.MON, Day.TUE, Day.WED, Day.THU, Day.FRI),
        fire_time_et=time(22, 30),
        wake_lead_minutes=0,
        deadline_offset_minutes=15,
        depends_on=(),
        requires_writer_lock=False,
        waivers=frozenset(),
    ),
    JobSchedule(
        # 2026-05-24: weekly Senate PTR refresh. Fires Sunday morning before
        # the Mon 04:00 retrain so the politician_flow_30d feature
        # incorporates the past week's new Senate disclosures (House is
        # backfilled separately). Default 30-day lookback catches stragglers
        # that were filed late; idempotent via already_parsed(doc_id).
        # Sunday 10:00 ET picked to (a) be off-market-hours, (b) give Akamai
        # rate-limit headroom on a low-traffic day, (c) finish well before
        # the Mon 04:00 retrain and Mon 07:00 autoresearch.
        label="com.sma.senate-ingest.weekly",
        days=(Day.SUN,),
        fire_time_et=time(10, 0),
        wake_lead_minutes=15,
        deadline_offset_minutes=120,
        depends_on=(),
        requires_writer_lock=True,
        waivers=frozenset(),
        requires_market_data=False,  # Sunday disclosure refresh; never a market day
        # Sundays have no market pipeline — only the 22:00 backup contends for the
        # writer lock. Kickable until 20:00 ET (deadline 12:00 + 8h); the scrape
        # typically runs well under an hour, finishing before the 22:00 backup
        # even in the worst kick slot (and the 23:00 watchdog re-kicks the backup
        # if it ever loses a lock race). Missed 7/12 + 7/19: the politician
        # features went a week stale.
        late_kick_max_hours=8.0,
    ),
    JobSchedule(
        # 2026-05-24: weekly House PTR refresh. Symmetric to Senate. House
        # has a bulk FD.zip with the full year's index, so a weekly run
        # is cheap (small download + already_parsed dedup against
        # politician_disclosure_docs makes re-runs idempotent). Sunday
        # 11:00 ET — 1 hour after Senate to avoid concurrent writer_lock
        # contention and to give the disclosures-clerk.house.gov server
        # rate-limit headroom on a low-traffic day.
        label="com.sma.house-ingest.weekly",
        days=(Day.SUN,),
        fire_time_et=time(11, 0),
        wake_lead_minutes=0,
        deadline_offset_minutes=120,
        depends_on=(),
        requires_writer_lock=True,
        waivers=frozenset(),
        requires_market_data=False,  # Sunday disclosure refresh; never a market day
        # Deadline 13:00 + 7h → kickable until 20:00 ET, same slot as senate; the
        # cheap bulk-FD refresh finishes well before the 22:00 backup's lock.
        late_kick_max_hours=7.0,
    ),
    JobSchedule(
        # 2026-08-29: weekly "week in review" digest -- P&L, trading activity,
        # live signal (IC), and ops health for the past week, so it reaches
        # Rayan without him asking (see sma/monitoring/weekly_digest.py).
        # Sunday 18:00 ET: after both weekly disclosure ingests (10:00/11:00)
        # so its ops section can see today's senate/house sentinels, and well
        # clear of the 22:00 backup. Never contends for the writer lock --
        # everything it reads (account_snapshots, paper_fills,
        # intended_orders, predictions, prices, plus sentinels and model
        # artifacts) is read-only, and its only writes are a markdown file
        # under ~/StockMarket/weekly/ and its own completion sentinel.
        label="com.sma.weekly-digest.weekly",
        days=(Day.SUN,),
        fire_time_et=time(18, 0),
        wake_lead_minutes=15,
        deadline_offset_minutes=15,
        depends_on=(),
        requires_writer_lock=False,
        waivers=frozenset(),
        requires_market_data=False,  # Sunday review; never a market day
        # Same Sunday-evening wake-risk margin as senate/house (8h/7h):
        # kickable until ~02:15 ET Monday, comfortably clear of the 04:00
        # retrain.
        late_kick_max_hours=8.0,
    ),
)


# OPTIONAL jobs (2026-09-26, strategy-expansion.md section 3): rendered to
# plists/units like SCHEDULE, but NOT installed by ops/launchd/install.sh and
# NOT in SCHEDULE, so nothing that iterates SCHEDULE (weekly digest, the
# render counts, preflight) sees them. The watchdog evaluates one ONLY when the
# OS scheduler reports it installed (SchedulerAdapter.installed), so a job
# that ships disabled can never page. Enable per ops/launchd/README.md.
_WEEKDAYS = (Day.MON, Day.TUE, Day.WED, Day.THU, Day.FRI)
OPTIONAL_SCHEDULE: tuple[JobSchedule, ...] = (
    JobSchedule(
        # 1-minute IEX bars through ~15:40 for the close session. Idempotent,
        # seconds of work; a later rerun of the day just completes the bars.
        label="com.sma.ingest.intraday",
        days=_WEEKDAYS,
        fire_time_et=time(15, 41),
        wake_lead_minutes=5,
        deadline_offset_minutes=60,
        depends_on=(),
        requires_writer_lock=True,
        waivers=frozenset(),
        # Deadline 16:41; kickable until 18:11, clear of the 18:30 ingest's
        # writer lock. In practice the next watchdog checkpoint (19:00) is past
        # that, so a missed run pages rather than being kicked.
        late_kick_max_hours=1.5,
    ),
    JobSchedule(
        # Midday session: window 10:30-11:00 ET (live.sessions.midday.window).
        label="com.sma.live.session.midday",
        days=_WEEKDAYS,
        fire_time_et=time(10, 35),
        wake_lead_minutes=5,
        deadline_offset_minutes=25,
        depends_on=(),
        requires_writer_lock=True,
        waivers=frozenset(),
        # Never late-kicked: outside its window the job only skips, so a kick
        # buys nothing. A miss (no sentinel) pages once it is past deadline.
        late_kick_max_hours=0.0,
    ),
    JobSchedule(
        # Close session: window 15:40-15:55 ET (live.sessions.close.window).
        label="com.sma.live.session.close",
        days=_WEEKDAYS,
        fire_time_et=time(15, 45),
        wake_lead_minutes=5,
        deadline_offset_minutes=10,
        depends_on=(),
        requires_writer_lock=True,
        waivers=frozenset(),
        late_kick_max_hours=0.0,
    ),
)


def get(label: str) -> JobSchedule:
    for j in (*SCHEDULE, *OPTIONAL_SCHEDULE):
        if j.label == label:
            return j
    raise KeyError(f"unknown job label: {label}")


def runs_today(label: str, *, asof: date) -> bool:
    j = get(label)
    return Day(asof.isoweekday()) in j.days


def deadline(label: str, *, asof: date) -> datetime:
    j = get(label)
    if not runs_today(label, asof=asof):
        raise ValueError(
            f"{label} does not run on {asof.isoformat()} (weekday={asof.isoweekday()})"
        )
    fire_dt = datetime.combine(asof, j.fire_time_et, tzinfo=NY_TZ)
    return fire_dt + timedelta(minutes=j.deadline_offset_minutes)


def next_fire(label: str, *, after: datetime) -> datetime:
    j = get(label)
    if after.tzinfo is None:
        # Naive input: treat as ET (the schedule's home timezone). Avoids
        # astimezone() on a naive datetime, which Python treats as local
        # system time and so breaks on UTC machines like CI.
        after = after.replace(tzinfo=NY_TZ)
    after_et = after.astimezone(NY_TZ)
    cursor = after_et
    for _ in range(14):  # search up to 2 weeks
        d = cursor.date()
        if Day(d.isoweekday()) in j.days:
            fire_dt = datetime.combine(d, j.fire_time_et, tzinfo=NY_TZ)
            if fire_dt > after_et:
                return fire_dt
        cursor = (cursor + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    raise RuntimeError(f"no upcoming fire found for {label} within 14 days")
