from datetime import date, datetime, time

import pytest

from sma.schedule import (
    NY_TZ,
    SCHEDULE,
    Day,
    JobSchedule,
    deadline,
    get,
    next_fire,
    runs_today,
)


def test_schedule_has_all_twelve_jobs():
    assert len(SCHEDULE) == 12
    labels = {j.label for j in SCHEDULE}
    assert labels == {
        "com.sma.ingest.daily",
        "com.sma.model.predict.daily",
        "com.sma.agents.daily",
        "com.sma.live.decide.daily",
        "com.sma.live.stop-loss.weekday",
        "com.sma.live.reconcile.daily",
        "com.sma.model.retrain.weekly",
        "com.sma.backup.daily",
        "com.sma.autoresearch.nightly",
        "com.sma.monitoring.daily",
        "com.sma.senate-ingest.weekly",
        "com.sma.house-ingest.weekly",
    }


def test_autoresearch_runs_monday_0700_et():
    """2026-05-18: moved Sun 03:00 → Mon 07:00 (after retrain, before ingest)."""
    j = get("com.sma.autoresearch.nightly")
    assert j.days == (Day.MON,)
    assert j.fire_time_et == time(7, 0)
    assert j.deadline_offset_minutes == 180  # config search runs ~1.5-2h
    assert j.requires_writer_lock is True
    assert j.depends_on == ("com.sma.model.retrain.weekly",)


def test_decide_runs_at_2000_with_use_theses_dependency():
    j = get("com.sma.live.decide.daily")
    assert j.fire_time_et == time(20, 0)
    # Hard (blocking) dependencies, EXACTLY: trading cannot proceed without
    # prices + predictions, and nothing else may be silently downgraded.
    assert j.depends_on == ("com.sma.ingest.daily", "com.sma.model.predict.daily")
    # Agents is an ADVISORY overlay: it informs theses but must NOT hard-block
    # trading (a network blip on the agent calls froze the bot on 2026-06-04).
    assert j.advisory_deps == frozenset({"com.sma.agents.daily"})


def test_retrain_moved_to_monday():
    """2026-05-18: moved Sat 02:00 → Mon 04:00 to avoid weekend Mac-off."""
    j = get("com.sma.model.retrain.weekly")
    assert j.days == (Day.MON,)
    assert j.fire_time_et == time(4, 0)


def test_runs_today_uses_local_calendar():
    # 2026-04-30 is Thursday
    assert runs_today("com.sma.ingest.daily", asof=date(2026, 4, 30))
    assert not runs_today("com.sma.ingest.daily", asof=date(2026, 5, 2))  # Saturday


def test_next_fire_skips_past_today_if_already_past():
    # If the asked-from time is after today's fire, return tomorrow's
    after = datetime(2026, 4, 30, 21, 0, tzinfo=NY_TZ)  # 21:00 ET, decide fires 20:00
    nxt = next_fire("com.sma.live.decide.daily", after=after)
    assert nxt == datetime(2026, 5, 1, 20, 0, tzinfo=NY_TZ)


def test_deadline_is_fire_plus_offset():
    d = deadline("com.sma.live.decide.daily", asof=date(2026, 4, 30))
    assert d == datetime(2026, 4, 30, 21, 0, tzinfo=NY_TZ)  # 20:00 + 60 min offset


def test_next_fire_handles_naive_datetime_input():
    """Naive datetime inputs are converted to ET and compared correctly,
    not raising TypeError on the comparison."""
    naive_after = datetime(2026, 4, 30, 21, 0)  # naive, no tzinfo
    nxt = next_fire("com.sma.live.decide.daily", after=naive_after)
    # Should not raise; should return a future fire time
    assert nxt.tzinfo is not None
    assert nxt > naive_after.replace(tzinfo=NY_TZ)


def test_deadline_raises_when_job_does_not_run_today():
    # retrain runs only on Monday; Saturday should raise
    with pytest.raises(ValueError, match="does not run on"):
        deadline("com.sma.model.retrain.weekly", asof=date(2026, 5, 16))  # Saturday


def test_job_schedule_rejects_empty_days():
    with pytest.raises(ValueError, match="must have at least one Day"):
        JobSchedule(
            label="test",
            days=(),
            fire_time_et=time(12, 0),
            wake_lead_minutes=0,
            deadline_offset_minutes=10,
            depends_on=(),
            requires_writer_lock=False,
            waivers=frozenset(),
        )


def test_backup_daily_schedule():
    j = get("com.sma.backup.daily")
    assert j.fire_time_et == time(22, 0)
    assert j.wake_lead_minutes == 15
    assert j.deadline_offset_minutes == 30
    assert j.requires_writer_lock is True
    assert set(j.days) == {Day.MON, Day.TUE, Day.WED, Day.THU, Day.FRI, Day.SAT, Day.SUN}
    assert j.depends_on == ()
    assert j.waivers == frozenset()


def test_backup_daily_runs_on_sunday():
    # backup is now 7-day; Sunday must be included
    assert runs_today("com.sma.backup.daily", asof=date(2026, 5, 3))  # Sunday


def test_backup_daily_runs_on_saturday():
    assert runs_today("com.sma.backup.daily", asof=date(2026, 5, 2))  # Saturday


def test_next_fire_weekly_job_cross_week_skip():
    """If we ask for the next fire of a weekly Monday job AFTER Monday's fire time,
    the loop must walk all 7 days to next Monday."""
    after = datetime(2026, 5, 18, 5, 0, tzinfo=NY_TZ)  # Monday 05:00 ET (after 04:00 retrain)
    nxt = next_fire("com.sma.model.retrain.weekly", after=after)
    assert nxt == datetime(2026, 5, 25, 4, 0, tzinfo=NY_TZ)  # next Monday 04:00 ET
