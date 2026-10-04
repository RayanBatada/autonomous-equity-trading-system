"""Unit tests for the ingest CLI's deadline-budget resolver.

2026-09-17 (found during the Sept 2026 outage repair): `python -m sma.ingest
run --asof-date D --sources finnhub_fundamentals,edgar,...` for a PAST D never
fetched overlay sources, because the deadline-budget guard compared D's
scheduled cutoff against the REAL current time and always found it "already
blown". The fix has two parts:

1. IngestRunner.run() (tests/unit/ingest/test_runner_retry.py) now only
   applies the budget on a scheduled SAME-DAY run (asof_date == "today" in
   ET) -- this alone makes any historical --asof-date run unbudgeted.
2. This module's `_resolve_ingest_deadline` adds an explicit `--no-deadline`
   CLI override on top, for forcing an unbudgeted run on ANY asof (including
   today's real scheduled run).
"""

from datetime import date

import pytest

from sma import schedule as sched
from sma.ingest.__main__ import _resolve_ingest_deadline


def test_no_deadline_flag_forces_none_on_a_scheduled_weekday():
    # 2026-09-16 is a Wednesday -- a day com.sma.ingest.daily normally runs
    # (and would otherwise get a real deadline back).
    asof = date(2026, 9, 16)
    assert _resolve_ingest_deadline(asof, no_deadline=True) is None


def test_no_deadline_flag_forces_none_on_a_non_scheduled_day_too():
    # 2026-05-16 is a Saturday (ingest doesn't run on it at all); the flag
    # must short-circuit before even consulting the schedule.
    asof = date(2026, 5, 16)
    assert _resolve_ingest_deadline(asof, no_deadline=True) is None


def test_default_returns_the_schedules_own_deadline_on_a_scheduled_weekday():
    asof = date(2026, 9, 16)
    expected = sched.deadline("com.sma.ingest.daily", asof=asof)
    assert _resolve_ingest_deadline(asof, no_deadline=False) == expected


def test_default_returns_none_when_the_job_does_not_run_that_day():
    # Preserves the pre-existing ValueError -> None behavior for non-trading
    # weekdays (a manual/backfill run against a Saturday, say).
    asof = date(2026, 5, 16)  # Saturday
    with pytest.raises(ValueError):
        sched.deadline("com.sma.ingest.daily", asof=asof)
    assert _resolve_ingest_deadline(asof, no_deadline=False) is None
