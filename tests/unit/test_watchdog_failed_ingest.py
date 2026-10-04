"""A failed ingest is retried by the watchdog (flaw hunt 2026-10-01, A1).

Before: any ingest sentinel, including one with quality.passed=false, made
the watchdog `continue`, so a night whose 18:30 ingest hit a dead network
was never re-run even when the network came back by 19:00. The watchdog
kicked predict and decide four times on 9/23 and never ingest (same shape
9/14, 9/16, 9/25). `python -m sma.ingest run` already re-runs when the
sentinel failed quality (ingest_succeeded_today), so the watchdog only has
to stop treating a failed sentinel as done.
"""

from datetime import date, datetime
from unittest.mock import MagicMock

import pytest

from sma import schedule as sched

INGEST = "com.sma.ingest.daily"
THU = date(2026, 4, 30)


@pytest.fixture
def wd(monkeypatch, tmp_path):
    from sma import watchdog as wd

    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path / "sentinels"))
    monkeypatch.setattr(wd, "_is_trading_day", lambda today, notify_fn=None: True)
    return wd


def _adapter(monkeypatch, wd, state="not running"):
    adapter = MagicMock()
    adapter.installed.return_value = False
    adapter.state.return_value = state
    adapter.kickstart.return_value = MagicMock(returncode=0, stdout="1", stderr="")
    monkeypatch.setattr(wd, "get_adapter", lambda: adapter)
    return adapter


def _ingest_sentinel(*, passed, blocking=(), holiday=False):
    from sma.sentinels import write_sentinel

    payload = {
        "label": INGEST,
        "asof": THU.isoformat(),
        "run_id": 1,
        "completed_at": "2026-04-30T22:42:47Z",
        "quality": {"passed": passed, "blocking_failures": list(blocking)},
    }
    if holiday:
        payload["holiday_skipped"] = True
    write_sentinel(label=INGEST, asof=THU, payload=payload)


def _kicked(adapter):
    return [c.args[0] for c in adapter.kickstart.call_args_list]


def test_failed_ingest_sentinel_is_rekicked_after_deadline(wd, monkeypatch):
    # The 10/2 sentinel shape: every source 0 rows, quality failed.
    _ingest_sentinel(passed=False, blocking=["all_tickers_have_price", "enough_sources_succeeded"])
    adapter = _adapter(monkeypatch, wd)
    monkeypatch.setattr(wd, "_now", lambda: datetime(2026, 4, 30, 21, 0, tzinfo=sched.NY_TZ))
    wd.check(notify_fn=lambda **kw: None)
    assert INGEST in _kicked(adapter)


def test_failed_ingest_not_kicked_before_deadline(wd, monkeypatch):
    _ingest_sentinel(passed=False, blocking=["all_tickers_have_price"])
    adapter = _adapter(monkeypatch, wd)
    monkeypatch.setattr(wd, "_now", lambda: datetime(2026, 4, 30, 20, 0, tzinfo=sched.NY_TZ))
    wd.check(notify_fn=lambda **kw: None)
    assert INGEST not in _kicked(adapter)


def test_failed_ingest_not_kicked_while_a_rerun_is_running(wd, monkeypatch):
    _ingest_sentinel(passed=False, blocking=["all_tickers_have_price"])
    adapter = _adapter(monkeypatch, wd, state="running")
    monkeypatch.setattr(wd, "_now", lambda: datetime(2026, 4, 30, 21, 0, tzinfo=sched.NY_TZ))
    wd.check(notify_fn=lambda **kw: None)
    assert INGEST not in _kicked(adapter)


def test_passing_ingest_sentinel_is_still_left_alone(wd, monkeypatch):
    _ingest_sentinel(passed=True)
    adapter = _adapter(monkeypatch, wd)
    monkeypatch.setattr(wd, "_now", lambda: datetime(2026, 4, 30, 21, 0, tzinfo=sched.NY_TZ))
    wd.check(notify_fn=lambda **kw: None)
    assert INGEST not in _kicked(adapter)


def test_holiday_skipped_ingest_is_left_alone(wd, monkeypatch):
    _ingest_sentinel(passed=True, holiday=True)
    adapter = _adapter(monkeypatch, wd)
    monkeypatch.setattr(wd, "_now", lambda: datetime(2026, 4, 30, 21, 0, tzinfo=sched.NY_TZ))
    wd.check(notify_fn=lambda **kw: None)
    assert INGEST not in _kicked(adapter)


def test_failed_sentinel_of_other_jobs_still_counts_as_done(wd, monkeypatch):
    """Only ingest is re-kicked on a failed verdict. A failed agents sentinel
    (credit out, every call errored) must not loop the agents job hourly:
    rerunning cannot fix it, and it would hold the writer lock across decide."""
    from sma.sentinels import write_sentinel

    _ingest_sentinel(passed=True)
    write_sentinel(
        label="com.sma.agents.daily",
        asof=THU,
        payload={
            "label": "com.sma.agents.daily",
            "asof": THU.isoformat(),
            "run_id": 2,
            "quality": {"passed": False, "blocking_failures": ["all_tickers_failed"]},
        },
    )
    adapter = _adapter(monkeypatch, wd)
    monkeypatch.setattr(wd, "_now", lambda: datetime(2026, 4, 30, 21, 0, tzinfo=sched.NY_TZ))
    wd.check(notify_fn=lambda **kw: None)
    assert "com.sma.agents.daily" not in _kicked(adapter)
