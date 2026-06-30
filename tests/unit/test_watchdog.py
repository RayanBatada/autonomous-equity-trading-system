from datetime import date, datetime
from unittest.mock import MagicMock, patch

from sma import schedule as sched


@patch("sma.watchdog.subprocess.run")
def test_kicks_missing_job_past_deadline(mock_run, tmp_path, monkeypatch):
    """When sentinel is missing AND deadline passed AND state is idle, watchdog kickstarts."""
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path / "sentinels"))
    # Simulate "now" at 21:00 ET on Thursday (after ingest deadline of 20:30)
    fake_now = datetime(2026, 4, 30, 21, 0, tzinfo=sched.NY_TZ)
    from sma import watchdog as wd

    monkeypatch.setattr(wd, "_now", lambda: fake_now)

    def _mock_run(args, **kwargs):
        if args[1] == "print":
            return MagicMock(returncode=0, stdout="state = not running\n", stderr="")
        return MagicMock(returncode=0, stdout="12345", stderr="")

    mock_run.side_effect = _mock_run

    wd.check()
    # ingest deadline is 18:30 + 120 min = 20:30 ET; now is 21:00 ET; sentinel missing; state idle
    kickstart_calls = [c for c in mock_run.call_args_list if "kickstart" in c.args[0]]
    assert any("com.sma.ingest.daily" in str(c) for c in kickstart_calls)


@patch("sma.watchdog.subprocess.run")
def test_does_not_kick_when_sentinel_exists(mock_run, tmp_path, monkeypatch):
    """If a sentinel for today exists, the watchdog skips that job."""
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path / "sentinels"))
    fake_now = datetime(2026, 4, 30, 21, 0, tzinfo=sched.NY_TZ)
    from sma import watchdog as wd

    monkeypatch.setattr(wd, "_now", lambda: fake_now)
    from sma.sentinels import write_sentinel

    write_sentinel(
        label="com.sma.ingest.daily",
        asof=date(2026, 4, 30),
        payload={
            "label": "com.sma.ingest.daily",
            "asof": "2026-04-30",
            "run_id": 1,
            "quality": {"passed": True},
        },
    )
    wd.check()
    # ingest sentinel exists, so the watchdog should not kick com.sma.ingest.daily
    calls = [str(c) for c in mock_run.call_args_list]
    assert not any("com.sma.ingest.daily" in c for c in calls)


@patch("sma.watchdog.subprocess.run")
def test_does_not_kick_before_deadline(mock_run, tmp_path, monkeypatch):
    """Before deadline, watchdog leaves the job alone (it's still expected to fire)."""
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path / "sentinels"))
    # 19:30 ET on Thursday: ingest deadline (20:30) hasn't passed yet
    fake_now = datetime(2026, 4, 30, 19, 30, tzinfo=sched.NY_TZ)
    from sma import watchdog as wd

    monkeypatch.setattr(wd, "_now", lambda: fake_now)
    wd.check()
    calls = [str(c) for c in mock_run.call_args_list]
    assert not any("com.sma.ingest.daily" in c for c in calls)


@patch("sma.watchdog.subprocess.run")
def test_does_not_kick_on_weekend_for_weekday_jobs(mock_run, tmp_path, monkeypatch):
    """On Saturday, watchdog ignores ingest (weekday-only job)."""
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path / "sentinels"))
    fake_now = datetime(2026, 5, 2, 21, 0, tzinfo=sched.NY_TZ)  # Saturday
    from sma import watchdog as wd

    monkeypatch.setattr(wd, "_now", lambda: fake_now)
    wd.check()
    calls = [str(c) for c in mock_run.call_args_list]
    assert not any("com.sma.ingest.daily" in c for c in calls)


@patch("sma.watchdog.subprocess.run")
def test_kickstart_failure_returns_nonzero(mock_run, tmp_path, monkeypatch):
    """If launchctl kickstart fails, watchdog returns nonzero."""
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path / "sentinels"))
    fake_now = datetime(2026, 4, 30, 21, 0, tzinfo=sched.NY_TZ)
    from sma import watchdog as wd

    monkeypatch.setattr(wd, "_now", lambda: fake_now)

    def _mock_run(args, **kwargs):
        if args[1] == "print":
            return MagicMock(returncode=0, stdout="state = not running\n", stderr="")
        return MagicMock(returncode=1, stdout="", stderr="kickstart failed")

    mock_run.side_effect = _mock_run
    rc = wd.check()
    assert rc == 1


@patch("sma.watchdog.subprocess.run")
def test_skips_when_launchctl_state_is_running(mock_run, tmp_path, monkeypatch):
    """Watchdog should skip jobs that are currently running."""
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path / "sentinels"))
    fake_now = datetime(2026, 4, 30, 21, 0, tzinfo=sched.NY_TZ)
    from sma import watchdog as wd

    monkeypatch.setattr(wd, "_now", lambda: fake_now)

    def _mock_run(args, **kwargs):
        if args[1] == "print":
            return MagicMock(returncode=0, stdout="state = running\n", stderr="")
        return MagicMock(returncode=0, stdout="123", stderr="")

    mock_run.side_effect = _mock_run
    wd.check()
    # No "kickstart" call should have been made
    kickstart_calls = [c for c in mock_run.call_args_list if "kickstart" in c.args[0]]
    assert len(kickstart_calls) == 0


@patch("sma.watchdog.subprocess.run")
def test_skips_when_launchctl_state_is_waiting(mock_run, tmp_path, monkeypatch):
    """Watchdog should skip jobs that are loaded but waiting (launchd plans to fire)."""
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path / "sentinels"))
    fake_now = datetime(2026, 4, 30, 21, 0, tzinfo=sched.NY_TZ)
    from sma import watchdog as wd

    monkeypatch.setattr(wd, "_now", lambda: fake_now)

    def _mock_run(args, **kwargs):
        if args[1] == "print":
            return MagicMock(returncode=0, stdout="state = waiting\n", stderr="")
        return MagicMock(returncode=0, stdout="123", stderr="")

    mock_run.side_effect = _mock_run
    wd.check()
    kickstart_calls = [c for c in mock_run.call_args_list if "kickstart" in c.args[0]]
    assert len(kickstart_calls) == 0


@patch("sma.watchdog.subprocess.run")
def test_kicks_when_state_is_not_running(mock_run, tmp_path, monkeypatch):
    """When state is 'not running' AND sentinel missing past deadline, watchdog kickstarts."""
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path / "sentinels"))
    fake_now = datetime(2026, 4, 30, 21, 0, tzinfo=sched.NY_TZ)
    from sma import watchdog as wd

    monkeypatch.setattr(wd, "_now", lambda: fake_now)

    def _mock_run(args, **kwargs):
        if args[1] == "print":
            return MagicMock(returncode=0, stdout="state = not running\n", stderr="")
        if "kickstart" in args:
            return MagicMock(returncode=0, stdout="999", stderr="")
        return MagicMock(returncode=0, stdout="", stderr="")

    mock_run.side_effect = _mock_run
    wd.check()
    kickstart_calls = [c for c in mock_run.call_args_list if "kickstart" in c.args[0]]
    assert len(kickstart_calls) > 0  # At least one job kicked


@patch("sma.watchdog.subprocess.run")
def test_too_late_job_alerts_instead_of_silent_skip(mock_run, tmp_path, monkeypatch):
    """A job >6h past deadline is too late to safely kick — but it must NOTIFY a
    human (a likely-missed trading day), not skip silently (2026-06-05 audit:
    this is why the freeze went unnoticed)."""
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path / "sentinels"))
    from sma import watchdog as wd

    monkeypatch.setattr(wd, "_is_trading_day", lambda today, **kw: True)
    fake_now = datetime(2026, 5, 3, 19, 30, tzinfo=sched.NY_TZ)  # Sunday 19:30 ET
    monkeypatch.setattr(wd, "_now", lambda: fake_now)

    notifies: list[tuple[str, str]] = []
    wd.check(notify_fn=lambda title, message: notifies.append((title, message)))

    # senate-ingest (Sun, deadline 12:00) is 7.5h past -> too late -> must alert.
    blob = " ".join(t + " " + m for t, m in notifies).lower()
    assert "senate-ingest" in blob
    assert "past deadline" in blob or "too late" in blob or "missed" in blob


@patch("sma.watchdog.subprocess.run")
def test_kickstart_failure_notifies(mock_run, tmp_path, monkeypatch):
    """A failed kickstart must NOTIFY a human, not just log (2026-06-05 audit)."""
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path / "sentinels"))
    from sma import watchdog as wd

    monkeypatch.setattr(wd, "_is_trading_day", lambda today, **kw: True)
    fake_now = datetime(2026, 4, 30, 21, 0, tzinfo=sched.NY_TZ)
    monkeypatch.setattr(wd, "_now", lambda: fake_now)

    def _mock_run(args, **kwargs):
        if args[1] == "print":
            return MagicMock(returncode=0, stdout="state = not running\n", stderr="")
        return MagicMock(returncode=1, stdout="", stderr="boom")

    mock_run.side_effect = _mock_run
    notifies: list[tuple[str, str]] = []
    rc = wd.check(notify_fn=lambda title, message: notifies.append((title, message)))

    assert rc == 1
    blob = " ".join(t + " " + m for t, m in notifies).lower()
    assert "kickstart" in blob and "fail" in blob


# --- lineage-aware re-kick + liveness labels (2026-06-09) ---------------------


def _mock_launchctl_idle(mock_run):
    def _mock(args, **kwargs):
        if args[1] == "print":
            return MagicMock(returncode=0, stdout="state = not running\n", stderr="")
        return MagicMock(returncode=0, stdout="12345", stderr="")

    mock_run.side_effect = _mock


@patch("sma.watchdog.subprocess.run")
def test_kicks_predict_when_sentinel_lineage_stale(mock_run, tmp_path, monkeypatch):
    """predict has a sentinel — but it consumed an OLDER ingest run than the
    current ingest sentinel records. The watchdog must re-kick predict so the
    healed ingest produces fresh predictions (2026-06-09 recovery hole)."""
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path / "sentinels"))
    fake_now = datetime(2026, 4, 30, 21, 30, tzinfo=sched.NY_TZ)
    from sma import watchdog as wd
    from sma.sentinels import write_sentinel

    monkeypatch.setattr(wd, "_now", lambda: fake_now)
    write_sentinel(
        label="com.sma.ingest.daily",
        asof=date(2026, 4, 30),
        payload={"run_id": 200, "quality": {"passed": True, "blocking_failures": []}},
    )
    write_sentinel(
        label="com.sma.model.predict.daily",
        asof=date(2026, 4, 30),
        payload={"run_id": 1, "completed_at": "x", "ingest_run_id": 100},
    )
    _mock_launchctl_idle(mock_run)
    wd.check(notify_fn=lambda title, message: None)
    kicks = [c for c in mock_run.call_args_list if "kickstart" in str(c)]
    assert any("com.sma.model.predict.daily" in str(c) for c in kicks)


@patch("sma.watchdog.subprocess.run")
def test_does_not_kick_predict_when_lineage_current(mock_run, tmp_path, monkeypatch):
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path / "sentinels"))
    fake_now = datetime(2026, 4, 30, 21, 30, tzinfo=sched.NY_TZ)
    from sma import watchdog as wd
    from sma.sentinels import write_sentinel

    monkeypatch.setattr(wd, "_now", lambda: fake_now)
    write_sentinel(
        label="com.sma.ingest.daily",
        asof=date(2026, 4, 30),
        payload={"run_id": 200, "quality": {"passed": True, "blocking_failures": []}},
    )
    write_sentinel(
        label="com.sma.model.predict.daily",
        asof=date(2026, 4, 30),
        payload={"run_id": 1, "completed_at": "x", "ingest_run_id": 200},
    )
    _mock_launchctl_idle(mock_run)
    wd.check(notify_fn=lambda title, message: None)
    kicks = [c for c in mock_run.call_args_list if "kickstart" in str(c)]
    assert not any("com.sma.model.predict.daily" in str(c) for c in kicks)


@patch("sma.watchdog.subprocess.run")
def test_reconcile_liveness_sentinel_prevents_daily_rekick(mock_run, tmp_path, monkeypatch):
    """reconcile's BATCH sentinel is keyed by the decide-date it reconciled
    (yesterday), never today — so the watchdog used to re-kick reconcile every
    hour 19:00-22:00, every day. It must check the run-date LIVENESS sentinel
    instead."""
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path / "sentinels"))
    fake_now = datetime(2026, 4, 30, 19, 0, tzinfo=sched.NY_TZ)
    from sma import watchdog as wd
    from sma.sentinels import write_sentinel

    monkeypatch.setattr(wd, "_now", lambda: fake_now)
    # liveness sentinel for TODAY exists (reconcile ran at 16:30, batch was
    # yesterday's asof)
    write_sentinel(
        label="com.sma.live.reconcile.daily.ran",
        asof=date(2026, 4, 30),
        payload={"run_id": 1, "completed_at": "x", "reconciled_asof": "2026-04-29"},
    )
    _mock_launchctl_idle(mock_run)
    wd.check(notify_fn=lambda title, message: None)
    kicks = [c for c in mock_run.call_args_list if "kickstart" in str(c)]
    assert not any("com.sma.live.reconcile.daily" in str(c) for c in kicks)


@patch("sma.watchdog.subprocess.run")
def test_reconcile_kicked_when_liveness_sentinel_missing(mock_run, tmp_path, monkeypatch):
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path / "sentinels"))
    fake_now = datetime(2026, 4, 30, 19, 0, tzinfo=sched.NY_TZ)
    from sma import watchdog as wd
    from sma.sentinels import write_sentinel

    monkeypatch.setattr(wd, "_now", lambda: fake_now)
    # batch sentinel for YESTERDAY exists, but no liveness sentinel for today
    write_sentinel(
        label="com.sma.live.reconcile.daily",
        asof=date(2026, 4, 29),
        payload={"run_id": 1, "completed_at": "x"},
    )
    _mock_launchctl_idle(mock_run)
    wd.check(notify_fn=lambda title, message: None)
    kicks = [c for c in mock_run.call_args_list if "kickstart" in str(c)]
    assert any("com.sma.live.reconcile.daily" in str(c) for c in kicks)


def test_watchdog_notifies_when_autoresearch_backup_left_behind(tmp_path, monkeypatch):
    """A SIGKILL during autoresearch eval leaves the LLM proposal swapped into
    the LIVE src/sma/strategy/active.py with the original stranded in
    .py.autoresearch_bak — live decide would import the unevaluated proposal.
    The watchdog must page when the backup file exists (Codex module review
    2026-06-11 HIGH)."""
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path / "s"))
    from sma import watchdog as wd

    fake_now = datetime(2026, 4, 30, 19, 0, tzinfo=sched.NY_TZ)
    monkeypatch.setattr(wd, "_now", lambda: fake_now)
    bak = tmp_path / "active.py.autoresearch_bak"
    bak.write_text("def tilt(...): ...")
    monkeypatch.setattr(wd, "_AUTORESEARCH_BAK_PATH", bak)

    notifications = []
    with patch("sma.watchdog.subprocess.run") as mock_run:
        def _mock(args, **kwargs):
            if args[1] == "print":
                return MagicMock(returncode=0, stdout="state = not running\n", stderr="")
            return MagicMock(returncode=0, stdout="1", stderr="")
        mock_run.side_effect = _mock
        wd.check(notify_fn=lambda title, message: notifications.append((title, message)))
    assert any("active.py" in (t + m) for t, m in notifications)
