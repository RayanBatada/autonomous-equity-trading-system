"""OPTIONAL_SCHEDULE: intraday ingest + the two sessions. Rendered, never
installed by install.sh, never evaluated by the watchdog unless installed."""

import plistlib
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

from sma import schedule as sched

REPO = Path(__file__).resolve().parents[3]
OPTIONAL = {"com.sma.ingest.intraday", "com.sma.live.session.midday",
            "com.sma.live.session.close"}


def test_optional_jobs_are_not_in_schedule_but_resolvable():
    assert {j.label for j in sched.OPTIONAL_SCHEDULE} == OPTIONAL
    assert not OPTIONAL & {j.label for j in sched.SCHEDULE}
    assert sched.get("com.sma.live.session.midday").fire_time_et.strftime("%H:%M") == "10:35"
    assert sched.get("com.sma.live.session.close").fire_time_et.strftime("%H:%M") == "15:45"
    assert sched.get("com.sma.ingest.intraday").fire_time_et.strftime("%H:%M") == "15:41"
    for label in OPTIONAL:
        assert len(sched.get(label).days) == 5


def test_committed_plists_match_renderer_and_args():
    import tempfile

    from ops.launchd.render_plists import render_all

    # The committed plists are the Mac's rendered output, so they name the
    # Mac's checkout and home. Render with THOSE paths, read back out of the
    # committed file, not with this machine's: CI checks out under
    # /home/runner and a worktree lives elsewhere, and both failed this
    # byte-for-byte match on paths alone (CI red 2026-09-27 to 2026-10-01).
    for label in OPTIONAL:
        committed_bytes = (REPO / "ops" / "launchd" / f"{label}.plist").read_bytes()
        committed = plistlib.loads(committed_bytes)
        repo_root = Path(committed["WorkingDirectory"])
        home_dir = Path(committed["StandardOutPath"].split("/Library/Logs/sma/")[0])
        venv = Path(committed["ProgramArguments"][2]).parent.parent
        with tempfile.TemporaryDirectory() as tmp:
            render_all(out_dir=Path(tmp), repo_root=repo_root, venv=venv, home_dir=home_dir)
            rendered = (Path(tmp) / f"{label}.plist").read_bytes()
        assert rendered == committed_bytes
    midday = plistlib.loads((REPO / "ops/launchd/com.sma.live.session.midday.plist").read_bytes())
    assert midday["ProgramArguments"][-4:] == ["sma.live", "session", "--name", "midday"]
    assert midday["RunAtLoad"] is False
    assert {(i["Hour"], i["Minute"]) for i in midday["StartCalendarInterval"]} == {(10, 35)}


def test_install_script_does_not_install_optional_jobs():
    text = (REPO / "ops" / "launchd" / "install.sh").read_text()
    for label in OPTIONAL:
        assert label not in text


def test_systemd_units_rendered(tmp_path):
    from ops.systemd.render_units import render_all
    render_all(out_dir=tmp_path, repo_root=REPO, venv=REPO / ".venv", home_dir=tmp_path)
    timer = (tmp_path / "sma-live.session.close.timer").read_text()
    assert "OnCalendar=Mon..Fri 15:45:00" in timer
    svc = (tmp_path / "sma-ingest.intraday.service").read_text()
    assert "-m sma.ingest intraday" in svc


def _check(monkeypatch, tmp_path, *, installed: bool, now):
    from sma import watchdog as wd
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path / "s"))
    monkeypatch.setattr(wd, "_now", lambda: now)
    monkeypatch.setattr(wd, "_is_trading_day", lambda today, notify_fn=None: True)
    adapter = MagicMock()
    adapter.installed.side_effect = lambda label: installed
    adapter.state.return_value = "not running"
    adapter.kickstart.return_value = MagicMock(returncode=0, stdout="1", stderr="")
    monkeypatch.setattr(wd, "get_adapter", lambda: adapter)
    notes = []
    wd.check(notify_fn=lambda **kw: notes.append(kw))
    return notes, adapter


def test_watchdog_ignores_uninstalled_optional_jobs(monkeypatch, tmp_path):
    # 13:00 Thu: midday session long past deadline, no sentinel, but not installed.
    notes, adapter = _check(monkeypatch, tmp_path, installed=False,
                            now=datetime(2026, 10, 1, 13, 0, tzinfo=sched.NY_TZ))
    assert not any("session" in n["message"] for n in notes)
    kicked = [c.args[0] for c in adapter.kickstart.call_args_list]
    assert not OPTIONAL & set(kicked)


def test_watchdog_pages_installed_missed_session_never_kicks_it(monkeypatch, tmp_path):
    notes, adapter = _check(monkeypatch, tmp_path, installed=True,
                            now=datetime(2026, 10, 1, 13, 0, tzinfo=sched.NY_TZ))
    assert any("com.sma.live.session.midday" in n["message"] for n in notes)
    kicked = [c.args[0] for c in adapter.kickstart.call_args_list]
    assert "com.sma.live.session.midday" not in kicked


def test_adapter_installed_defaults(monkeypatch, tmp_path):
    from sma import sched_adapter as sa
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    run = MagicMock(return_value=MagicMock(returncode=0, stdout="enabled\n"))
    monkeypatch.setattr(sa.subprocess, "run", run)
    assert sa.LaunchdAdapter().installed("com.sma.live.session.midday") is False  # no plist
    (tmp_path / "Library" / "LaunchAgents").mkdir(parents=True)
    (tmp_path / "Library/LaunchAgents/com.sma.live.session.midday.plist").write_text("x")
    assert sa.LaunchdAdapter().installed("com.sma.live.session.midday") is True
    assert sa.SystemdAdapter().installed("com.sma.live.session.midday") is True
    assert run.call_args.args[0] == ["systemctl", "--user", "is-enabled",
                                     "sma-live.session.midday.timer"]
    run.return_value = MagicMock(returncode=1, stdout="disabled\n")
    assert sa.SystemdAdapter().installed("com.sma.live.session.midday") is False
