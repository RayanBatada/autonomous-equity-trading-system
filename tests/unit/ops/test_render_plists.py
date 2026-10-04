import plistlib
from pathlib import Path

from ops.launchd.render_plists import _launchd_weekday, render_all


def test_decide_runs_at_2000_with_use_theses(tmp_path):
    """2026-05-11: --use-theses re-enabled after agents.daily proved it was
    generating useful theses (1,513 rows accumulated). The 2026-05-06
    turnover bug that originally motivated removing the flag was unrelated
    (translate.py architectural issue, fixed in 4ee4b64)."""
    out_dir = tmp_path / "rendered"
    render_all(out_dir=out_dir, repo_root=Path("/REPO"), venv=Path("/VENV"), home_dir=Path("/HOME"))
    decide = plistlib.loads((out_dir / "com.sma.live.decide.daily.plist").read_bytes())
    intervals = decide["StartCalendarInterval"]
    assert all(d["Hour"] == 20 and d["Minute"] == 0 for d in intervals)
    args = decide["ProgramArguments"]
    assert "--use-theses" in args


def test_retrain_runs_monday_0400(tmp_path):
    """2026-05-18: moved off Saturday to avoid weekend Mac-off."""
    out_dir = tmp_path / "rendered"
    render_all(out_dir=out_dir, repo_root=Path("/REPO"), venv=Path("/VENV"), home_dir=Path("/HOME"))
    retrain = plistlib.loads((out_dir / "com.sma.model.retrain.weekly.plist").read_bytes())
    intervals = retrain["StartCalendarInterval"]
    assert len(intervals) == 1
    # launchd Weekday=1 is Monday
    assert intervals[0] == {"Weekday": 1, "Hour": 4, "Minute": 0}


def test_caffeinate_s_except_dashboard(tmp_path):
    out_dir = tmp_path / "rendered"
    render_all(out_dir=out_dir, repo_root=Path("/REPO"), venv=Path("/VENV"), home_dir=Path("/HOME"))
    for plist_file in out_dir.glob("com.sma.*.plist"):
        data = plistlib.loads(plist_file.read_bytes())
        args = data["ProgramArguments"]
        if data["Label"] == "com.sma.dashboard":
            assert "-i" in args
        else:
            assert "-s" in args


def test_watchdog_hourly_checkpoints(tmp_path):
    out_dir = tmp_path / "rendered"
    render_all(out_dir=out_dir, repo_root=Path("/REPO"), venv=Path("/VENV"), home_dir=Path("/HOME"))
    watchdog = plistlib.loads((out_dir / "com.sma.watchdog.plist").read_bytes())
    intervals = watchdog["StartCalendarInterval"]
    # 0/1 (2026-08-05): overnight checkpoints added post the 8/4 no-trade
    # outage so a late-finishing evening chain can still self-heal.
    assert sorted(d["Hour"] for d in intervals) == [0, 1, 10, 13, 16, 19, 20, 21, 22, 23]
    assert watchdog["RunAtLoad"] is True


def test_watchdog_overnight_checkpoints_are_half_past(tmp_path):
    """0:30/1:30 (not on the hour): inside predict's (deadline 19:45 + 6h
    late-kick = 01:45) and decide's (deadline 21:00 + 6h = 03:00) windows."""
    out_dir = tmp_path / "rendered"
    render_all(out_dir=out_dir, repo_root=Path("/REPO"), venv=Path("/VENV"), home_dir=Path("/HOME"))
    watchdog = plistlib.loads((out_dir / "com.sma.watchdog.plist").read_bytes())
    intervals = watchdog["StartCalendarInterval"]
    overnight = {(d["Hour"], d["Minute"]) for d in intervals if d["Hour"] in (0, 1)}
    assert overnight == {(0, 30), (1, 30)}


def test_dashboard_keepalive(tmp_path):
    out_dir = tmp_path / "rendered"
    render_all(out_dir=out_dir, repo_root=Path("/REPO"), venv=Path("/VENV"), home_dir=Path("/HOME"))
    dashboard = plistlib.loads((out_dir / "com.sma.dashboard.plist").read_bytes())
    # Either KeepAlive boolean or dict; just verify it's truthy
    assert dashboard.get("KeepAlive") in (True, {"SuccessfulExit": False})
    assert dashboard["RunAtLoad"] is True


def test_renders_all_18_plists(tmp_path):
    out_dir = tmp_path / "rendered"
    render_all(out_dir=out_dir, repo_root=Path("/REPO"), venv=Path("/VENV"), home_dir=Path("/HOME"))
    plists = list(out_dir.glob("com.sma.*.plist"))
    assert len(plists) == 18  # 13 SCHEDULE + 3 OPTIONAL + watchdog + dashboard


def test_weekly_digest_plist_runs_sunday_1800(tmp_path):
    """2026-08-29: week-in-review digest -- read-only, no writer lock."""
    out_dir = tmp_path / "rendered"
    render_all(out_dir=out_dir, repo_root=Path("/REPO"), venv=Path("/VENV"), home_dir=Path("/HOME"))
    digest = plistlib.loads(
        (out_dir / "com.sma.weekly-digest.weekly.plist").read_bytes()
    )
    intervals = digest["StartCalendarInterval"]
    assert len(intervals) == 1
    # launchd Weekday=0 is Sunday
    assert intervals[0] == {"Weekday": 0, "Hour": 18, "Minute": 0}
    args = digest["ProgramArguments"]
    assert "sma.monitoring" in args
    assert "weekly-digest" in args
    assert digest["RunAtLoad"] is False


def test_autoresearch_nightly_plist_monday_0700(tmp_path):
    """2026-05-18: moved off Sunday to avoid weekend Mac-off."""
    out_dir = tmp_path / "rendered"
    render_all(out_dir=out_dir, repo_root=Path("/REPO"), venv=Path("/VENV"), home_dir=Path("/HOME"))
    autoresearch = plistlib.loads(
        (out_dir / "com.sma.autoresearch.nightly.plist").read_bytes()
    )
    intervals = autoresearch["StartCalendarInterval"]
    assert len(intervals) == 1
    # launchd Weekday=1 is Monday
    assert intervals[0] == {"Weekday": 1, "Hour": 7, "Minute": 0}
    args = autoresearch["ProgramArguments"]
    assert "sma.autoresearch" in args
    assert "search" in args
    assert "--n-configs" in args
    assert "run" not in args  # the old tilt()-rewriting loop is gone


def test_retrain_plist_keeps_demean_labels(tmp_path):
    """The weekly retrain MUST train demean labels (multi-regime fix). A missing
    --demean-labels would silently regress to raw labels (regime inversion)."""
    out_dir = tmp_path / "rendered"
    render_all(out_dir=out_dir, repo_root=Path("/REPO"), venv=Path("/VENV"), home_dir=Path("/HOME"))
    retrain = plistlib.loads(
        (out_dir / "com.sma.model.retrain.weekly.plist").read_bytes()
    )
    assert "--demean-labels" in retrain["ProgramArguments"]


def test_launchd_weekday_conversion():
    """Day enum to launchd Weekday: Mon-Sat map directly (1-6); Sun maps from 7 to 0."""
    from sma.schedule import Day

    assert _launchd_weekday(Day.MON) == 1
    assert _launchd_weekday(Day.SAT) == 6
    assert _launchd_weekday(Day.SUN) == 0


def test_only_watchdog_and_dashboard_run_at_load(tmp_path):
    """RunAtLoad must stay OFF for every SCHEDULED job.

    2026-08-12: the Mac rebooted at 20:11 ET and all 14 jobs ran at boot. The
    first suspicion was RunAtLoad, but it was already false everywhere except
    the two jobs below -- installed plists included, verified with
    `launchctl print` (only watchdog/dashboard carry the `runatload` property).
    The actual mechanism is launchd's missed-StartCalendarInterval catch-up: a
    job whose calendar time passed while the machine was off runs at the next
    boot or wake, and that is NOT suppressible from the plist.

    So this test does not fix the boot storm -- the in-job guards do (see
    _sweep_skip_reason in sma/live/__main__.py and the ingest gate in
    sma/agents/__main__.py). It exists so nobody "fixes" a future boot storm by
    switching RunAtLoad on, which would add a second, genuinely
    plist-controlled way to fire the whole manifest at login.

    watchdog: RunAtLoad is deliberate -- a fast, deadline-guarded recovery pass
    at boot. dashboard: a KeepAlive daemon, not a scheduled job.
    """
    out_dir = tmp_path / "rendered"
    render_all(out_dir=out_dir, repo_root=Path("/REPO"), venv=Path("/VENV"), home_dir=Path("/HOME"))

    run_at_load = {
        plistlib.loads(f.read_bytes())["Label"]: plistlib.loads(f.read_bytes())["RunAtLoad"]
        for f in out_dir.glob("com.sma.*.plist")
    }

    assert {label for label, v in run_at_load.items() if v} == {
        "com.sma.watchdog",
        "com.sma.dashboard",
    }
    # And every scheduled job must actually HAVE a calendar trigger, since that
    # is now its only way to start.
    for f in out_dir.glob("com.sma.*.plist"):
        data = plistlib.loads(f.read_bytes())
        if data["Label"] == "com.sma.dashboard":
            continue
        assert data.get("StartCalendarInterval"), f"{data['Label']} has no trigger"


def test_backup_fires_2200_every_day(tmp_path):
    """The backup is the last job of the night and the only one that runs 7
    days a week; a dropped interval loses the day's snapshot silently."""
    out_dir = tmp_path / "rendered"
    render_all(out_dir=out_dir, repo_root=Path("/REPO"), venv=Path("/VENV"), home_dir=Path("/HOME"))
    backup = plistlib.loads((out_dir / "com.sma.backup.daily.plist").read_bytes())

    intervals = backup["StartCalendarInterval"]
    assert sorted(d["Weekday"] for d in intervals) == [0, 1, 2, 3, 4, 5, 6]
    assert all(d["Hour"] == 22 and d["Minute"] == 0 for d in intervals)
    assert backup["RunAtLoad"] is False
