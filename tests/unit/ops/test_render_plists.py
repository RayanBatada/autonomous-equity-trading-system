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
    assert sorted(d["Hour"] for d in intervals) == [19, 20, 21, 22]
    assert watchdog["RunAtLoad"] is True


def test_dashboard_keepalive(tmp_path):
    out_dir = tmp_path / "rendered"
    render_all(out_dir=out_dir, repo_root=Path("/REPO"), venv=Path("/VENV"), home_dir=Path("/HOME"))
    dashboard = plistlib.loads((out_dir / "com.sma.dashboard.plist").read_bytes())
    # Either KeepAlive boolean or dict; just verify it's truthy
    assert dashboard.get("KeepAlive") in (True, {"SuccessfulExit": False})
    assert dashboard["RunAtLoad"] is True


def test_renders_all_14_plists(tmp_path):
    out_dir = tmp_path / "rendered"
    render_all(out_dir=out_dir, repo_root=Path("/REPO"), venv=Path("/VENV"), home_dir=Path("/HOME"))
    plists = list(out_dir.glob("com.sma.*.plist"))
    assert len(plists) == 14  # 12 SCHEDULE jobs + watchdog + dashboard


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
