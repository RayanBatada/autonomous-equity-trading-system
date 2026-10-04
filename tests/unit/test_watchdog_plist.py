import plistlib
from pathlib import Path

PLIST = Path(__file__).parents[2] / "ops" / "launchd" / "com.sma.watchdog.plist"


def test_watchdog_plist_runs_at_load_and_checkpoints():
    data = plistlib.loads(PLIST.read_bytes())
    assert data["Label"] == "com.sma.watchdog"
    assert data["RunAtLoad"] is True
    intervals = data["StartCalendarInterval"]
    hours = sorted(d["Hour"] for d in intervals)
    # 19-23: evening pipeline checks; 23:00 catches a missed 22:00 backup
    # (deadline 22:30 — review 2026-07-04). 10/13/16: daytime checkpoints so a
    # Mac that slept through the early slots and wakes MIDDAY is caught while
    # retrain/autoresearch/senate/house are still inside their late-kick windows
    # (2026-07-13/20: the first check at 19:00 was always "too late").
    # 0/1 (2026-08-05 post-mortem): overnight checkpoints so a chain that
    # finishes after the last evening checkpoint (8/4: ingest finished 23:54,
    # after 23:00) still gets re-kicked inside predict/decide's late-kick
    # windows (close 01:45/03:00) instead of losing the whole night.
    assert hours == [0, 1, 10, 13, 16, 19, 20, 21, 22, 23]


def test_watchdog_plist_overnight_checkpoints_are_half_past():
    """0:30/1:30, not on the hour — chosen to sit inside both predict's
    (deadline 19:45 + 6h = 01:45) and decide's (deadline 21:00 + 6h = 03:00)
    late-kick windows (2026-08-05 post-mortem)."""
    data = plistlib.loads(PLIST.read_bytes())
    intervals = data["StartCalendarInterval"]
    overnight = {(d["Hour"], d["Minute"]) for d in intervals if d["Hour"] in (0, 1)}
    assert overnight == {(0, 30), (1, 30)}


def test_watchdog_plist_uses_caffeinate_s():
    data = plistlib.loads(PLIST.read_bytes())
    args = data["ProgramArguments"]
    assert args[0].endswith("caffeinate")
    assert "-s" in args


def test_watchdog_plist_runs_python_module():
    data = plistlib.loads(PLIST.read_bytes())
    args = data["ProgramArguments"]
    assert "sma.watchdog" in args
