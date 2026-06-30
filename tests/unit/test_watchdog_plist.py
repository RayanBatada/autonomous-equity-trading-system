import plistlib
from pathlib import Path

PLIST = Path(__file__).parents[2] / "ops" / "launchd" / "com.sma.watchdog.plist"


def test_watchdog_plist_runs_at_load_and_4_checkpoints():
    data = plistlib.loads(PLIST.read_bytes())
    assert data["Label"] == "com.sma.watchdog"
    assert data["RunAtLoad"] is True
    intervals = data["StartCalendarInterval"]
    hours = sorted(d["Hour"] for d in intervals)
    assert hours == [19, 20, 21, 22]


def test_watchdog_plist_uses_caffeinate_s():
    data = plistlib.loads(PLIST.read_bytes())
    args = data["ProgramArguments"]
    assert args[0].endswith("caffeinate")
    assert "-s" in args


def test_watchdog_plist_runs_python_module():
    data = plistlib.loads(PLIST.read_bytes())
    args = data["ProgramArguments"]
    assert "sma.watchdog" in args
