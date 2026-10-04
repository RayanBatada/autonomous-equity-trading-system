"""Render macOS launchd plist files from the schedule manifest.

Usage:
    python -m ops.launchd.render_plists [--out-dir DIR] [--check]

--out-dir DIR : write plists to DIR (default: ops/launchd/ relative to repo root)
--check       : render to a temp dir, diff against ~/Library/LaunchAgents/com.sma.*.plist,
                exit 1 on drift
"""

from __future__ import annotations

import argparse
import difflib
import plistlib
import sys
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Short log name: strip com.sma. prefix and trailing frequency suffix.
# Examples:
#   com.sma.ingest.daily        -> ingest
#   com.sma.live.decide.daily   -> live.decide
#   com.sma.model.retrain.weekly -> model.retrain
#   com.sma.live.stop-loss.weekday -> live.stop-loss
# ---------------------------------------------------------------------------
_FREQ_SUFFIXES = (".daily", ".weekday", ".weekly")


def _short_name(label: str) -> str:
    name = label.removeprefix("com.sma.")
    for suffix in _FREQ_SUFFIXES:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name


def _launchd_weekday(day: object) -> int:
    """Convert a Day enum value to a launchd StartCalendarInterval Weekday integer.

    launchd uses 0=Sunday, 1=Monday, ..., 6=Saturday.
    Day enum uses 1=Monday, ..., 6=Saturday, 7=Sunday (ISO isoweekday).
    Mon-Sat overlap (1-6), but Day.SUN=7 must become 0.
    """
    from sma.schedule import Day

    val = int(day)
    if val == Day.SUN:
        return 0
    return val


def _program_args(label: str, venv_python: str) -> list[str]:
    """Return ProgramArguments list for a given job label."""
    if label == "com.sma.dashboard":
        venv_dir = str(Path(venv_python).parent)
        return [
            "/usr/bin/caffeinate",
            "-i",
            f"{venv_dir}/streamlit",
            "run",
            "dashboard/app.py",
            "--server.port",
            "8765",
            "--server.headless",
            "true",
            "--browser.gatherUsageStats",
            "false",
        ]

    base = ["/usr/bin/caffeinate", "-s", venv_python]

    _map: dict[str, list[str]] = {
        "com.sma.ingest.daily": ["-m", "sma.ingest", "run"],
        "com.sma.model.predict.daily": ["-m", "sma.model", "predict"],
        # --demean-labels is REQUIRED: the live model trains on cross-sectionally
        # demeaned (alpha) labels (multi-regime fix, 2026-06-16). The installed
        # plist had it but this manifest had dropped it — a re-render+install
        # would have silently regressed the retrain to raw labels (regime
        # inversion). Added 2026-06-19; test_retrain_plist_keeps_demean_labels guards it.
        "com.sma.model.retrain.weekly": ["-m", "sma.model", "train", "--demean-labels"],
        "com.sma.agents.daily": ["-m", "sma.agents", "run"],
        # 2026-05-11: --use-theses re-enabled. Agents pipeline has produced
        # 1,513 theses since 5/3; not using them was leaving paid LLM
        # analysis on the table. The 2026-05-06 turnover bug that originally
        # motivated removing the flag was an unrelated architectural issue
        # in translate.py (fixed in 4ee4b64), not caused by theses.
        "com.sma.live.decide.daily": ["-m", "sma.live", "decide", "--use-theses"],
        "com.sma.live.stop-loss.weekday": ["-m", "sma.live", "stop-loss-sweep"],
        "com.sma.live.reconcile.daily": ["-m", "sma.live", "reconcile"],
        "com.sma.backup.daily": ["-m", "sma.backup", "run"],
        "com.sma.watchdog": ["-m", "sma.watchdog"],
        # The config search (CV-IC over the training space, auto-promotes a
        # winner through the IC gate). Replaced the old tilt()-rewriting `run`
        # loop (2026-06-19); --n-configs bounds runtime to ~1.5-2h (each config
        # is a 5-fold walk-forward CV on ~460k rows). Seed defaults to the date.
        "com.sma.autoresearch.nightly": ["-m", "sma.autoresearch", "search",
                                          "--n-configs", "6"],
        "com.sma.monitoring.daily": ["-m", "sma.monitoring", "check"],
        "com.sma.senate-ingest.weekly": ["-m", "sma.ingest.sources.senate_trades"],
        "com.sma.house-ingest.weekly": ["-m", "sma.ingest.sources.politician_trades"],
        "com.sma.weekly-digest.weekly": ["-m", "sma.monitoring", "weekly-digest"],
        # OPTIONAL_SCHEDULE (2026-09-26): rendered, never installed by install.sh.
        "com.sma.ingest.intraday": ["-m", "sma.ingest", "intraday"],
        "com.sma.live.session.midday": ["-m", "sma.live", "session", "--name", "midday"],
        "com.sma.live.session.close": ["-m", "sma.live", "session", "--name", "close"],
    }
    extra = _map.get(label)
    if extra is None:
        raise ValueError(f"unknown label: {label!r}")
    return base + extra


def _render_schedule_plist(
    job: object,
    repo_root: Path,
    venv: Path,
    home_dir: Path,
) -> dict:
    """Render a plist dict for a SCHEDULE job."""
    from sma.schedule import JobSchedule

    assert isinstance(job, JobSchedule)
    venv_python = str(venv / "bin" / "python")
    short = _short_name(job.label)
    intervals = [
        {
            "Weekday": _launchd_weekday(d),
            "Hour": job.fire_time_et.hour,
            "Minute": job.fire_time_et.minute,
        }
        for d in job.days
    ]
    return {
        "Label": job.label,
        "ProgramArguments": _program_args(job.label, venv_python),
        "WorkingDirectory": str(repo_root),
        "EnvironmentVariables": {
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            # Codex MED 2026-05-14: launchd StartCalendarInterval uses LOCAL
            # wall-clock time. If the laptop is ever in a non-ET timezone,
            # jobs fire at the wrong market phase. TZ=ET pins the process
            # clock (used by datetime.now()) but does NOT affect when launchd
            # schedules the wake — that still follows system local time.
            # This is defence in depth, not a substitute for keeping the
            # host on ET. Documented limitation; flag if you travel.
            "TZ": "America/New_York",
        },
        "StartCalendarInterval": intervals,
        "StandardOutPath": f"{home_dir}/Library/Logs/sma/{short}.out.log",
        "StandardErrorPath": f"{home_dir}/Library/Logs/sma/{short}.err.log",
        "RunAtLoad": False,
    }


def _render_watchdog_plist(
    repo_root: Path,
    venv: Path,
    home_dir: Path,
) -> dict:
    """Render the watchdog plist: checkpoints 10/13/16 + 19-23 ET + 0:30/1:30
    overnight, RunAtLoad=true.

    19-23: evening pipeline checks; 23:00 catches a missed backup.daily (fires
    22:00, deadline 22:30 — review 2026-07-04). 10/13/16: DAYTIME checkpoints so
    a Mac that slept through the early slots (Mon 04:00 retrain / 07:00
    autoresearch, Sun 10:00/11:00 senate/house) and wakes midday is caught while
    those jobs are still inside their per-job late-kick windows — with only the
    19:00 first check, every wake-after-sleep was "too late to kick" and the
    model went a week stale (2026-07-13/20).

    0:30/1:30 (2026-08-05 post-mortem): a battery-slowed Mac let 8/4's ingest
    run 18:30->23:54 (5.5h) holding the writer lock, past the last evening
    checkpoint (23:00) — so predict/decide's late-kick windows (deadline+6h:
    predict 19:45->01:45, decide 21:00->03:00) never got a checkpoint inside
    them, nothing re-kicked predict once ingest finally released the lock, and
    decide correctly refused to trade on stale/missing predictions. These two
    overnight checkpoints land inside both windows so a chain that finishes
    very late still gets one more chance to self-heal instead of losing the
    whole night's rebalance.
    """
    venv_python = str(venv / "bin" / "python")
    return {
        "Label": "com.sma.watchdog",
        "ProgramArguments": _program_args("com.sma.watchdog", venv_python),
        "WorkingDirectory": str(repo_root),
        "EnvironmentVariables": {
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            # Codex MED 2026-05-14: launchd StartCalendarInterval uses LOCAL
            # wall-clock time. If the laptop is ever in a non-ET timezone,
            # jobs fire at the wrong market phase. TZ=ET pins the process
            # clock (used by datetime.now()) but does NOT affect when launchd
            # schedules the wake — that still follows system local time.
            # This is defence in depth, not a substitute for keeping the
            # host on ET. Documented limitation; flag if you travel.
            "TZ": "America/New_York",
        },
        "StartCalendarInterval": [
            {"Hour": h, "Minute": 0} for h in (10, 13, 16, 19, 20, 21, 22, 23)
        ] + [
            # Overnight checkpoints (2026-08-05, see docstring): half-past so
            # they sit inside both predict's (closes 01:45) and decide's
            # (closes 03:00) late-kick windows rather than landing exactly on
            # a boundary.
            {"Hour": 0, "Minute": 30},
            {"Hour": 1, "Minute": 30},
        ],
        "StandardOutPath": f"{home_dir}/Library/Logs/sma/watchdog.out.log",
        "StandardErrorPath": f"{home_dir}/Library/Logs/sma/watchdog.err.log",
        "RunAtLoad": True,
    }


def _render_dashboard_plist(
    repo_root: Path,
    venv: Path,
    home_dir: Path,
) -> dict:
    """Render the dashboard plist: caffeinate -i, KeepAlive=true, RunAtLoad=true."""
    venv_python = str(venv / "bin" / "python")
    return {
        "Label": "com.sma.dashboard",
        "ProgramArguments": _program_args("com.sma.dashboard", venv_python),
        "WorkingDirectory": str(repo_root),
        "EnvironmentVariables": {
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            # Codex MED 2026-05-14: launchd StartCalendarInterval uses LOCAL
            # wall-clock time. If the laptop is ever in a non-ET timezone,
            # jobs fire at the wrong market phase. TZ=ET pins the process
            # clock (used by datetime.now()) but does NOT affect when launchd
            # schedules the wake — that still follows system local time.
            # This is defence in depth, not a substitute for keeping the
            # host on ET. Documented limitation; flag if you travel.
            "TZ": "America/New_York",
        },
        "StandardOutPath": f"{home_dir}/Library/Logs/sma/dashboard.out.log",
        "StandardErrorPath": f"{home_dir}/Library/Logs/sma/dashboard.err.log",
        "RunAtLoad": True,
        "KeepAlive": True,
    }


def render_all(
    out_dir: Path,
    repo_root: Path,
    venv: Path,
    home_dir: Path,
) -> None:
    """Render all 18 SMA plists into out_dir.

    1. 13 plists from SCHEDULE (one per job, including house/senate/weekly-digest),
       plus 3 from OPTIONAL_SCHEDULE (intraday ingest + the two sessions; never
       installed by install.sh).
    2. com.sma.watchdog (hourly checkpoints, not in SCHEDULE).
    3. com.sma.dashboard (long-running daemon, not in SCHEDULE).
    """
    from sma.schedule import OPTIONAL_SCHEDULE, SCHEDULE

    out_dir.mkdir(parents=True, exist_ok=True)

    # OPTIONAL_SCHEDULE plists are rendered next to the rest but install.sh's
    # JOBS list does not include them: they ship unloaded (see README.md).
    for job in (*SCHEDULE, *OPTIONAL_SCHEDULE):
        data = _render_schedule_plist(job, repo_root=repo_root, venv=venv, home_dir=home_dir)
        _write_plist(out_dir / f"{job.label}.plist", data)

    watchdog = _render_watchdog_plist(repo_root=repo_root, venv=venv, home_dir=home_dir)
    _write_plist(out_dir / "com.sma.watchdog.plist", watchdog)

    dashboard = _render_dashboard_plist(repo_root=repo_root, venv=venv, home_dir=home_dir)
    _write_plist(out_dir / "com.sma.dashboard.plist", dashboard)


def _write_plist(path: Path, data: dict) -> None:
    """Write plist data to path using binary plistlib, then re-read as XML for consistency."""
    path.write_bytes(plistlib.dumps(data, fmt=plistlib.FMT_XML, sort_keys=False))


def _check_drift(repo_root: Path, venv: Path, home_dir: Path) -> int:
    """Render to a tempdir, diff against ~/Library/LaunchAgents/com.sma.*.plist.

    Returns 0 if no drift, 1 on any difference.
    """
    agents_dir = home_dir / "Library" / "LaunchAgents"
    installed = list(agents_dir.glob("com.sma.*.plist"))
    if not installed:
        print(f"No installed SMA plists found in {agents_dir}")
        return 1

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        render_all(out_dir=tmp_dir, repo_root=repo_root, venv=venv, home_dir=home_dir)

        drifted: list[str] = []
        for installed_path in sorted(installed):
            rendered_path = tmp_dir / installed_path.name
            if not rendered_path.exists():
                print(f"  MISSING in rendered output: {installed_path.name}")
                drifted.append(installed_path.name)
                continue

            installed_text = installed_path.read_text()
            rendered_text = rendered_path.read_text()
            if installed_text != rendered_text:
                diff = list(
                    difflib.unified_diff(
                        installed_text.splitlines(keepends=True),
                        rendered_text.splitlines(keepends=True),
                        fromfile=f"installed/{installed_path.name}",
                        tofile=f"rendered/{installed_path.name}",
                    )
                )
                print(f"  DRIFT: {installed_path.name}")
                sys.stdout.writelines(diff)
                drifted.append(installed_path.name)

    if drifted:
        print(f"\n{len(drifted)} plist(s) differ from manifest-rendered output.")
        return 1

    print("OK: all installed plists match the manifest.")
    return 0


def _default_repo_root() -> Path:
    """Repo root = two directories up from this file (ops/launchd/render_plists.py)."""
    return Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render SMA launchd plists from schedule manifest."
    )
    parser.add_argument(
        "--out-dir", type=Path, default=None, help="Output directory (default: ops/launchd/)"
    )
    parser.add_argument(
        "--check", action="store_true", help="Check drift against installed plists; exit 1 on drift"
    )
    args = parser.parse_args(argv)

    repo_root = _default_repo_root()
    venv = repo_root / ".venv"
    home_dir = Path.home()

    if args.check:
        return _check_drift(repo_root=repo_root, venv=venv, home_dir=home_dir)

    out_dir = args.out_dir if args.out_dir is not None else repo_root / "ops" / "launchd"
    render_all(out_dir=out_dir, repo_root=repo_root, venv=venv, home_dir=home_dir)
    print(f"Rendered 18 plists to {out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
