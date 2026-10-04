"""Render systemd .service/.timer units from the schedule manifest.

The Linux twin of ops/launchd/render_plists.py — same manifest
(src/sma/schedule.py), same "renderer, not hand-written units" philosophy.
Written as pre-work for the Mac -> always-on-host migration
(~/StockMarket/host-migration-runbook.md Section 2b), which already
anticipated this file (see ops/launchd/README.md's "Linux migration" note).
This module renders plain text and never touches systemd itself, so it runs
fine in tests on macOS.

Usage:
    python -m ops.systemd.render_units [--out-dir DIR] [--check] [--installed-dir DIR]

--out-dir DIR       : write units to DIR (default: ops/systemd/ relative to repo root)
--check             : render to a temp dir, diff against --installed-dir's sma-*
                       units, exit 1 on drift
--installed-dir DIR : where installed units live for --check (default:
                       ~/.config/systemd/user, the recommended --user install path;
                       pass /etc/systemd/system if installed as system units instead)

TIMEZONE: every job in SCHEDULE fires on ET wall-clock time. systemd's
OnCalendar= (like launchd's StartCalendarInterval) fires on the HOST's system
local time — NOT on any TZ= set in a unit's [Service] section. Modern systemd
(systemd.time(7)) does support an explicit per-line timezone suffix (e.g.
"OnCalendar=Mon..Fri 18:30:00 America/New_York"), but the target host's exact
systemd version isn't pinned yet (Hetzner CX32, "Ubuntu LTS or Debian stable,
either is fine" per the runbook) and embedding timezones adds a second,
untested correctness surface on top of the one that already had a real
production incident (2026-08-12 boot storm) on this exact "does the wake
happen at the intended wall-clock moment" question. This generator therefore
follows the runbook's own resolution (Section 1c) instead: assume the HOST's
system timezone is set to America/New_York (`sudo timedatectl set-timezone
America/New_York` at provision time, Phase 1 step 2) and keep
Environment=TZ=America/New_York in every unit as defense-in-depth (pins the
Python process's own clock, matching the identical existing comment in
ops/launchd/render_plists.py) — it does NOT affect when the timer fires.
This assumption is repeated in every generated unit's header, not just here.

INSTALL SCOPE: units are written as install-mode-agnostic text — no User=/
Group= directive, and [Install] uses WantedBy=timers.target / default.target
(valid under both `systemctl --user` and system-scope `systemctl`). Which
scope to install under is an ops decision documented in ops/systemd/README.md,
not baked into the generator.
"""

from __future__ import annotations

import argparse
import difflib
import sys
import tempfile
from pathlib import Path

from ops.launchd.render_plists import _program_args, _short_name

# ---------------------------------------------------------------------------
# Shared header, repeated verbatim at the top of every generated unit file.
# Keep this in sync with the module docstring's TIMEZONE paragraph above.
# ---------------------------------------------------------------------------
_TZ_ASSUMPTION = """\
# GENERATED FILE — do not hand-edit. Source: src/sma/schedule.py, rendered by
# ops/systemd/render_units.py (the systemd twin of ops/launchd/render_plists.py).
#
# TIMEZONE: OnCalendar= fires on the HOST's system local time, not on TZ= set
# below. This unit assumes the host's system timezone is set to
# America/New_York (`sudo timedatectl set-timezone America/New_York`); see
# ops/systemd/README.md. Environment=TZ=America/New_York is defense-in-depth
# only (pins the Python process clock, matches ops/launchd/render_plists.py's
# identical comment) — it does NOT control when the timer fires.
"""

_BOOT_CATCHUP_NOTE = """\
#
# Persistent=true reproduces launchd's missed-job catch-up: a job whose
# scheduled time passed while the host was off fires once at next boot. This
# is not new risk — it is parity with what launchd already did in
# production, including the 2026-08-12 incident where the Mac was down all
# day, rebooted 20:11 ET, and every missed job fired at boot. That event is
# exactly why this codebase's application-layer guards (dead-zone checks in
# sma/live/__main__.py, the ingest-sentinel gate in sma/agents/__main__.py,
# writer_lock/heavy_job_lock serialization, sentinel idempotency) were
# hardened — they are what makes Persistent=true safe here, not the other
# way around. See tests/unit/ops/test_render_plists.py::
# test_only_watchdog_and_dashboard_run_at_load for the launchd-side history.
"""


def _strip_caffeinate(args: list[str]) -> list[str]:
    """Drop render_plists.py's caffeinate wrapper (`/usr/bin/caffeinate -s|-i`).

    There is no Linux equivalent to reach for — nothing to keep awake on a
    host that never sleeps (host-migration-runbook.md Section 2b). The
    remaining args (starting with the venv python or streamlit binary) are
    reused verbatim so the two OS's job flags (--use-theses,
    --demean-labels, --n-configs, ...) can never drift apart.
    """
    assert args[0] == "/usr/bin/caffeinate", f"expected caffeinate wrapper, got {args!r}"
    return args[2:]


_DAY_ABBREV = {1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat", 7: "Sun"}
_MON_FRI = (1, 2, 3, 4, 5)


def _compress_weekdays(days: tuple) -> str:
    """Compress a tuple of sma.schedule.Day values into a systemd weekday token.

    Day enum values are 1=Mon..6=Sat, 7=Sun (sma.schedule.Day), already the
    natural Mon-first week order systemd expects — no Sunday-wraparound
    fixup needed here (unlike the launchd renderer's _launchd_weekday, which
    must remap Day.SUN=7 to launchd's Weekday=0).
    """
    ordered = sorted(int(d) for d in days)
    if tuple(ordered) == _MON_FRI:
        return "Mon..Fri"
    if len(ordered) == 1:
        return _DAY_ABBREV[ordered[0]]
    return ",".join(_DAY_ABBREV[d] for d in ordered)


def _on_calendar_spec(job: object) -> str:
    """Translate a JobSchedule's days + fire_time_et into an OnCalendar= value."""
    time_str = f"{job.fire_time_et.hour:02d}:{job.fire_time_et.minute:02d}:00"
    if len(job.days) == 7:
        # Every day: bare date-glob, no weekday token — matches the
        # host-migration runbook's own worked example for backup.daily.
        return f"*-*-* {time_str}"
    return f"{_compress_weekdays(job.days)} {time_str}"


def _render_service(
    *,
    description: str,
    exec_start: str,
    repo_root: Path,
    log_name: str,
    log_dir: Path,
    oneshot: bool,
    extra_service_lines: str = "",
) -> str:
    kind = "oneshot" if oneshot else "simple"
    lines = [
        _TZ_ASSUMPTION,
        "[Unit]",
        f"Description=SMA {description}",
        "",
        "[Service]",
        f"Type={kind}",
        f"WorkingDirectory={repo_root}",
        f"EnvironmentFile={repo_root / '.env'}",
        "Environment=TZ=America/New_York",
        f"ExecStart={exec_start}",
        f"StandardOutput=append:{log_dir}/{log_name}.out.log",
        f"StandardError=append:{log_dir}/{log_name}.err.log",
    ]
    if extra_service_lines:
        lines.append(extra_service_lines)
    return "\n".join(lines) + "\n"


def _render_timer(*, description: str, on_calendar_lines: list[str], onboot_sec: str = "") -> str:
    lines = [
        _TZ_ASSUMPTION,
        _BOOT_CATCHUP_NOTE,
        "[Unit]",
        f"Description=SMA {description} schedule",
        "",
        "[Timer]",
    ]
    lines.extend(f"OnCalendar={spec}" for spec in on_calendar_lines)
    if onboot_sec:
        lines.append(f"OnBootSec={onboot_sec}")
    lines.append("Persistent=true")
    lines.extend(["", "[Install]", "WantedBy=timers.target"])
    return "\n".join(lines) + "\n"


def _render_schedule_unit_pair(
    job: object, *, repo_root: Path, venv: Path, log_dir: Path
) -> tuple[str, str]:
    from sma.schedule import JobSchedule

    assert isinstance(job, JobSchedule)
    venv_python = str(venv / "bin" / "python")
    short = _short_name(job.label)
    description = job.label.removeprefix("com.sma.")
    exec_start = " ".join(_strip_caffeinate(_program_args(job.label, venv_python)))

    service = _render_service(
        description=description,
        exec_start=exec_start,
        repo_root=repo_root,
        log_name=short,
        log_dir=log_dir,
        oneshot=True,
    )
    timer = _render_timer(
        description=description,
        on_calendar_lines=[_on_calendar_spec(job)],
    )
    return service, timer


def _render_watchdog_unit_pair(*, repo_root: Path, venv: Path, log_dir: Path) -> tuple[str, str]:
    """Watchdog: checkpoints 10/13/16 + 19-23 ET + 0:30/1:30 overnight (same
    schedule as ops/launchd/render_plists.py's _render_watchdog_plist — see
    that function's docstring for why each checkpoint exists).

    launchd's RunAtLoad=true gives the watchdog a fast recovery pass at
    boot/wake; there is no service-level equivalent for a oneshot systemd
    unit (only a timer can fire it), so OnBootSec= reproduces it on the timer
    instead.
    """
    venv_python = str(venv / "bin" / "python")
    exec_start = " ".join(_strip_caffeinate(_program_args("com.sma.watchdog", venv_python)))
    service = _render_service(
        description="watchdog",
        exec_start=exec_start,
        repo_root=repo_root,
        log_name="watchdog",
        log_dir=log_dir,
        oneshot=True,
    )
    timer = _render_timer(
        description="watchdog",
        on_calendar_lines=[
            "*-*-* 10,13,16,19,20,21,22,23:00:00",
            # Overnight (2026-08-05 parity with the launchd plist): half-past
            # so they sit inside predict's and decide's late-kick windows
            # rather than landing exactly on a boundary.
            "*-*-* 00,01:30:00",
        ],
        onboot_sec="2min",
    )
    return service, timer


def _render_dashboard_service(*, repo_root: Path, venv: Path, log_dir: Path) -> str:
    """Dashboard: long-running Streamlit daemon. Restart=always + Type=simple
    is the systemd analog of launchd's KeepAlive=true + RunAtLoad=true. No
    caffeinate wrapper (nothing to keep awake) and no .timer — it is enabled
    directly, not scheduled."""
    venv_python = str(venv / "bin" / "python")
    exec_start = " ".join(_strip_caffeinate(_program_args("com.sma.dashboard", venv_python)))
    extra = "\n".join(
        [
            "Restart=always",
            "RestartSec=10",
            "",
            "[Install]",
            "WantedBy=default.target",
        ]
    )
    return _render_service(
        description="dashboard",
        exec_start=exec_start,
        repo_root=repo_root,
        log_name="dashboard",
        log_dir=log_dir,
        oneshot=False,
        extra_service_lines=extra,
    )


def render_all(
    out_dir: Path,
    repo_root: Path,
    venv: Path,
    home_dir: Path,
    log_dir: Path | None = None,
) -> None:
    """Render all SMA systemd units into out_dir.

    1. 13 SCHEDULE + 3 OPTIONAL_SCHEDULE jobs -> sma-<short>.service +
       sma-<short>.timer (32 files).
    2. sma-watchdog.service + sma-watchdog.timer (hourly checkpoints, not in SCHEDULE).
    3. sma-dashboard.service only (long-running daemon, no timer).
    35 files total, mirroring the 18 launchd plists 1:1 at the job level.
    """
    from sma.schedule import OPTIONAL_SCHEDULE, SCHEDULE

    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = log_dir if log_dir is not None else home_dir / "logs" / "sma"

    # OPTIONAL_SCHEDULE units are rendered but ship disabled: nothing enables
    # their timers (see ops/systemd/README.md "Intraday sessions").
    for job in (*SCHEDULE, *OPTIONAL_SCHEDULE):
        short = _short_name(job.label)
        service, timer = _render_schedule_unit_pair(
            job, repo_root=repo_root, venv=venv, log_dir=log_dir
        )
        (out_dir / f"sma-{short}.service").write_text(service)
        (out_dir / f"sma-{short}.timer").write_text(timer)

    wd_service, wd_timer = _render_watchdog_unit_pair(
        repo_root=repo_root, venv=venv, log_dir=log_dir
    )
    (out_dir / "sma-watchdog.service").write_text(wd_service)
    (out_dir / "sma-watchdog.timer").write_text(wd_timer)

    dashboard = _render_dashboard_service(repo_root=repo_root, venv=venv, log_dir=log_dir)
    (out_dir / "sma-dashboard.service").write_text(dashboard)


def _check_drift(
    *, repo_root: Path, venv: Path, home_dir: Path, installed_dir: Path, log_dir: Path | None = None
) -> int:
    """Render to a tempdir, diff against installed_dir's sma-* units.

    Returns 0 if no drift, 1 on any difference (mirrors
    ops/launchd/render_plists.py's _check_drift).
    """
    if not installed_dir.is_dir():
        print(f"No installed dir found at {installed_dir}")
        return 1
    installed = list(installed_dir.glob("sma-*"))
    if not installed:
        print(f"No installed SMA units found in {installed_dir}")
        return 1

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        render_all(
            out_dir=tmp_dir, repo_root=repo_root, venv=venv, home_dir=home_dir, log_dir=log_dir
        )

        drifted: list[str] = []
        for installed_path in sorted(installed):
            if installed_path.is_dir():
                continue
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
        print(f"\n{len(drifted)} unit(s) differ from manifest-rendered output.")
        return 1

    print("OK: all installed units match the manifest.")
    return 0


def _default_repo_root() -> Path:
    """Repo root = two directories up from this file (ops/systemd/render_units.py)."""
    return Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render SMA systemd units from schedule manifest."
    )
    parser.add_argument(
        "--out-dir", type=Path, default=None, help="Output directory (default: ops/systemd/)"
    )
    parser.add_argument(
        "--check", action="store_true", help="Check drift against installed units; exit 1 on drift"
    )
    parser.add_argument(
        "--installed-dir",
        type=Path,
        default=None,
        help="Installed-units dir for --check (default: ~/.config/systemd/user)",
    )
    args = parser.parse_args(argv)

    repo_root = _default_repo_root()
    venv = repo_root / ".venv"
    home_dir = Path.home()

    if args.check:
        installed_dir = (
            args.installed_dir
            if args.installed_dir is not None
            else home_dir / ".config" / "systemd" / "user"
        )
        return _check_drift(
            repo_root=repo_root, venv=venv, home_dir=home_dir, installed_dir=installed_dir
        )

    out_dir = args.out_dir if args.out_dir is not None else repo_root / "ops" / "systemd"
    render_all(out_dir=out_dir, repo_root=repo_root, venv=venv, home_dir=home_dir)
    print(f"Rendered 35 unit files (18 jobs) to {out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
