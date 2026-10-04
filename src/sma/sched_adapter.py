"""Scheduler adapter seam: the OS-level job-state/kick calls that
`src/sma/watchdog.py` (and human-facing alert text in
`src/sma/live/__main__.py` and `src/sma/monitoring/__init__.py`) need, so the
same Python logic runs against launchd (macOS, current production) or
systemd --user (Linux, the host-migration target) without any of those
modules knowing which OS it's on.

Pre-work for `~/StockMarket/host-migration-runbook.md` Section 2c and the
"Known remaining item" in `ops/systemd/README.md` -- the two functional
`launchctl` calls `watchdog.py` used to shell out to inline, plus the two
cosmetic hardcoded-launchctl-text spots in human-facing pages.

Selected by `sys.platform` (`"darwin"` -> Launchd, else -> Systemd),
overridable via the `SMA_SCHED_ADAPTER` env var (`"launchd"` | `"systemd"`,
case-insensitive) so tests -- and a manual dry run against a scratch systemd
unit, per the runbook -- can force either adapter regardless of the host
they actually run on.

State vocabulary: both adapters normalize to launchd's own vocabulary --
`"running"`, `"waiting"`, `"not running"`, `"unknown"` -- so
`watchdog.py`'s `state in ("running", "waiting")` skip-check needs zero
per-OS branching. systemd has no "waiting" concept for a plain
`Type=oneshot` unit (no queued-but-not-yet-fired state distinct from
"about to run"), so `SystemdAdapter.state()` never returns it.
"""

from __future__ import annotations

import os
import subprocess
import sys
from abc import ABC, abstractmethod


class SchedulerAdapter(ABC):
    """OS-scheduler seam used by watchdog.py and human-facing alert text."""

    @abstractmethod
    def state(self, label: str) -> str:
        """Return the job's current state: one of "running", "waiting",
        "not running", "unknown"."""

    @abstractmethod
    def kickstart(self, label: str) -> subprocess.CompletedProcess:
        """Force-start `label` now. Returns the underlying scheduler
        command's CompletedProcess (rc/stdout/stderr)."""

    def installed(self, label: str) -> bool:
        """True only when `label` is installed AND loaded/enabled in the OS
        scheduler. Used by the watchdog for OPTIONAL_SCHEDULE jobs, which ship
        disabled and must never page while they are. Conservative: any doubt
        is False."""
        return False

    @abstractmethod
    def rekick_hint(self, label: str) -> str:
        """One-line, copy-pasteable command that re-kicks `label` by hand --
        used in human-facing alert text (live/__main__.py's `_rekick_hint`,
        monitoring/__init__.py's stale-retrain page)."""


class LaunchdAdapter(SchedulerAdapter):
    """macOS launchd. Byte-identical to watchdog.py's pre-seam inline
    commands (regression-pinned in tests/unit/test_sched_adapter.py) --
    this IS current production behavior, unchanged."""

    def state(self, label: str) -> str:
        target = f"gui/{os.getuid()}/{label}"
        cp = subprocess.run(
            ["launchctl", "print", target],
            capture_output=True,
            text=True,
        )
        if cp.returncode != 0:
            return "unknown"
        for line in cp.stdout.splitlines():
            line = line.strip()
            if line.startswith("state ="):
                return line.split("=", 1)[1].strip()
        return "unknown"

    def kickstart(self, label: str) -> subprocess.CompletedProcess:
        target = f"gui/{os.getuid()}/{label}"
        return subprocess.run(
            ["launchctl", "kickstart", "-p", target],
            capture_output=True,
            text=True,
        )

    def rekick_hint(self, label: str) -> str:
        return f"launchctl kickstart -p gui/$UID/{label}"

    def installed(self, label: str) -> bool:
        from pathlib import Path

        plist = Path.home() / "Library" / "LaunchAgents" / f"{label}.plist"
        if not plist.exists():
            return False
        cp = subprocess.run(
            ["launchctl", "print", f"gui/{os.getuid()}/{label}"],
            capture_output=True,
            text=True,
        )
        return cp.returncode == 0


# ActiveState values systemd reports for a plain `Type=oneshot` unit (no
# RemainAfterExit) -- ops/systemd/README.md's "Known remaining item" and
# host-migration-runbook.md Section 2c. Anything not listed (e.g.
# "deactivating", "reloading", an empty string) maps to "unknown", same as a
# non-zero rc.
_SYSTEMD_STATE_MAP: dict[str, str] = {
    "active": "running",
    "activating": "running",
    "inactive": "not running",
    "failed": "not running",
}


def _systemd_unit(label: str) -> str:
    """`com.sma.ingest.daily` -> `sma-ingest.service`. Reuses `_short_name`
    from ops/launchd/render_plists.py -- the same function
    ops/systemd/render_units.py imports directly -- so the watchdog and the
    installed unit files can never name a job differently."""
    from ops.launchd.render_plists import _short_name

    return f"sma-{_short_name(label)}.service"


class SystemdAdapter(SchedulerAdapter):
    """Linux systemd --user twin, per ops/systemd/README.md's "Known
    remaining item" and host-migration-runbook.md Section 2c."""

    def state(self, label: str) -> str:
        unit = _systemd_unit(label)
        cp = subprocess.run(
            ["systemctl", "--user", "show", unit, "--property=ActiveState", "--value"],
            capture_output=True,
            text=True,
        )
        if cp.returncode != 0:
            return "unknown"
        return _SYSTEMD_STATE_MAP.get(cp.stdout.strip(), "unknown")

    def kickstart(self, label: str) -> subprocess.CompletedProcess:
        # No `-p` flag: systemd's `start` has no launchd "kill and restart"
        # nuance to express (host-migration-runbook.md Section 2c).
        unit = _systemd_unit(label)
        return subprocess.run(
            ["systemctl", "--user", "start", unit],
            capture_output=True,
            text=True,
        )

    def rekick_hint(self, label: str) -> str:
        return f"systemctl --user start {_systemd_unit(label)}"

    def installed(self, label: str) -> bool:
        timer = _systemd_unit(label).removesuffix(".service") + ".timer"
        cp = subprocess.run(
            ["systemctl", "--user", "is-enabled", timer],
            capture_output=True,
            text=True,
        )
        return cp.returncode == 0 and cp.stdout.strip() == "enabled"


_ADAPTERS: dict[str, type[SchedulerAdapter]] = {
    "launchd": LaunchdAdapter,
    "systemd": SystemdAdapter,
}


def get_adapter() -> SchedulerAdapter:
    """Pick the scheduler adapter.

    `SMA_SCHED_ADAPTER` env var (`"launchd"` | `"systemd"`, case-insensitive)
    wins if set -- for tests, and for a manual dry run against a scratch
    systemd unit per the migration runbook. Otherwise `sys.platform`:
    `"darwin"` -> launchd, anything else -> systemd.
    """
    override = os.environ.get("SMA_SCHED_ADAPTER")
    if override:
        try:
            return _ADAPTERS[override.strip().lower()]()
        except KeyError:
            raise ValueError(
                f"SMA_SCHED_ADAPTER={override!r} unrecognized; expected one of {sorted(_ADAPTERS)}"
            ) from None
    if sys.platform == "darwin":
        return LaunchdAdapter()
    return SystemdAdapter()
