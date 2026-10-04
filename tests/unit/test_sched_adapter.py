"""Scheduler-adapter seam: launchd (macOS, current production) vs systemd
--user (Linux, host-migration target). See host-migration-runbook.md Section
2c and ops/systemd/README.md's "Known remaining item" section for the
launchctl -> systemctl command equivalents this pins.

LaunchdAdapter tests regression-pin TODAY's exact watchdog.py commands
(pre-seam behavior) byte-for-byte. SystemdAdapter tests pin the documented
systemctl equivalents. get_adapter() selection tests cover the env override
and the sys.platform fallback.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from sma.sched_adapter import (
    LaunchdAdapter,
    SchedulerAdapter,
    SystemdAdapter,
    get_adapter,
)

UID = 501


# ---------------------------------------------------------------------------
# LaunchdAdapter — byte-identical to watchdog.py's pre-seam commands.
# ---------------------------------------------------------------------------


@patch("sma.sched_adapter.os.getuid", return_value=UID)
@patch("sma.sched_adapter.subprocess.run")
def test_launchd_state_command_regression_pin(mock_run, _uid):
    """Exact argv watchdog.py used to build inline for state checks."""
    mock_run.return_value = MagicMock(returncode=0, stdout="state = running\n", stderr="")
    LaunchdAdapter().state("com.sma.ingest.daily")
    mock_run.assert_called_once_with(
        ["launchctl", "print", f"gui/{UID}/com.sma.ingest.daily"],
        capture_output=True,
        text=True,
    )


@patch("sma.sched_adapter.os.getuid", return_value=UID)
@patch("sma.sched_adapter.subprocess.run")
def test_launchd_state_parses_running(mock_run, _uid):
    mock_run.return_value = MagicMock(returncode=0, stdout="state = running\n", stderr="")
    assert LaunchdAdapter().state("com.sma.ingest.daily") == "running"


@patch("sma.sched_adapter.os.getuid", return_value=UID)
@patch("sma.sched_adapter.subprocess.run")
def test_launchd_state_parses_waiting(mock_run, _uid):
    mock_run.return_value = MagicMock(returncode=0, stdout="state = waiting\n", stderr="")
    assert LaunchdAdapter().state("com.sma.ingest.daily") == "waiting"


@patch("sma.sched_adapter.os.getuid", return_value=UID)
@patch("sma.sched_adapter.subprocess.run")
def test_launchd_state_parses_not_running(mock_run, _uid):
    mock_run.return_value = MagicMock(returncode=0, stdout="state = not running\n", stderr="")
    assert LaunchdAdapter().state("com.sma.ingest.daily") == "not running"


@patch("sma.sched_adapter.os.getuid", return_value=UID)
@patch("sma.sched_adapter.subprocess.run")
def test_launchd_state_unknown_on_nonzero_rc(mock_run, _uid):
    mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="No such process")
    assert LaunchdAdapter().state("com.sma.ingest.daily") == "unknown"


@patch("sma.sched_adapter.os.getuid", return_value=UID)
@patch("sma.sched_adapter.subprocess.run")
def test_launchd_state_unknown_when_no_state_line(mock_run, _uid):
    mock_run.return_value = MagicMock(returncode=0, stdout="some other output\n", stderr="")
    assert LaunchdAdapter().state("com.sma.ingest.daily") == "unknown"


@patch("sma.sched_adapter.os.getuid", return_value=UID)
@patch("sma.sched_adapter.subprocess.run")
def test_launchd_kickstart_command_regression_pin(mock_run, _uid):
    """Exact argv watchdog.py used to build inline for the kickstart call."""
    mock_run.return_value = MagicMock(returncode=0, stdout="12345", stderr="")
    cp = LaunchdAdapter().kickstart("com.sma.ingest.daily")
    mock_run.assert_called_once_with(
        ["launchctl", "kickstart", "-p", f"gui/{UID}/com.sma.ingest.daily"],
        capture_output=True,
        text=True,
    )
    assert cp.returncode == 0
    assert cp.stdout == "12345"


def test_launchd_rekick_hint_matches_current_hardcoded_text():
    """Byte-identical to the strings currently hardcoded in
    live/__main__.py's _rekick_hint() and monitoring/__init__.py's
    stale-retrain page."""
    assert (
        LaunchdAdapter().rekick_hint("com.sma.model.predict.daily")
        == "launchctl kickstart -p gui/$UID/com.sma.model.predict.daily"
    )
    assert (
        LaunchdAdapter().rekick_hint("com.sma.model.retrain.weekly")
        == "launchctl kickstart -p gui/$UID/com.sma.model.retrain.weekly"
    )


# ---------------------------------------------------------------------------
# SystemdAdapter — documented systemctl --user equivalents.
# ---------------------------------------------------------------------------


@patch("sma.sched_adapter.subprocess.run")
def test_systemd_state_command_uses_documented_equivalent(mock_run):
    """ops/systemd/README.md: `systemctl --user show sma-<short>.service
    --property=ActiveState --value`."""
    mock_run.return_value = MagicMock(returncode=0, stdout="active\n", stderr="")
    SystemdAdapter().state("com.sma.ingest.daily")
    mock_run.assert_called_once_with(
        ["systemctl", "--user", "show", "sma-ingest.service", "--property=ActiveState", "--value"],
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(
    "active_state,expected",
    [
        ("active", "running"),
        ("activating", "running"),
        ("inactive", "not running"),
        ("failed", "not running"),
        ("deactivating", "unknown"),
        ("", "unknown"),
    ],
)
@patch("sma.sched_adapter.subprocess.run")
def test_systemd_state_vocabulary_normalizes_to_launchd_vocabulary(
    mock_run, active_state, expected
):
    """systemd's ActiveState values (active/activating/inactive/failed) do
    not match launchd's (running/waiting/not running/unknown) vocabulary —
    the adapter normalizes so watchdog.py's `state in ("running", "waiting")`
    skip-check needs zero per-OS branching (host-migration-runbook.md
    Section 2c)."""
    mock_run.return_value = MagicMock(returncode=0, stdout=f"{active_state}\n", stderr="")
    assert SystemdAdapter().state("com.sma.ingest.daily") == expected


@patch("sma.sched_adapter.subprocess.run")
def test_systemd_state_unknown_on_nonzero_rc(mock_run):
    mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="Unit not found")
    assert SystemdAdapter().state("com.sma.ingest.daily") == "unknown"


@patch("sma.sched_adapter.subprocess.run")
def test_systemd_kickstart_command_uses_documented_equivalent(mock_run):
    """ops/systemd/README.md: `systemctl --user start sma-<short>.service`
    (no `-p` — systemd's `start` has no launchd "kill and restart" nuance to
    express, per the migration runbook Section 2c)."""
    mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
    cp = SystemdAdapter().kickstart("com.sma.ingest.daily")
    mock_run.assert_called_once_with(
        ["systemctl", "--user", "start", "sma-ingest.service"],
        capture_output=True,
        text=True,
    )
    assert cp.returncode == 0


def test_systemd_unit_naming_matches_render_units_short_name():
    """Multi-segment labels (e.g. com.sma.live.decide.daily) must produce the
    exact same sma-<short>.service name ops/systemd/render_units.py renders
    -- both derive `_short_name` from the same ops/launchd/render_plists.py
    function, so the two can never name a job differently."""
    from ops.launchd.render_plists import _short_name

    label = "com.sma.live.decide.daily"
    with patch("sma.sched_adapter.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="active\n", stderr="")
        SystemdAdapter().state(label)
    called_unit = mock_run.call_args.args[0][3]
    assert called_unit == f"sma-{_short_name(label)}.service"
    assert called_unit == "sma-live.decide.service"


def test_systemd_rekick_hint():
    assert (
        SystemdAdapter().rekick_hint("com.sma.model.predict.daily")
        == "systemctl --user start sma-model.predict.service"
    )


# ---------------------------------------------------------------------------
# get_adapter() selection: env override wins, else sys.platform.
# ---------------------------------------------------------------------------


def test_get_adapter_env_override_launchd(monkeypatch):
    monkeypatch.setenv("SMA_SCHED_ADAPTER", "launchd")
    assert isinstance(get_adapter(), LaunchdAdapter)


def test_get_adapter_env_override_systemd(monkeypatch):
    monkeypatch.setenv("SMA_SCHED_ADAPTER", "systemd")
    assert isinstance(get_adapter(), SystemdAdapter)


def test_get_adapter_env_override_case_insensitive(monkeypatch):
    monkeypatch.setenv("SMA_SCHED_ADAPTER", "SystemD")
    assert isinstance(get_adapter(), SystemdAdapter)


def test_get_adapter_env_override_unrecognized_raises(monkeypatch):
    monkeypatch.setenv("SMA_SCHED_ADAPTER", "upstart")
    with pytest.raises(ValueError, match="upstart"):
        get_adapter()


def test_get_adapter_defaults_to_darwin_platform(monkeypatch):
    monkeypatch.delenv("SMA_SCHED_ADAPTER", raising=False)
    monkeypatch.setattr("sma.sched_adapter.sys.platform", "darwin")
    assert isinstance(get_adapter(), LaunchdAdapter)


def test_get_adapter_defaults_to_linux_platform(monkeypatch):
    monkeypatch.delenv("SMA_SCHED_ADAPTER", raising=False)
    monkeypatch.setattr("sma.sched_adapter.sys.platform", "linux")
    assert isinstance(get_adapter(), SystemdAdapter)


def test_adapters_implement_the_interface():
    assert isinstance(LaunchdAdapter(), SchedulerAdapter)
    assert isinstance(SystemdAdapter(), SchedulerAdapter)
