"""Integration test: watchdog detects a missed job and calls launchctl kickstart.

Higher-fidelity than the unit tests: exercises the full
launchctl print -> state-check -> kickstart decision pipeline with a real
mocked subprocess flow, real sentinel reads, and the real schedule module.

Scenario: ingest deadline has passed (21:00 ET Thursday), sentinel absent,
launchctl print returns "state = not running". Watchdog must:
1. Call ``launchctl print gui/<uid>/com.sma.ingest.daily`` to check state.
2. Call ``launchctl kickstart -p gui/<uid>/com.sma.ingest.daily``.

A second scenario tests that when a sentinel IS present the subprocess is
never called (no launchctl traffic at all for that job).
"""

from __future__ import annotations

from datetime import date, datetime
from unittest.mock import MagicMock, patch

import pytest

from sma import schedule as sched
from sma import watchdog as wd
from sma.sentinels import write_sentinel

# ---- fixtures ---------------------------------------------------------------


@pytest.fixture(autouse=True)
def _pin_launchd_adapter(monkeypatch):
    """Pin the launchd adapter regardless of host OS (CI runs ubuntu-latest,
    which would otherwise default to SystemdAdapter) -- this suite asserts
    launchd-specific launchctl argv, per its own docstring above. See
    sma/sched_adapter.py."""
    monkeypatch.setenv("SMA_SCHED_ADAPTER", "launchd")


# ---- helpers ---------------------------------------------------------------


def _fake_now_past_deadline() -> datetime:
    """21:00 ET on 2026-04-30 (Thursday): past ingest deadline (20:30 ET)."""
    return datetime(2026, 4, 30, 21, 0, tzinfo=sched.NY_TZ)


def _fake_now_before_deadline() -> datetime:
    """19:00 ET on 2026-04-30 (Thursday): before ingest deadline (20:30 ET)."""
    return datetime(2026, 4, 30, 19, 0, tzinfo=sched.NY_TZ)


def _launchctl_mock(state: str, kickstart_rc: int = 0):
    """Return a subprocess.run-compatible mock that handles print + kickstart."""

    def _run(args, **kwargs):
        if args[1] == "print":
            return MagicMock(returncode=0, stdout=f"state = {state}\n", stderr="")
        if args[1] == "kickstart":
            return MagicMock(returncode=kickstart_rc, stdout="12345", stderr="")
        return MagicMock(returncode=0, stdout="", stderr="")

    return _run


# ---- tests -----------------------------------------------------------------


@patch("sma.sched_adapter.subprocess.run")
def test_watchdog_kickstarts_missed_idle_job(mock_run, tmp_path, monkeypatch):
    """Past deadline + sentinel absent + state 'not running' -> kickstart fired."""
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path / "sentinels"))
    monkeypatch.setattr(wd, "_now", _fake_now_past_deadline)
    mock_run.side_effect = _launchctl_mock("not running")

    rc = wd.check()

    # At least one kickstart call for ingest
    kickstart_calls = [c for c in mock_run.call_args_list if "kickstart" in c.args[0]]
    assert len(kickstart_calls) >= 1, "expected at least one kickstart call"
    ingest_kicked = any("com.sma.ingest.daily" in str(c.args[0]) for c in kickstart_calls)
    assert ingest_kicked, f"expected com.sma.ingest.daily to be kicked; calls: {kickstart_calls}"
    # rc is 0 because kickstart succeeded (rc=0 from mock)
    assert rc == 0


@patch("sma.sched_adapter.subprocess.run")
def test_watchdog_no_subprocess_when_sentinel_present(mock_run, tmp_path, monkeypatch):
    """When the ingest sentinel exists the watchdog must not touch launchctl for it."""
    sentinel_dir = tmp_path / "sentinels"
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(sentinel_dir))
    monkeypatch.setattr(wd, "_now", _fake_now_past_deadline)
    mock_run.side_effect = _launchctl_mock("not running")

    today = date(2026, 4, 30)
    write_sentinel(
        label="com.sma.ingest.daily",
        asof=today,
        payload={
            "label": "com.sma.ingest.daily",
            "asof": today.isoformat(),
            "run_id": 1,
            "quality": {"passed": True, "blocking_failures": []},
        },
    )

    wd.check()

    # No launchctl call should mention ingest specifically
    ingest_calls = [c for c in mock_run.call_args_list if "com.sma.ingest.daily" in str(c.args[0])]
    assert ingest_calls == [], (
        f"ingest sentinel present; expected 0 launchctl calls for it; got: {ingest_calls}"
    )


@patch("sma.sched_adapter.subprocess.run")
def test_watchdog_skips_running_job(mock_run, tmp_path, monkeypatch):
    """Past deadline + sentinel absent + state 'running' -> no kickstart."""
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path / "sentinels"))
    monkeypatch.setattr(wd, "_now", _fake_now_past_deadline)
    mock_run.side_effect = _launchctl_mock("running")

    wd.check()

    kickstart_calls = [c for c in mock_run.call_args_list if "kickstart" in c.args[0]]
    assert kickstart_calls == [], (
        f"job is 'running'; expected no kickstart calls; got: {kickstart_calls}"
    )


@patch("sma.sched_adapter.subprocess.run")
def test_watchdog_skips_waiting_job(mock_run, tmp_path, monkeypatch):
    """Past deadline + sentinel absent + state 'waiting' -> no kickstart."""
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path / "sentinels"))
    monkeypatch.setattr(wd, "_now", _fake_now_past_deadline)
    mock_run.side_effect = _launchctl_mock("waiting")

    wd.check()

    kickstart_calls = [c for c in mock_run.call_args_list if "kickstart" in c.args[0]]
    assert kickstart_calls == [], (
        f"job is 'waiting'; expected no kickstart calls; got: {kickstart_calls}"
    )


@patch("sma.sched_adapter.subprocess.run")
def test_watchdog_returns_nonzero_on_kickstart_failure(mock_run, tmp_path, monkeypatch):
    """If launchctl kickstart returns non-zero, watchdog returns 1."""
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path / "sentinels"))
    monkeypatch.setattr(wd, "_now", _fake_now_past_deadline)
    mock_run.side_effect = _launchctl_mock("not running", kickstart_rc=1)

    rc = wd.check()
    assert rc == 1, "expected rc=1 when kickstart fails"


@patch("sma.sched_adapter.subprocess.run")
def test_watchdog_does_not_fire_before_deadline(mock_run, tmp_path, monkeypatch):
    """Before the ingest deadline, no launchctl calls for ingest."""
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path / "sentinels"))
    monkeypatch.setattr(wd, "_now", _fake_now_before_deadline)

    wd.check()

    ingest_calls = [c for c in mock_run.call_args_list if "com.sma.ingest.daily" in str(c.args[0])]
    assert ingest_calls == [], (
        f"before ingest deadline; expected 0 launchctl calls for it; got: {ingest_calls}"
    )


@patch("sma.sched_adapter.subprocess.run")
def test_watchdog_print_precedes_kickstart(mock_run, tmp_path, monkeypatch):
    """launchctl print must be called before kickstart for each missed job.

    This verifies the state-check -> kickstart ordering in the implementation,
    not just that both calls happen.
    """
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path / "sentinels"))
    monkeypatch.setattr(wd, "_now", _fake_now_past_deadline)
    call_log: list[str] = []

    def _ordered_run(args, **kwargs):
        op = args[1]
        call_log.append(op)
        if op == "print":
            return MagicMock(returncode=0, stdout="state = not running\n", stderr="")
        if op == "kickstart":
            return MagicMock(returncode=0, stdout="12345", stderr="")
        return MagicMock(returncode=0, stdout="", stderr="")

    mock_run.side_effect = _ordered_run

    wd.check()

    # For each kickstart there must have been a preceding print
    print_positions = [i for i, op in enumerate(call_log) if op == "print"]
    kickstart_positions = [i for i, op in enumerate(call_log) if op == "kickstart"]
    assert kickstart_positions, "expected at least one kickstart"
    for ks_pos in kickstart_positions:
        preceding_prints = [p for p in print_positions if p < ks_pos]
        assert preceding_prints, (
            f"kickstart at position {ks_pos} had no preceding print in call_log={call_log}"
        )
