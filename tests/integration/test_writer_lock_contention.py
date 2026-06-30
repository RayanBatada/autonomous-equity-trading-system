"""Writer-lock contention integration test.

Two real ``mp.Process`` workers both try to acquire ``writer_lock`` on the
same lock file. One wins; the other must block until it times out.

This complements the unit tests in ``tests/unit/test_locks.py`` (which also
use mp.Process) by being explicitly labelled as an integration test and by
exercising the contention path from a clean test isolation perspective with a
shared tmp_path lock file that lives on the real filesystem.

The test verifies cross-process flock semantics:
1. ``holder`` acquires the lock and signals via a ``mp.Event``.
2. ``contender`` (in-process) attempts to acquire the same lock with a short
   timeout and must raise ``WriterLockTimeoutError``.
3. After ``holder`` terminates the lock is released and a new acquisition
   succeeds immediately.
"""

from __future__ import annotations

import multiprocessing as mp
import time as time_mod
from pathlib import Path

import pytest

from sma.locks import WriterLockTimeoutError, writer_lock

# ---------------------------------------------------------------------------
# Helper: runs in a subprocess; acquires the lock, signals ready, then sleeps
# ---------------------------------------------------------------------------


def _hold_lock(lock_path: str, hold_for_s: float, ready_event) -> None:
    """Acquire writer_lock, set the event, hold for hold_for_s, then exit."""
    with writer_lock(lock_path=Path(lock_path), label="holder", timeout_s=2.0):
        ready_event.set()
        time_mod.sleep(hold_for_s)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_two_processes_contend_for_writer_lock(tmp_path):
    """Second process times out when first holds the lock."""
    lock_path = tmp_path / ".sma-writer.lock"
    ready = mp.Event()
    holder = mp.Process(target=_hold_lock, args=(str(lock_path), 3.0, ready))
    holder.start()
    try:
        assert ready.wait(timeout=5.0), "holder process never signalled ready"

        # Contender in this process: must raise because holder owns the lock.
        with (
            pytest.raises(WriterLockTimeoutError),
            writer_lock(lock_path=lock_path, label="contender", timeout_s=0.3),
        ):
            pass
    finally:
        if holder.is_alive():
            holder.terminate()
        holder.join(timeout=5.0)


def test_lock_released_after_holder_exits(tmp_path):
    """Once the holding process exits, a fresh acquisition succeeds immediately."""
    lock_path = tmp_path / ".sma-writer.lock"
    ready = mp.Event()
    # Use a very short hold time so the holder exits quickly.
    holder = mp.Process(target=_hold_lock, args=(str(lock_path), 0.1, ready))
    holder.start()
    try:
        assert ready.wait(timeout=5.0), "holder process never signalled ready"
        holder.join(timeout=5.0)
        assert not holder.is_alive(), "holder did not exit within timeout"
    finally:
        if holder.is_alive():
            holder.terminate()
            holder.join(timeout=3.0)

    # Now no one holds the lock; acquisition must succeed.
    with writer_lock(lock_path=lock_path, label="successor", timeout_s=1.0):
        assert lock_path.with_suffix(".pid").exists()


def test_contention_timeout_error_message_names_holder(tmp_path):
    """WriterLockTimeoutError message must reference the holder's label."""
    lock_path = tmp_path / ".sma-writer.lock"
    ready = mp.Event()
    holder = mp.Process(target=_hold_lock, args=(str(lock_path), 3.0, ready))
    holder.start()
    try:
        assert ready.wait(timeout=5.0), "holder process never signalled ready"

        with (
            pytest.raises(WriterLockTimeoutError, match="holder:"),
            writer_lock(lock_path=lock_path, label="late-comer", timeout_s=0.3),
        ):
            pass
    finally:
        if holder.is_alive():
            holder.terminate()
        holder.join(timeout=5.0)


def test_pid_file_absent_after_lock_released(tmp_path):
    """After the holder exits and the lock is released, the PID file is gone."""
    lock_path = tmp_path / ".sma-writer.lock"
    pid_path = lock_path.with_suffix(".pid")
    ready = mp.Event()
    holder = mp.Process(target=_hold_lock, args=(str(lock_path), 0.05, ready))
    holder.start()
    try:
        assert ready.wait(timeout=5.0), "holder process never signalled ready"
        # While held, PID file should exist.
        assert pid_path.exists(), "PID file must exist while lock is held"
        holder.join(timeout=5.0)
    finally:
        if holder.is_alive():
            holder.terminate()
            holder.join(timeout=3.0)

    # After release: PID file must be cleaned up.
    # The pid file is removed by the context manager on exit; give a tiny grace
    # period for the OS to propagate the unlink from the subprocess.
    deadline = time_mod.monotonic() + 1.0
    while time_mod.monotonic() < deadline and pid_path.exists():
        time_mod.sleep(0.01)
    assert not pid_path.exists(), "PID file must be absent after lock is released"
