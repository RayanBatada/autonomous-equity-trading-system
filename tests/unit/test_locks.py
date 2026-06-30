import multiprocessing as mp
import time as time_mod
from pathlib import Path

import pytest

from sma.locks import WriterLockTimeout, writer_lock


def _hold_lock(lock_path, hold_for_s, ready_event):
    with writer_lock(lock_path=Path(lock_path), label="holder", timeout_s=2.0):
        ready_event.set()
        time_mod.sleep(hold_for_s)


def test_lock_acquired_when_uncontended(tmp_path):
    lock_path = tmp_path / ".sma-writer.lock"
    with writer_lock(lock_path=lock_path, label="t1", timeout_s=1.0):
        # PID file must exist while held
        assert (lock_path.parent / ".sma-writer.pid").exists()


def test_pid_file_cleared_after_release(tmp_path):
    lock_path = tmp_path / ".sma-writer.lock"
    with writer_lock(lock_path=lock_path, label="t1", timeout_s=1.0):
        pass
    # PID file should be removed on release
    assert not (lock_path.parent / ".sma-writer.pid").exists()


def test_pid_file_contains_pid_and_label(tmp_path):
    lock_path = tmp_path / ".sma-writer.lock"
    with writer_lock(lock_path=lock_path, label="my-job", timeout_s=1.0):
        contents = (lock_path.parent / ".sma-writer.pid").read_text().strip()
        # format: "<pid> <label>"
        parts = contents.split(maxsplit=1)
        assert len(parts) == 2
        assert int(parts[0]) > 0
        assert parts[1] == "my-job"


def test_contention_raises_after_timeout(tmp_path):
    lock_path = tmp_path / ".sma-writer.lock"
    ready = mp.Event()
    holder = mp.Process(target=_hold_lock, args=(str(lock_path), 3.0, ready))
    holder.start()
    try:
        ready.wait(timeout=3.0)
        with (
            pytest.raises(WriterLockTimeout, match=r"holder:"),
            writer_lock(lock_path=lock_path, label="t2", timeout_s=0.5),
        ):
            pass
    finally:
        if holder.is_alive():
            holder.terminate()
        holder.join(timeout=5.0)


def test_empty_label_rejected(tmp_path):
    lock_path = tmp_path / ".sma-writer.lock"
    with (
        pytest.raises(ValueError, match="non-empty label"),
        writer_lock(lock_path=lock_path, label="", timeout_s=1.0),
    ):
        pass


def test_label_with_newline_rejected(tmp_path):
    lock_path = tmp_path / ".sma-writer.lock"
    with (
        pytest.raises(ValueError, match="must not contain newlines"),
        writer_lock(lock_path=lock_path, label="my\njob", timeout_s=1.0),
    ):
        pass


def test_label_with_carriage_return_rejected(tmp_path):
    lock_path = tmp_path / ".sma-writer.lock"
    with (
        pytest.raises(ValueError, match="must not contain newlines"),
        writer_lock(lock_path=lock_path, label="my\rjob", timeout_s=1.0),
    ):
        pass


def test_pid_path_derived_from_lock_path(tmp_path):
    """Custom lock_path gets its own PID file scoped to that lock (not shared)."""
    lock_path = tmp_path / "custom-name.lock"
    with writer_lock(lock_path=lock_path, label="custom", timeout_s=1.0):
        # PID file derived from lock_path via with_suffix(".pid")
        assert (tmp_path / "custom-name.pid").exists()
        # And the legacy hardcoded path does NOT exist
        assert not (tmp_path / ".sma-writer.pid").exists()


def test_lock_released_after_exception_in_block(tmp_path):
    lock_path = tmp_path / ".sma-writer.lock"
    with (
        pytest.raises(RuntimeError, match="boom"),
        writer_lock(lock_path=lock_path, label="t1", timeout_s=1.0),
    ):
        raise RuntimeError("boom")
    # After the exception, the next acquisition should succeed
    with writer_lock(lock_path=lock_path, label="t2", timeout_s=1.0):
        assert (lock_path.parent / ".sma-writer.pid").exists()
