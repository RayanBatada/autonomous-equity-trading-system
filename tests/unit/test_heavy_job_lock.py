"""heavy_job_lock: serializes the memory-heavy jobs (retrain / autoresearch),
but is a SEPARATE lock from writer_lock so it never blocks evening trading.
Cross-process serialization (the property that matters under launchd) is
verified directly with two processes (Codex review 2026-06-30)."""
import multiprocessing as mp
import time

import pytest

from sma.locks import WriterLockTimeoutError, heavy_job_lock, writer_lock


def _acquire_and_release(lock_path, **kwargs):
    with heavy_job_lock(lock_path=lock_path, **kwargs):
        pass


def test_serializes_two_heavy_jobs(tmp_path):
    # While one heavy job holds the lock, a second cannot run concurrently —
    # exactly what prevents two training-set builds from thrashing RAM.
    lock = tmp_path / ".heavy.lock"
    with heavy_job_lock(label="first", lock_path=lock), pytest.raises(WriterLockTimeoutError):
        _acquire_and_release(lock, label="second", timeout_s=0.2)


def test_independent_of_writer_lock(tmp_path):
    # Holding heavy_job_lock must NOT block the DB writer_lock — evening trading
    # jobs (ingest/predict/decide) must never wait on a long retrain build.
    with heavy_job_lock(label="heavy", lock_path=tmp_path / ".heavy.lock"), writer_lock(
        lock_path=tmp_path / ".writer.lock", label="evening", timeout_s=0.5
    ):
        pass  # acquires immediately — a different lock file


def test_releases_on_exit(tmp_path):
    # After a heavy job finishes, the next one acquires without waiting.
    lock = tmp_path / ".heavy.lock"
    with heavy_job_lock(label="run1", lock_path=lock):
        pass
    with heavy_job_lock(label="run2", lock_path=lock, timeout_s=0.5):
        pass  # no timeout — the first run released it


def _hold_until_signaled(lock_path_str, ready_str, release_str):
    from pathlib import Path

    from sma.locks import heavy_job_lock
    with heavy_job_lock(label="child", lock_path=Path(lock_path_str)):
        Path(ready_str).write_text("held")
        for _ in range(400):  # hold up to ~20s awaiting the release signal
            if Path(release_str).exists():
                return
            time.sleep(0.05)


def test_serializes_across_processes(tmp_path):
    # The property that matters under launchd: two SEPARATE processes cannot both
    # hold the heavy lock. Pins the heavy lock's path wiring across processes.
    lock = tmp_path / ".heavy.lock"
    ready, release = tmp_path / "ready", tmp_path / "release"
    proc = mp.get_context("spawn").Process(
        target=_hold_until_signaled, args=(str(lock), str(ready), str(release))
    )
    proc.start()
    try:
        for _ in range(200):  # wait until the child actually holds it
            if ready.exists():
                break
            time.sleep(0.05)
        assert ready.exists(), "child never acquired the heavy lock"
        with pytest.raises(WriterLockTimeoutError):
            _acquire_and_release(lock, label="parent", timeout_s=0.3)
    finally:
        release.write_text("go")
        proc.join(timeout=15)
