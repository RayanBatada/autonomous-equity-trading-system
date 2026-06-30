"""File-based exclusive writer lock around DuckDB-writable sections.

Acquires fcntl.LOCK_EX | LOCK_NB; retries with exponential backoff up to
timeout_s. While held, writes the holder's PID + label to a sibling
.pid file so contention can be debugged. Releases on exit (normal
or exception).

Used by every scheduled writable job: ingest, predict, agents, decide,
reconcile, stop_loss_sweep, retrain. Single writer at a time across the
whole system.

PID file format (stable contract; consumed by Store.connect() in Task 4):

    "<pid> <label>\\n"

The path is derived from the lock_path via .with_suffix(".pid"), so each
distinct lock has its own forensic file.
"""

from __future__ import annotations

import errno
import fcntl
import os
import time
from contextlib import contextmanager
from pathlib import Path

from loguru import logger

__all__ = [
    "DEFAULT_LOCK_PATH",
    "HEAVY_JOB_LOCK_PATH",
    "WriterLockTimeout",
    "WriterLockTimeoutError",
    "heavy_job_lock",
    "writer_lock",
]


class WriterLockTimeoutError(Exception):
    pass


# Alias so callers can use either name; prefer the Error-suffix form.
WriterLockTimeout = WriterLockTimeoutError


DEFAULT_LOCK_PATH = Path("data/.sma-writer.lock")

# A SEPARATE lock from the DB writer lock, used to serialize the memory-heavy
# jobs (retrain, autoresearch) so they never build training sets concurrently
# and thrash RAM (2026-06-29 incident: both coalesced onto one wake on a
# 16GB Mac → 12GB swap → a 1-min build stuck 2.5h). It is deliberately NOT the
# writer lock, so a long-running heavy job never blocks the evening DB-writing
# trading jobs (ingest/predict/decide).
#
# Anchored to the repo root (parents[2] of src/sma/locks.py), NOT a CWD-relative
# path: the two heavy jobs run from launchd with their own WorkingDirectory, and
# a relative path could resolve to different files and silently never contend
# (Codex review 2026-06-30). An absolute path guarantees both processes take the
# SAME lock regardless of CWD.
HEAVY_JOB_LOCK_PATH = Path(__file__).resolve().parents[2] / "data" / ".sma-heavy.lock"


def _read_holder(pid_path: Path) -> str:
    """Read the holder PID/label string atomically, returning '<unknown>' on any
    OSError. The PID file is forensics-only; a stale or missing file MUST NOT
    break the caller. This guards against a TOCTOU race where the holder
    releases between our exists() check and our read."""
    try:
        return pid_path.read_text().strip()
    except OSError:
        return "<unknown>"


@contextmanager
def writer_lock(
    *,
    lock_path: Path = DEFAULT_LOCK_PATH,
    label: str,
    timeout_s: float = 30.0,
    poll_initial_s: float = 0.05,
    poll_max_s: float = 1.0,
):
    """Hold an exclusive flock on `lock_path` for the duration of the with-block.

    Writes our PID + label to <lock_path>.with_suffix(".pid") while held.
    On contention, retries with exponential backoff up to timeout_s. Each retry
    logs the blocking holder for forensics.

    Raises WriterLockTimeoutError on timeout. Always releases on exit (normal
    or exception).
    """
    if not label:
        raise ValueError("writer_lock requires a non-empty label for forensics")
    if "\n" in label or "\r" in label:
        raise ValueError(
            "writer_lock label must not contain newlines (PID file format requires single-line)"
        )
    lock_path = Path(lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path = lock_path.with_suffix(".pid")

    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o644)
    deadline = time.monotonic() + timeout_s
    backoff = poll_initial_s
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                pid_path.write_text(f"{os.getpid()} {label}\n")
                break
            except OSError as exc:
                if exc.errno not in (errno.EAGAIN, errno.EACCES):
                    raise
                if time.monotonic() >= deadline:
                    raise WriterLockTimeoutError(
                        f"writer_lock({label!r}) timed out after {timeout_s}s; "
                        f"holder: {_read_holder(pid_path)}"
                    ) from exc
                logger.debug(
                    f"writer_lock({label!r}) waiting on {_read_holder(pid_path)}; "
                    f"retry in {backoff:.2f}s"
                )
                time.sleep(backoff)
                backoff = min(backoff * 2, poll_max_s)
        try:
            yield
        finally:
            try:
                pid_path.unlink(missing_ok=True)
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


@contextmanager
def heavy_job_lock(
    *,
    label: str = "heavy-job",
    timeout_s: float = 7200.0,
    lock_path: Path | None = None,
):
    """Serialize the memory-heavy jobs (retrain, autoresearch) so they never run
    their training-set builds concurrently and exhaust RAM.

    This is a SEPARATE flock from writer_lock (HEAVY_JOB_LOCK_PATH), so holding
    it for a long compute never blocks the evening DB-writing trading jobs.
    flock auto-releases when the holding process exits, so a crashed job cannot
    leave a stale lock that wedges the next run. Default 2h timeout: a waiter
    raises WriterLockTimeoutError rather than blocking forever if the holder
    hangs (the caller can then proceed or skip). `lock_path` overrides the lock
    file (tests only); production always uses the absolute HEAVY_JOB_LOCK_PATH.

    LOCK ORDERING (deadlock avoidance): callers that also take writer_lock MUST
    acquire heavy_job_lock FIRST (outer), then writer_lock (inner) — e.g.
    `with heavy_job_lock(), writer_lock(label=...):`. Never the reverse.
    """
    with writer_lock(
        lock_path=lock_path or HEAVY_JOB_LOCK_PATH, label=label, timeout_s=timeout_s
    ):
        yield
