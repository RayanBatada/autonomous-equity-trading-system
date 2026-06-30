"""Lock-tolerant read-only DuckDB connect.

DuckDB takes an OS-level file lock: a read-only open from one process FAILS while
another process holds a read-write connection (it does not queue). Our scheduled
reader jobs (notably `predict`) therefore crash with "Conflicting lock" when they
overlap a writer job (ingest/agents/decide) — which blocks the whole pipeline
(no predictions -> decide can't run). The fix is to retry the open with capped
backoff so a transient overlap waits the writer out instead of failing the job.

This is separate from `sma.locks.writer_lock` (an fcntl advisory lock used to
serialize *writers* in-app); DuckDB's own file lock is what bites cross-process
readers, so readers need this retry even though they never take the writer_lock.
"""

from __future__ import annotations

import time
from pathlib import Path

import duckdb

# Substrings DuckDB uses when another process holds a conflicting lock.
_LOCK_HINTS = ("Conflicting lock", "Could not set lock", "set lock on file")


def read_only_connect(
    path: str | Path,
    *,
    retries: int = 12,
    base_delay: float = 0.5,
    max_delay: float = 10.0,
    _connect=None,
) -> duckdb.DuckDBPyConnection:
    """Open `path` read-only, retrying when a conflicting writer lock is held.

    Re-raises immediately on any non-lock error. After `retries` exhausted
    lock-conflict attempts, re-raises the last lock error.
    """
    # Resolve at call time (not as a default) so monkeypatching duckdb.connect works.
    connect = _connect or duckdb.connect
    last: Exception | None = None
    for attempt in range(retries):
        try:
            return connect(str(path), read_only=True)
        except Exception as e:  # noqa: BLE001 - inspect message, re-raise non-lock
            if not any(hint in str(e) for hint in _LOCK_HINTS):
                raise
            last = e
            time.sleep(min(base_delay * (2 ** attempt), max_delay))
    assert last is not None
    raise last
