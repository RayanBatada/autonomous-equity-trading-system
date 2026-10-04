"""2026-08-23 incident: senate-ingest and house-ingest both missed their
normal Sunday morning slots (machine off), and the watchdog/launchd caught
up BOTH jobs (plus a stale Saturday backup) within the same second when the
Mac finally booted at 19:11:40 — all three raced for the single writer
lock. House won; senate retried on the default 30s writer_lock timeout,
exhausted it, and exited 1 — that week's Senate PTR ingest was skipped.

The two jobs' nominal Sunday fire times (10:00 / 11:00 ET, see
src/sma/schedule.py) are already staggered 60 minutes apart specifically to
avoid writer_lock contention — that staggering did nothing here because a
boot-catchup collapses all overdue jobs into the same instant regardless of
their nominal spacing. The real fix is patience, not spacing: these jobs are
not urgent (mirrors src/sma/backup/runner.py's `writer_lock(..., timeout_s=
900.0)` reasoning — "a backup is never urgent; waiting beats failing").
House's actual 8/23 run took well under a minute; a 900s timeout gives any
reasonable queue of Sunday catch-up jobs room to serialize through the lock
instead of racing and failing.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import sma.ingest.sources.politician_trades as politician_trades_mod
import sma.ingest.sources.senate_trades as senate_trades_mod
import sma.locks as locks_mod
from sma.locks import writer_lock as real_writer_lock


def _make_lock_spy(lock_path: Path) -> tuple:
    calls: list[dict] = []

    @contextmanager
    def _spy(*, label: str, **kwargs):
        calls.append({"label": label, **kwargs})
        with real_writer_lock(lock_path=lock_path, label=label, **kwargs):
            yield

    return _spy, calls


def test_senate_ingest_writer_lock_is_patient(tmp_path, monkeypatch):
    lock_path = tmp_path / ".sma-writer.lock"
    monkeypatch.setattr("sma.locks.DEFAULT_LOCK_PATH", lock_path)
    monkeypatch.setattr(
        senate_trades_mod, "run_senate_ingest",
        lambda **kw: {"filings_seen": 0, "filings_parsed": 0},
    )
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path / "sentinels"))

    spy, calls = _make_lock_spy(lock_path)
    db = tmp_path / "sma.duckdb"
    argv = ["senate_trades.py", "--db", str(db), "--limit", "0"]

    with (
        monkeypatch.context() as m,
        patch.object(locks_mod, "writer_lock", spy),
    ):
        m.setattr("sys.argv", argv)
        senate_trades_mod._main()

    assert calls, "writer_lock was never called"
    assert calls[0]["label"] == "senate_ingest"
    assert calls[0].get("timeout_s") == 900.0, (
        "senate_ingest's writer_lock must use a patient (900s) timeout, not "
        f"the 30s default — got {calls[0]}"
    )


def test_house_ingest_writer_lock_is_patient(tmp_path, monkeypatch):
    lock_path = tmp_path / ".sma-writer.lock"
    monkeypatch.setattr("sma.locks.DEFAULT_LOCK_PATH", lock_path)
    monkeypatch.setattr(
        politician_trades_mod, "run_ingest",
        lambda **kw: {"filings_seen": 0, "filings_parsed": 0},
    )
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path / "sentinels"))

    spy, calls = _make_lock_spy(lock_path)
    db = tmp_path / "sma.duckdb"
    argv = ["politician_trades.py", "--db", str(db), "--limit", "0"]

    with (
        monkeypatch.context() as m,
        patch.object(locks_mod, "writer_lock", spy),
    ):
        m.setattr("sys.argv", argv)
        politician_trades_mod._main()

    assert calls, "writer_lock was never called"
    assert calls[0]["label"] == "politician_ingest"
    assert calls[0].get("timeout_s") == 900.0, (
        "politician_ingest's (house) writer_lock must use a patient (900s) "
        f"timeout, not the 30s default — got {calls[0]}"
    )
