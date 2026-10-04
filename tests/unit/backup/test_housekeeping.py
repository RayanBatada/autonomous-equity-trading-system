"""Unit tests for the 2026-08-24 backup housekeeping additions:
sentinel pruning (`_prune_old_sentinels`) and log rotation
(`_rotate_large_logs`), plus their wiring into `run_backup`.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import duckdb
import pytest

import sma.locks as _locks
from sma.backup.runner import (
    _LOG_ROTATE_EXCLUDE,
    _prune_old_sentinels,
    _rotate_large_logs,
    run_backup,
)
from sma.locks import writer_lock
from sma.sentinels import sentinel_path, write_sentinel

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_valid_db(path: Path) -> None:
    conn = duckdb.connect(str(path))
    try:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS _schema_version "
            "(version BIGINT PRIMARY KEY, applied_at TIMESTAMP NOT NULL)"
        )
        conn.execute("INSERT INTO _schema_version VALUES (1, CURRENT_TIMESTAMP)")
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# _prune_old_sentinels
# ---------------------------------------------------------------------------


def test_prune_deletes_sentinels_older_than_retain_days(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    sentinel_dir = tmp_path / "sentinels"
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(sentinel_dir))

    asof = date(2026, 8, 24)
    old_date = asof - timedelta(days=61)  # just past the 60-day default
    write_sentinel(
        label="com.sma.ingest.daily", asof=old_date, payload={"completed_at": "x"}
    )
    old_path = sentinel_path(label="com.sma.ingest.daily", asof=old_date)
    assert old_path.exists()

    pruned = _prune_old_sentinels(asof=asof, retain_days=60)

    assert pruned == 1
    assert not old_path.exists()


def test_prune_keeps_sentinels_within_retain_days(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    sentinel_dir = tmp_path / "sentinels"
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(sentinel_dir))

    asof = date(2026, 8, 24)
    recent_date = asof - timedelta(days=59)  # inside the 60-day window
    write_sentinel(
        label="com.sma.ingest.daily", asof=recent_date, payload={"completed_at": "x"}
    )
    recent_path = sentinel_path(label="com.sma.ingest.daily", asof=recent_date)

    pruned = _prune_old_sentinels(asof=asof, retain_days=60)

    assert pruned == 0
    assert recent_path.exists()


def test_prune_keeps_today_and_boundary_exactly_retain_days_old(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """cutoff = asof - retain_days; a file exactly on the cutoff date is kept."""
    sentinel_dir = tmp_path / "sentinels"
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(sentinel_dir))

    asof = date(2026, 8, 24)
    boundary_date = asof - timedelta(days=60)
    write_sentinel(
        label="com.sma.backup.daily", asof=boundary_date, payload={"completed_at": "x"}
    )
    boundary_path = sentinel_path(label="com.sma.backup.daily", asof=boundary_date)

    pruned = _prune_old_sentinels(asof=asof, retain_days=60)

    assert pruned == 0
    assert boundary_path.exists()


def test_prune_ignores_non_sentinel_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    sentinel_dir = tmp_path / "sentinels"
    sentinel_dir.mkdir(parents=True)
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(sentinel_dir))

    # Not the "<label>-<date>.json" shape at all — must survive untouched.
    stray = sentinel_dir / "README.txt"
    stray.write_text("not a sentinel")

    pruned = _prune_old_sentinels(asof=date(2026, 8, 24), retain_days=60)

    assert pruned == 0
    assert stray.exists()


def test_prune_missing_dir_is_a_noop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path / "does-not-exist"))
    assert _prune_old_sentinels(asof=date(2026, 8, 24), retain_days=60) == 0


# ---------------------------------------------------------------------------
# _rotate_large_logs
# ---------------------------------------------------------------------------


def test_rotate_trims_oversized_log_keeping_the_tail(tmp_path: Path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    f = log_dir / "ingest.err.log"
    # 10 bytes/line * 300 lines = 3000 bytes, well over a tiny max_bytes.
    lines = [f"line-{i:05d}\n" for i in range(300)]
    f.write_text("".join(lines))
    original_size = f.stat().st_size

    trimmed = _rotate_large_logs(log_dir=log_dir, max_bytes=1000, keep_bytes=300)

    assert f.name in trimmed
    assert trimmed[f.name] == original_size - f.stat().st_size
    new_content = f.read_text()
    assert len(new_content) <= 300
    # The tail (most recent lines) must survive; the head must not.
    assert "line-00299" in new_content
    assert "line-00000" not in new_content


def test_rotate_leaves_small_logs_untouched(tmp_path: Path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    f = log_dir / "agents.err.log"
    f.write_text("small log\n")

    trimmed = _rotate_large_logs(log_dir=log_dir, max_bytes=1000, keep_bytes=300)

    assert trimmed == {}
    assert f.read_text() == "small log\n"


def test_rotate_excludes_dashboard_logs_even_if_oversized(tmp_path: Path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    for name in _LOG_ROTATE_EXCLUDE:
        (log_dir / name).write_text("x" * 5000)

    trimmed = _rotate_large_logs(log_dir=log_dir, max_bytes=1000, keep_bytes=300)

    assert trimmed == {}
    for name in _LOG_ROTATE_EXCLUDE:
        assert (log_dir / name).stat().st_size == 5000


def test_rotate_ignores_non_log_files(tmp_path: Path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    (log_dir / "notes.txt").write_text("x" * 5000)

    trimmed = _rotate_large_logs(log_dir=log_dir, max_bytes=1000, keep_bytes=300)

    assert trimmed == {}


def test_rotate_missing_dir_is_a_noop(tmp_path: Path):
    assert _rotate_large_logs(log_dir=tmp_path / "nope") == {}


# ---------------------------------------------------------------------------
# Wired into run_backup
# ---------------------------------------------------------------------------


def test_run_backup_prunes_sentinels_and_rotates_logs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    db = tmp_path / "sma.duckdb"
    backup_dir = tmp_path / "backups"
    sentinel_dir = tmp_path / "sentinels"
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    _make_valid_db(db)

    monkeypatch.setenv("SMA_SENTINEL_DIR", str(sentinel_dir))
    monkeypatch.setattr(_locks, "DEFAULT_LOCK_PATH", tmp_path / ".sma-writer.lock")

    asof = date(2026, 8, 24)

    # A stale sentinel that should be pruned...
    old_date = asof - timedelta(days=90)
    write_sentinel(
        label="com.sma.ingest.daily", asof=old_date, payload={"completed_at": "x"}
    )
    old_sentinel_path = sentinel_path(label="com.sma.ingest.daily", asof=old_date)

    # ...and an oversized log that should be rotated.
    big_log = log_dir / "ingest.err.log"
    big_log.write_text("x" * 5000)

    with writer_lock(label="test", lock_path=tmp_path / ".sma-writer.lock"):
        payload = run_backup(
            db_path=db,
            backup_dir=backup_dir,
            asof=asof,
            models_dir=None,
            log_dir=log_dir,
            log_rotate_max_bytes=1000,
            log_rotate_keep_bytes=300,
        )

    assert not old_sentinel_path.exists()
    assert payload["pruned_sentinels"] == 1
    assert big_log.stat().st_size == 300
    assert payload["rotated_logs"] == {"ingest.err.log": 5000 - 300}
    # Today's own backup sentinel must still be written normally.
    assert sentinel_path(label="com.sma.backup.daily", asof=asof).exists()


def test_run_backup_housekeeping_failure_does_not_fail_backup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A broken housekeeping step must not take down the main backup."""
    db = tmp_path / "sma.duckdb"
    backup_dir = tmp_path / "backups"
    _make_valid_db(db)

    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path / "sentinels"))
    monkeypatch.setattr(_locks, "DEFAULT_LOCK_PATH", tmp_path / ".sma-writer.lock")

    def _boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("sma.backup.runner._prune_old_sentinels", _boom)
    monkeypatch.setattr("sma.backup.runner._rotate_large_logs", _boom)

    asof = date(2026, 8, 24)
    with writer_lock(label="test", lock_path=tmp_path / ".sma-writer.lock"):
        payload = run_backup(
            db_path=db, backup_dir=backup_dir, asof=asof, models_dir=None
        )

    assert payload["verified"] is True
    assert payload["pruned_sentinels"] == 0
    assert payload["rotated_logs"] == {}


def test_prune_keeps_reconcile_batch_sentinels_of_any_age(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Flaw hunt 2026-10-01 A4. Reconcile's batch sentinel is its only record
    that a decide-date was reconciled (_unreconciled_batches treats a batch
    with no sentinel as open). Pruning it at 60 days made every afternoon's
    reconcile re-fetch every batch older than that, re-run the drift
    detectors, re-send their old alerts, and rewrite the sentinels, which the
    22:00 backup then deleted again (45 May-July batches rewritten 2026-10-04
    15:41). The run-date liveness sentinel (.ran) carries no such meaning and
    is still pruned."""
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path / "sentinels"))
    asof = date(2026, 10, 4)
    old = date(2026, 5, 4)
    write_sentinel(label="com.sma.live.reconcile.daily", asof=old, payload={"completed_at": "x"})
    write_sentinel(
        label="com.sma.live.reconcile.daily.ran", asof=old, payload={"completed_at": "x"}
    )
    batch = sentinel_path(label="com.sma.live.reconcile.daily", asof=old)
    ran = sentinel_path(label="com.sma.live.reconcile.daily.ran", asof=old)

    pruned = _prune_old_sentinels(asof=asof, retain_days=60)

    assert batch.exists()
    assert not ran.exists()
    assert pruned == 1
