"""Unit tests for sma.backup.runner."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import duckdb
import pytest

import sma.locks as _locks
from sma.backup.runner import _apply_retention, run_backup
from sma.locks import writer_lock
from sma.sentinels import read_sentinel

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_valid_db(path: Path) -> None:
    """Create a minimal DuckDB file with the _schema_version table."""
    conn = duckdb.connect(str(path))
    try:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS _schema_version "
            "(version BIGINT PRIMARY KEY, applied_at TIMESTAMP NOT NULL)"
        )
        conn.execute("INSERT INTO _schema_version VALUES (1, CURRENT_TIMESTAMP)")
    finally:
        conn.close()


def _make_backup_file(backup_dir: Path, d: date) -> Path:
    """Create a stub backup file (empty) in backup_dir for the given date."""
    p = backup_dir / f"sma-{d.isoformat()}.duckdb"
    p.write_bytes(b"stub")
    return p


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_backup_copies_db_to_destination(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    db = tmp_path / "sma.duckdb"
    backup_dir = tmp_path / "backups"
    _make_valid_db(db)

    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path / "sentinels"))
    monkeypatch.setattr(_locks, "DEFAULT_LOCK_PATH", tmp_path / ".sma-writer.lock")

    asof = date(2026, 4, 29)
    with writer_lock(label="test", lock_path=tmp_path / ".sma-writer.lock"):
        run_backup(db_path=db, backup_dir=backup_dir, asof=asof)

    expected = backup_dir / f"sma-{asof.isoformat()}.duckdb"
    assert expected.exists(), "Backup file was not created"
    assert expected.stat().st_size > 0


def test_backup_verifies_destination_is_readable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Corrupt the destination file after the atomic rename; verified=False."""
    db = tmp_path / "sma.duckdb"
    backup_dir = tmp_path / "backups"
    _make_valid_db(db)

    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path / "sentinels"))
    monkeypatch.setattr(_locks, "DEFAULT_LOCK_PATH", tmp_path / ".sma-writer.lock")

    asof = date(2026, 4, 29)

    # Patch _verify_backup to always return False to simulate a corrupt file.
    monkeypatch.setattr("sma.backup.runner._verify_backup", lambda path: False)

    with writer_lock(label="test", lock_path=tmp_path / ".sma-writer.lock"):
        payload = run_backup(db_path=db, backup_dir=backup_dir, asof=asof)

    assert payload["verified"] is False


def test_backup_writes_sentinel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    db = tmp_path / "sma.duckdb"
    backup_dir = tmp_path / "backups"
    sentinel_dir = tmp_path / "sentinels"
    _make_valid_db(db)

    monkeypatch.setenv("SMA_SENTINEL_DIR", str(sentinel_dir))
    monkeypatch.setattr(_locks, "DEFAULT_LOCK_PATH", tmp_path / ".sma-writer.lock")

    asof = date(2026, 4, 29)
    with writer_lock(label="test", lock_path=tmp_path / ".sma-writer.lock"):
        payload = run_backup(db_path=db, backup_dir=backup_dir, asof=asof)

    sentinel = read_sentinel(label="com.sma.backup.daily", asof=asof)
    assert sentinel is not None
    assert sentinel["label"] == "com.sma.backup.daily"
    assert sentinel["asof"] == asof.isoformat()
    assert "completed_at" in sentinel
    assert sentinel["backup_size_bytes"] > 0
    assert isinstance(sentinel["verified"], bool)
    assert "retained_dailies" in sentinel
    assert "retained_monthlies" in sentinel
    # payload returned by run_backup must match what was written
    assert sentinel["asof"] == payload["asof"]


def test_backup_retention_keeps_recent_dailies(tmp_path: Path):
    backup_dir = tmp_path / "backups"
    backup_dir.mkdir()

    asof = date(2026, 4, 29)
    # Create 35 daily backups (5 beyond retain_days=30)
    for i in range(35):
        d = asof - timedelta(days=34 - i)
        _make_backup_file(backup_dir, d)

    retained_dailies, retained_monthlies = _apply_retention(
        backup_dir=backup_dir,
        asof=asof,
        retain_days=30,
        retain_months=12,
    )

    remaining = list(backup_dir.glob("sma-*.duckdb"))
    # The 5 oldest should be deleted; 30 remain (the newest 30)
    assert len(remaining) == 30
    assert retained_dailies == 30
    assert retained_monthlies == 0


def test_backup_retention_keeps_monthly_snapshots(tmp_path: Path):
    """Monthly snapshots (last day of month) older than retain_days are kept."""
    backup_dir = tmp_path / "backups"
    backup_dir.mkdir()

    asof = date(2026, 4, 29)
    # Plant a monthly snapshot (last day of February 2026 = Feb 28) clearly
    # outside the 30-day retain window (asof - 29 = March 31, so Feb 28 is older).
    monthly_date = date(2026, 2, 28)
    _make_backup_file(backup_dir, monthly_date)

    # Also plant one recent daily within the retain_days window
    recent_date = asof - timedelta(days=5)
    _make_backup_file(backup_dir, recent_date)

    retained_dailies, retained_monthlies = _apply_retention(
        backup_dir=backup_dir,
        asof=asof,
        retain_days=30,
        retain_months=12,
    )

    remaining_names = {f.name for f in backup_dir.glob("sma-*.duckdb")}
    # Both the monthly snapshot AND the recent daily should survive
    assert f"sma-{monthly_date.isoformat()}.duckdb" in remaining_names
    assert f"sma-{recent_date.isoformat()}.duckdb" in remaining_names
    assert retained_monthlies == 1


def test_backup_file_has_restricted_permissions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Backup file must be chmod 0o600 after the atomic rename."""
    db = tmp_path / "sma.duckdb"
    backup_dir = tmp_path / "backups"
    _make_valid_db(db)

    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path / "sentinels"))
    monkeypatch.setattr(_locks, "DEFAULT_LOCK_PATH", tmp_path / ".sma-writer.lock")

    asof = date(2026, 4, 29)
    with writer_lock(label="test", lock_path=tmp_path / ".sma-writer.lock"):
        run_backup(db_path=db, backup_dir=backup_dir, asof=asof)

    expected = backup_dir / f"sma-{asof.isoformat()}.duckdb"
    assert expected.exists()
    assert oct(expected.stat().st_mode & 0o777) == "0o600"


def test_backup_retention_deletes_old_non_monthly(tmp_path: Path):
    """Old daily files that are NOT last-day-of-month get deleted."""
    backup_dir = tmp_path / "backups"
    backup_dir.mkdir()

    asof = date(2026, 4, 29)
    # Old mid-month file: outside retain_days window and not a monthly snapshot
    old_non_monthly = date(2026, 2, 15)
    _make_backup_file(backup_dir, old_non_monthly)

    # Old monthly snapshot (Feb 28) also outside window but should be kept
    old_monthly = date(2026, 2, 28)
    _make_backup_file(backup_dir, old_monthly)

    _apply_retention(
        backup_dir=backup_dir,
        asof=asof,
        retain_days=30,
        retain_months=12,
    )

    remaining_names = {f.name for f in backup_dir.glob("sma-*.duckdb")}
    assert f"sma-{old_non_monthly.isoformat()}.duckdb" not in remaining_names
    assert f"sma-{old_monthly.isoformat()}.duckdb" in remaining_names


def test_default_backup_dir_is_not_in_icloud():
    """Backups must NOT default to iCloud: sync-during-copy can corrupt the very
    file you'd restore from, and it thrashes the Mac (the original SMA saga).
    Off-machine DR can be opted into via SMA_BACKUP_DIR, but the safe default is
    local."""
    from sma.backup.runner import DEFAULT_BACKUP_DIR
    s = str(DEFAULT_BACKUP_DIR)
    assert "Mobile Documents" not in s and "CloudDocs" not in s, \
        f"backup dir defaults into iCloud: {s}"


def test_backup_verification_failure_marks_quality_failed_and_notifies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A failed verification must NOT look like a healthy backup: the sentinel
    carries quality.passed=False and a human is notified (2026-06-05 audit)."""
    db = tmp_path / "sma.duckdb"
    backup_dir = tmp_path / "backups"
    _make_valid_db(db)
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path / "sentinels"))
    monkeypatch.setattr(_locks, "DEFAULT_LOCK_PATH", tmp_path / ".sma-writer.lock")
    monkeypatch.setattr("sma.backup.runner._verify_backup", lambda path: False)

    asof = date(2026, 4, 29)
    notified: list[tuple[str, str]] = []
    with writer_lock(label="test", lock_path=tmp_path / ".sma-writer.lock"):
        payload = run_backup(
            db_path=db, backup_dir=backup_dir, asof=asof,
            notify_fn=lambda title, message: notified.append((title, message)),
        )

    assert payload["verified"] is False
    assert payload["quality"]["passed"] is False
    assert "backup_verification_failed" in payload["quality"]["blocking_failures"]
    assert notified


def test_backup_success_marks_quality_passed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    db = tmp_path / "sma.duckdb"
    backup_dir = tmp_path / "backups"
    _make_valid_db(db)
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path / "sentinels"))
    monkeypatch.setattr(_locks, "DEFAULT_LOCK_PATH", tmp_path / ".sma-writer.lock")

    asof = date(2026, 4, 29)
    with writer_lock(label="test", lock_path=tmp_path / ".sma-writer.lock"):
        payload = run_backup(db_path=db, backup_dir=backup_dir, asof=asof)

    assert payload["verified"] is True
    assert payload["quality"]["passed"] is True


def _make_models_dir(p: Path) -> Path:
    p.mkdir()
    (p / "xgb_x.pkl").write_bytes(b"\x80\x04fake")
    (p / "xgb_x.json").write_text('{"model_id": "xgb_x"}')
    return p


def test_backup_includes_model_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """2026-06-11 incident: artifacts had NO backup — the DB-only set left a
    destroyed models_artifacts unrecoverable except by retrain."""
    db = tmp_path / "sma.duckdb"
    backup_dir = tmp_path / "backups"
    _make_valid_db(db)
    models = _make_models_dir(tmp_path / "models_artifacts")
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path / "sentinels"))
    monkeypatch.setattr(_locks, "DEFAULT_LOCK_PATH", tmp_path / ".sma-writer.lock")
    asof = date(2026, 4, 29)
    with writer_lock(label="test", lock_path=tmp_path / ".sma-writer.lock"):
        payload = run_backup(db_path=db, backup_dir=backup_dir, asof=asof, models_dir=models)
    dest = backup_dir / f"models-{asof.isoformat()}"
    assert (dest / "xgb_x.pkl").exists() and (dest / "xgb_x.json").exists()
    assert payload["models_files"] == 2
    assert payload["quality"]["passed"] is True


def test_backup_missing_models_dir_warns_but_passes(tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch):
    db = tmp_path / "sma.duckdb"
    _make_valid_db(db)
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path / "sentinels"))
    monkeypatch.setattr(_locks, "DEFAULT_LOCK_PATH", tmp_path / ".sma-writer.lock")
    notes = []
    with writer_lock(label="test", lock_path=tmp_path / ".sma-writer.lock"):
        payload = run_backup(
            db_path=db, backup_dir=tmp_path / "b", asof=date(2026, 4, 29),
            models_dir=tmp_path / "nope", notify_fn=lambda title, message: notes.append(title),
        )
    assert payload["models_files"] == 0
    assert payload["quality"]["passed"] is True  # DB backup itself is fine
    assert any("model" in t.lower() for t in notes)  # but a human hears about it


def test_retention_prunes_model_dirs_with_db_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    db = tmp_path / "sma.duckdb"
    backup_dir = tmp_path / "backups"
    _make_valid_db(db)
    backup_dir.mkdir()
    old = date(2025, 3, 12)  # not month-end, far older than retention
    (backup_dir / f"sma-{old.isoformat()}.duckdb").write_bytes(b"x")
    olddir = backup_dir / f"models-{old.isoformat()}"
    olddir.mkdir()
    (olddir / "stale.pkl").write_bytes(b"x")
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path / "sentinels"))
    monkeypatch.setattr(_locks, "DEFAULT_LOCK_PATH", tmp_path / ".sma-writer.lock")
    with writer_lock(label="test", lock_path=tmp_path / ".sma-writer.lock"):
        run_backup(db_path=db, backup_dir=backup_dir, asof=date(2026, 4, 29))
    assert not (backup_dir / f"sma-{old.isoformat()}.duckdb").exists()
    assert not olddir.exists(), "model dirs must follow the same retention"
