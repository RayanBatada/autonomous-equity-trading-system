"""Tests for the 2026-08-25 monthly DB compaction addition to
sma.backup.runner: `_is_first_saturday_of_month` (the scheduling gate),
`_compact_database` (the actual COPY FROM DATABASE mechanism, real on a
small synthetic DB here -- see the runner module docstring for the one real
measurement on a full copy of the production DB), and their non-fatal wiring
into `run_backup`.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import duckdb
import pytest

import sma.locks as _locks
from sma.backup.runner import _compact_database, _is_first_saturday_of_month, run_backup
from sma.locks import writer_lock

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
# _is_first_saturday_of_month -- pure scheduling condition, frozen dates
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "d, expected",
    [
        (date(2026, 8, 1), True),  # first Saturday of Aug 2026
        (date(2026, 8, 8), False),  # second Saturday -- day > 7
        (date(2026, 8, 15), False),
        (date(2026, 8, 22), False),
        (date(2026, 8, 29), False),
        (date(2026, 8, 2), False),  # a Sunday, not a Saturday at all
        (date(2026, 8, 24), False),  # a Monday
        (date(2026, 1, 3), True),  # first Saturday of Jan 2026
        (date(2026, 2, 7), True),  # first Saturday of Feb 2026 -- day==7 boundary
        (date(2027, 5, 1), True),  # first Saturday of May 2027
    ],
)
def test_is_first_saturday_of_month(d: date, expected: bool):
    assert _is_first_saturday_of_month(d) is expected


# ---------------------------------------------------------------------------
# _compact_database -- real (not mocked) on a small synthetic DB
# ---------------------------------------------------------------------------


_ROW_EXPR = (
    "md5(i::VARCHAR || '_a' || random()::VARCHAR) || "
    "md5(i::VARCHAR || '_b' || random()::VARCHAR) || "
    "md5(i::VARCHAR || '_c' || random()::VARCHAR)"
)


@pytest.mark.serial
def test_compact_database_reclaims_space_and_preserves_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Build a DB with an EARLY table (occupies the front of the file) and a
    LATER table (occupies the tail), then delete all of the early table's
    rows and checkpoint. Per the runner module's docstring finding (verified
    on a full copy of the production DB), CHECKPOINT does not reclaim free
    blocks that aren't at the tail of the file -- deleting a non-trailing
    table's rows leaves a real, non-shrinkable hole (unlike deleting
    scattered rows from a single table, whose mostly-still-occupied blocks
    checkpoint CAN often still compact away, which doesn't exercise the
    interesting case). Then compact: the file must shrink and every
    remaining row must survive intact."""
    db = tmp_path / "sma.duckdb"
    monkeypatch.setattr(_locks, "DEFAULT_LOCK_PATH", tmp_path / ".sma-writer.lock")

    conn = duckdb.connect(str(db))
    conn.execute(
        "CREATE TABLE _schema_version (version BIGINT PRIMARY KEY, applied_at TIMESTAMP NOT NULL)"
    )
    conn.execute("INSERT INTO _schema_version VALUES (1, CURRENT_TIMESTAMP)")
    conn.execute("CREATE TABLE early (id BIGINT, val VARCHAR)")
    conn.execute(f"INSERT INTO early SELECT i, {_ROW_EXPR} FROM range(60000) t(i)")
    conn.execute("CHECKPOINT")
    conn.execute("CREATE TABLE late (id BIGINT, val VARCHAR)")
    conn.execute(f"INSERT INTO late SELECT i, {_ROW_EXPR} FROM range(60000) t(i)")
    conn.execute("CHECKPOINT")
    conn.execute("DELETE FROM early")
    conn.execute("CHECKPOINT")
    conn.close()

    size_before = db.stat().st_size

    with writer_lock(label="test", lock_path=tmp_path / ".sma-writer.lock"):
        result = _compact_database(db)

    assert result["ran"] is True
    assert result["size_before_bytes"] == size_before
    assert result["reclaimed_bytes"] > 500_000  # deleted a whole 60k-row table's worth
    assert db.stat().st_size == result["size_after_bytes"]
    assert db.stat().st_size < size_before

    verify = duckdb.connect(str(db), read_only=True)
    try:
        assert verify.execute("SELECT COUNT(*) FROM late").fetchone()[0] == 60000
        assert verify.execute("SELECT COUNT(*) FROM early").fetchone()[0] == 0
        version = verify.execute("SELECT MAX(version) FROM _schema_version").fetchone()[0]
        assert version is not None and version >= 1
    finally:
        verify.close()

    # No leftover .compact.tmp file.
    assert not db.with_name(db.name + ".compact.tmp").exists()


# ---------------------------------------------------------------------------
# run_backup wiring -- _compact_database mocked here (scheduling + non-fatal
# wrapper only; the real mechanism is exercised above).
# ---------------------------------------------------------------------------


def test_run_backup_compacts_on_first_saturday(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    db = tmp_path / "sma.duckdb"
    backup_dir = tmp_path / "backups"
    _make_valid_db(db)
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path / "sentinels"))
    monkeypatch.setattr(_locks, "DEFAULT_LOCK_PATH", tmp_path / ".sma-writer.lock")

    canned = {
        "ran": True,
        "size_before_bytes": 100,
        "size_after_bytes": 80,
        "reclaimed_bytes": 20,
    }
    calls: list[Path] = []
    monkeypatch.setattr(
        "sma.backup.runner._compact_database",
        lambda path: (calls.append(path), dict(canned))[1],
    )

    asof = date(2026, 8, 1)  # first Saturday of Aug 2026
    with writer_lock(label="test", lock_path=tmp_path / ".sma-writer.lock"):
        payload = run_backup(db_path=db, backup_dir=backup_dir, asof=asof)

    assert calls == [db]
    assert payload["compaction"] == canned


def test_run_backup_skips_compaction_on_non_first_saturday(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    db = tmp_path / "sma.duckdb"
    backup_dir = tmp_path / "backups"
    _make_valid_db(db)
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path / "sentinels"))
    monkeypatch.setattr(_locks, "DEFAULT_LOCK_PATH", tmp_path / ".sma-writer.lock")

    calls: list[Path] = []
    monkeypatch.setattr(
        "sma.backup.runner._compact_database", lambda path: calls.append(path)
    )

    asof = date(2026, 8, 8)  # second Saturday -- must NOT compact
    with writer_lock(label="test", lock_path=tmp_path / ".sma-writer.lock"):
        payload = run_backup(db_path=db, backup_dir=backup_dir, asof=asof)

    assert calls == []
    assert payload["compaction"] == {"ran": False}


def test_run_backup_compaction_failure_is_non_fatal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    db = tmp_path / "sma.duckdb"
    backup_dir = tmp_path / "backups"
    _make_valid_db(db)
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path / "sentinels"))
    monkeypatch.setattr(_locks, "DEFAULT_LOCK_PATH", tmp_path / ".sma-writer.lock")

    def _boom(path):
        raise RuntimeError("disk full")

    monkeypatch.setattr("sma.backup.runner._compact_database", _boom)

    asof = date(2026, 8, 1)  # first Saturday
    with writer_lock(label="test", lock_path=tmp_path / ".sma-writer.lock"):
        payload = run_backup(db_path=db, backup_dir=backup_dir, asof=asof)  # must not raise

    assert payload["compaction"]["ran"] is False
    assert "disk full" in payload["compaction"]["error"]
    # The rest of the backup must complete normally despite the compaction
    # failure -- non-fatal means non-fatal.
    expected = backup_dir / f"sma-{asof.isoformat()}.duckdb"
    assert expected.exists()
    assert payload["verified"] is True
    assert payload["quality"]["passed"] is True
