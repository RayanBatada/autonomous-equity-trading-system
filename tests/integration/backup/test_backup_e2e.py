"""End-to-end integration test for the backup CLI.

Creates a real DuckDB, runs the CLI via Click test runner, verifies the
backup file exists, the sentinel exists, and verified=true.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import duckdb
import pytest
from click.testing import CliRunner

from sma.backup.__main__ import cli
from sma.sentinels import read_sentinel


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


def test_backup_e2e_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Full CLI run: backup file created, sentinel written, verified=true."""
    db = tmp_path / "sma.duckdb"
    backup_dir = tmp_path / "backups"
    sentinel_dir = tmp_path / "sentinels"

    _make_valid_db(db)

    monkeypatch.setenv("SMA_SENTINEL_DIR", str(sentinel_dir))
    # Note: DEFAULT_LOCK_PATH is already patched by the autouse _hold_writer_lock_for_test
    # fixture in conftest.py, which also holds the lock. Do NOT re-patch it here;
    # run_backup's internal writer_lock(label="backup") uses the same default path and
    # Store.connect() checks the same DEFAULT_LOCK_PATH for the pid file.

    asof = date(2026, 4, 29)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "run",
            "--asof",
            asof.isoformat(),
            "--backup-dir",
            str(backup_dir),
            "--db",
            str(db),
        ],
    )

    assert result.exit_code == 0, result.output

    # Backup file should exist
    backup_file = backup_dir / f"sma-{asof.isoformat()}.duckdb"
    assert backup_file.exists(), f"Expected backup at {backup_file}"
    assert backup_file.stat().st_size > 0

    # Sentinel should be written and verified
    sentinel = read_sentinel(label="com.sma.backup.daily", asof=asof)
    assert sentinel is not None
    assert sentinel["verified"] is True
    assert sentinel["backup_path"] == str(backup_file)
    assert sentinel["backup_size_bytes"] > 0
    assert sentinel["label"] == "com.sma.backup.daily"
    assert sentinel["asof"] == asof.isoformat()


def test_restore_roundtrip(tmp_path, monkeypatch):
    """backup run -> restore --yes: DB readable + artifacts back in place."""
    import duckdb
    from click.testing import CliRunner

    import sma.locks as _locks
    from sma.backup.__main__ import cli
    from sma.locks import writer_lock

    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path / "s"))
    monkeypatch.setattr(_locks, "DEFAULT_LOCK_PATH", tmp_path / ".lk")
    db = tmp_path / "sma.duckdb"
    con = duckdb.connect(str(db)); con.execute("CREATE TABLE t AS SELECT 42 AS v"); con.close()
    models = tmp_path / "models_artifacts"
    models.mkdir(); (models / "m.pkl").write_bytes(b"x")
    bdir = tmp_path / "backups"

    with writer_lock(label="test", lock_path=tmp_path / ".lk"):
        r = CliRunner().invoke(cli, [
            "run", "--asof", "2026-04-29", "--db", str(db),
            "--backup-dir", str(bdir), "--models-dir", str(models),
        ])
    assert r.exit_code == 0, r.output

    # wreck live state
    db.unlink(); (models / "m.pkl").unlink()

    r = CliRunner().invoke(cli, [
        "restore", "--asof", "2026-04-29", "--backup-dir", str(bdir),
        "--db", str(db), "--models-dir", str(models), "--yes",
    ])
    assert r.exit_code == 0, r.output
    con = duckdb.connect(str(db), read_only=True)
    assert con.execute("SELECT v FROM t").fetchone()[0] == 42
    con.close()
    assert (models / "m.pkl").exists()
