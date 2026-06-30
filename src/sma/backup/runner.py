"""Backup runner: checkpoint, copy, verify, apply retention, write sentinel.

Public entry point: run_backup().
"""

from __future__ import annotations

import calendar
import contextlib
import os
import re
import shutil
import time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import duckdb
from loguru import logger

from sma.ingest.notify import notify_failure
from sma.ingest.store import Store
from sma.locks import writer_lock
from sma.sentinels import write_sentinel

__all__ = ["DEFAULT_BACKUP_DIR", "run_backup"]

_BACKUP_FILENAME_RE = re.compile(r"^sma-(\d{4}-\d{2}-\d{2})\.duckdb$")

# LOCAL by default — NOT iCloud. iCloud's daemon can sync a .tmp mid-write and
# leave a partially-synced copy under the final name on another device, corrupting
# the very backup you'd restore from; it also thrashes the Mac (the SMA iCloud
# saga). Opt into an off-machine location explicitly via SMA_BACKUP_DIR (and
# prefer one outside ~/Library/Mobile Documents).
DEFAULT_BACKUP_DIR = Path(
    os.environ.get("SMA_BACKUP_DIR", "~/sma-backups")
).expanduser()


def _is_last_day_of_month(d: date) -> bool:
    return d.day == calendar.monthrange(d.year, d.month)[1]


def _backup_dates_in_dir(backup_dir: Path) -> list[date]:
    """Return sorted list of dates for which backup files exist.

    iCloud's filesystem driver occasionally raises `InterruptedError` (errno 4)
    when iterating a directory mid-sync. Retry up to 5 times with a short
    backoff before letting the error bubble up. Without this retry, the
    backup daemon's retention-pruning step crashes whenever iCloud is
    actively syncing during the 22:00 ET fire.
    """
    last_err: Exception | None = None
    for attempt in range(5):
        try:
            dates: list[date] = []
            for f in backup_dir.iterdir():
                m = _BACKUP_FILENAME_RE.match(f.name)
                if m:
                    with contextlib.suppress(ValueError):
                        dates.append(date.fromisoformat(m.group(1)))
            dates.sort()
            return dates
        except InterruptedError as e:
            last_err = e
            time.sleep(0.5 * (attempt + 1))
    assert last_err is not None
    raise last_err


def _apply_retention(
    backup_dir: Path,
    asof: date,
    retain_days: int,
    retain_months: int,
) -> tuple[int, int]:
    """Delete old backup files according to retention policy.

    Keep rules (applied in priority order):
    1. Any backup whose date is within the last retain_days calendar days
       (i.e. date >= asof - timedelta(days=retain_days-1)) is kept
       unconditionally as a daily backup.
    2. Of the remaining (older) backups, keep ones whose date is the last
       day of their respective month (monthly snapshot). Cap at retain_months
       such snapshots (newest first).
    3. Delete everything else.

    Returns (retained_dailies, retained_monthlies).
    """
    all_dates = _backup_dates_in_dir(backup_dir)
    cutoff = asof - timedelta(days=retain_days - 1)

    recent = {d for d in all_dates if d >= cutoff}
    older = {d for d in all_dates if d < cutoff}

    # From older, keep monthly snapshots (last day of month), capped.
    monthly_candidates = sorted(
        [d for d in older if _is_last_day_of_month(d)],
        reverse=True,
    )
    monthly_keep = set(monthly_candidates[:retain_months])

    to_delete = older - monthly_keep

    for d in sorted(to_delete):
        target = backup_dir / f"sma-{d.isoformat()}.duckdb"
        try:
            target.unlink(missing_ok=True)
            logger.info("backup: deleted old file {}", target.name)
        except OSError as exc:
            logger.warning("backup: could not delete {}: {}", target.name, exc)
        models_target = backup_dir / f"models-{d.isoformat()}"
        if models_target.is_dir():
            try:
                shutil.rmtree(models_target)
                logger.info("backup: deleted old model dir {}", models_target.name)
            except OSError as exc:
                logger.warning(
                    "backup: could not delete {}: {}", models_target.name, exc
                )

    retained_dailies = len(recent)
    retained_monthlies = len(monthly_keep)
    return retained_dailies, retained_monthlies


def _checkpoint_db(db_path: Path) -> None:
    """Open the DB read-write via Store, run CHECKPOINT, then close.

    Routes through Store so the writer_lock assertion is respected.
    This flushes the WAL so the file-copy is consistent.
    Skips gracefully if the file does not exist yet (test environments).
    Must be called while writer_lock is held.
    """
    if not db_path.exists():
        logger.debug("backup: db_path {} does not exist; skipping CHECKPOINT", db_path)
        return
    store = Store(path=db_path).connect(read_only=False)
    try:
        store.conn.execute("CHECKPOINT")
        logger.debug("backup: CHECKPOINT complete for {}", db_path)
    finally:
        store.close()


def _verify_backup(backup_path: Path) -> bool:
    """Open the backup read-only and run a minimal query.

    Returns True if the file is a valid DuckDB with an _schema_version table,
    False on any error.
    """
    try:
        conn = duckdb.connect(str(backup_path), read_only=True)
        try:
            row = conn.execute("SELECT MAX(version) FROM _schema_version").fetchone()
            verified = row is not None
            logger.debug("backup: verify {} -> version={}", backup_path.name, row)
            return bool(verified)
        finally:
            conn.close()
    except Exception as exc:
        logger.warning("backup: verification failed for {}: {}", backup_path.name, exc)
        return False


def _backup_models(models_dir: Path, backup_dir: Path, asof: date) -> tuple[int, int]:
    """Copy model artifacts into backup_dir/models-<asof>/ (atomic via .tmp
    rename). Returns (files, bytes). The 2026-06-11 incident destroyed
    models_artifacts and the DB-only backup set could not restore it."""
    dest = backup_dir / f"models-{asof.isoformat()}"
    tmp = backup_dir / f"models-{asof.isoformat()}.tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    shutil.copytree(models_dir, tmp)
    n_files = 0
    n_bytes = 0
    for f in tmp.rglob("*"):
        if f.is_file():
            f.chmod(0o600)
            n_files += 1
            n_bytes += f.stat().st_size
    if dest.exists():
        shutil.rmtree(dest)
    tmp.replace(dest)
    return n_files, n_bytes


def run_backup(
    *,
    db_path: Path,
    backup_dir: Path = DEFAULT_BACKUP_DIR,
    asof: date,
    retain_days: int = 30,
    retain_months: int = 12,
    models_dir: Path | None = Path("models_artifacts"),
    notify_fn=notify_failure,
) -> dict:
    """Perform one daily backup run.

    Steps:
    1. Acquire writer_lock(label="backup").
    2. CHECKPOINT the source DB so the WAL is flushed.
    3. Atomic copy: copy2 to a .tmp then rename to final destination.
    4. Verify the backup is readable.
    5. Apply retention policy (delete old files).
    6. Write sentinel.

    Returns the sentinel payload dict.
    """
    db_path = Path(db_path)
    backup_dir = Path(backup_dir)

    if "Mobile Documents" in str(backup_dir):
        logger.warning(
            "backup: target {} is inside iCloud — sync-during-copy can corrupt the "
            "backup you'd restore from. Prefer a local or non-iCloud path.",
            backup_dir,
        )

    if not db_path.exists():
        raise FileNotFoundError(f"Source DB not found: {db_path}")

    backup_dir.mkdir(parents=True, exist_ok=True)

    backup_filename = f"sma-{asof.isoformat()}.duckdb"
    backup_path = backup_dir / backup_filename
    tmp_path = backup_dir / f"{backup_filename}.tmp"

    with writer_lock(label="backup"):
        # Step 2: flush WAL
        _checkpoint_db(db_path)

        # Step 3: atomic copy
        shutil.copy2(db_path, tmp_path)
        tmp_path.replace(backup_path)
        backup_path.chmod(0o600)
        backup_size = backup_path.stat().st_size
        logger.info(
            "backup: copied {} -> {} ({:.1f} MB)",
            db_path,
            backup_path,
            backup_size / 1_048_576,
        )

        # Step 4: verify
        verified = _verify_backup(backup_path)
        if not verified:
            logger.warning("backup: verification FAILED for {}", backup_path)
            # A corrupt/unreadable backup must reach a human — otherwise the
            # disaster-recovery copy is silently useless (2026-06-05 audit).
            notify_fn(
                title="sma: backup verification FAILED",
                message=(
                    f"The {asof} backup at {backup_path} did not verify as a "
                    "readable DuckDB — the backup may be corrupt. Investigate "
                    "before relying on it for recovery."
                ),
            )

        # Step 4b: model artifacts (2026-06-11 incident: no artifact backup
        # existed; a destroyed models_artifacts was only recoverable by
        # retraining). Missing/empty dir warns a human but does not fail the
        # DB backup; a copy ERROR fails closed like verification.
        models_files = 0
        models_bytes = 0
        models_ok = True
        if models_dir is not None:
            models_dir = Path(models_dir)
            if not models_dir.is_dir() or not any(models_dir.iterdir()):
                logger.warning("backup: models dir {} missing/empty", models_dir)
                notify_fn(
                    title="sma: backup found no model artifacts",
                    message=(
                        f"{models_dir} is missing or empty — the backup set has "
                        "no model artifacts to protect. If a retrain just ran, "
                        "investigate; tonight's predict needs an artifact."
                    ),
                )
            else:
                try:
                    models_files, models_bytes = _backup_models(
                        models_dir, backup_dir, asof
                    )
                    logger.info(
                        "backup: copied {} model files ({:.1f} MB)",
                        models_files, models_bytes / 1_048_576,
                    )
                except Exception as e:
                    models_ok = False
                    logger.exception("backup: model artifacts copy FAILED: {}", e)
                    notify_fn(
                        title="sma: model artifacts backup FAILED",
                        message=f"copy of {models_dir} failed: {e!r}",
                    )

        # Step 5: retention
        retained_dailies, retained_monthlies = _apply_retention(
            backup_dir=backup_dir,
            asof=asof,
            retain_days=retain_days,
            retain_months=retain_months,
        )

        # Step 6: sentinel (written while lock is still held)
        completed_at = datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        payload = {
            "label": "com.sma.backup.daily",
            "asof": asof.isoformat(),
            "completed_at": completed_at,
            "backup_path": str(backup_path),
            "backup_size_bytes": backup_size,
            "verified": verified,
            # Quality block so downstream readiness/monitoring fail closed on a
            # bad backup instead of treating any written sentinel as success.
            "models_files": models_files,
            "models_bytes": models_bytes,
            "quality": {
                "passed": verified and models_ok,
                "blocking_failures": (
                    ([] if verified else ["backup_verification_failed"])
                    + ([] if models_ok else ["models_backup_failed"])
                ),
            },
            "retained_dailies": retained_dailies,
            "retained_monthlies": retained_monthlies,
        }
        write_sentinel(label="com.sma.backup.daily", asof=asof, payload=payload)
        logger.info("backup: sentinel written for {}", asof)

    return payload
