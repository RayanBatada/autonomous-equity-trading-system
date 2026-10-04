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

from loguru import logger

from sma.db_connect import read_only_connect
from sma.ingest.notify import notify_failure
from sma.ingest.store import Store
from sma.locks import writer_lock
from sma.sentinels import sentinel_dir as _sentinel_dir
from sma.sentinels import write_sentinel

__all__ = [
    "DEFAULT_BACKUP_DIR",
    "DEFAULT_LOG_DIR",
    "DEFAULT_OFFSITE_BACKUP_DIR",
    "SENTINEL_RETAIN_DAYS",
    "run_backup",
]

_BACKUP_FILENAME_RE = re.compile(r"^sma-(\d{4}-\d{2}-\d{2})\.duckdb$")

# Sentinel filenames are "<label>-<YYYY-MM-DD>.json" and labels themselves
# contain dots (e.g. "com.sma.ingest.daily"), so match the trailing date
# rather than splitting on the first '-'.
_SENTINEL_FILENAME_RE = re.compile(r"-(\d{4}-\d{2}-\d{2})\.json$")

# LOCAL by default — NOT iCloud. iCloud's daemon can sync a .tmp mid-write and
# leave a partially-synced copy under the final name on another device, corrupting
# the very backup you'd restore from; it also thrashes the Mac (the SMA iCloud
# saga). Opt into an off-machine location explicitly via SMA_BACKUP_DIR (and
# prefer one outside ~/Library/Mobile Documents).
DEFAULT_BACKUP_DIR = Path(
    os.environ.get("SMA_BACKUP_DIR", "~/sma-backups")
).expanduser()

# 2026-07-30: an OFFSITE copy, deliberately IN iCloud — the opposite direction
# of the DEFAULT_BACKUP_DIR move above. The 2026-06-04 corruption happened
# because the sync daemon could observe a backup file mid-write; here we only
# ever copy a file that _verify_backup has already confirmed is a complete,
# readable DuckDB on local disk, and the copy itself lands via a temp name +
# rename (see _copy_offsite) so the daemon only ever sees a finished file.
DEFAULT_OFFSITE_BACKUP_DIR = Path(
    os.environ.get(
        "SMA_OFFSITE_BACKUP_DIR",
        "~/Library/Mobile Documents/com~apple~CloudDocs/sma-backups",
    )
).expanduser()

# Bound iCloud usage: each DB backup is ~330MB, so 7 copies is ~2.3GB.
OFFSITE_RETAIN_COUNT = 7

# 2026-08-24 incident: an out-of-band run of `python -m sma.backup run`
# against a near-empty worktree dev database (.worktrees/brk-earnings-fix/
# data/sma.duckdb, 1,323,008 bytes — freshly initialized, all 8 schema
# migrations applied within the same second) defaulted --asof to today (the
# CLI's real behavior when --asof is omitted, matching production) and had
# no way to redirect the offsite copy (no --offsite-dir flag exists; the
# offsite target is a module-level default bound at import time), so the
# stub landed at the REAL production offsite filename in the real iCloud
# folder. `_verify_backup` passed it: the tiny db still had a fully-migrated
# `_schema_version` table, so the schema-only check has no opinion on
# whether the file is anywhere near the size a healthy backup should be.
# MIN_BACKUP_SIZE_RATIO gates that: a backup under this fraction of (a) the
# source DB it was just copied from, or (b) the most recent prior backup
# already on disk, is rejected — never trusted offsite, never counted as a
# healthy backup.
MIN_BACKUP_SIZE_RATIO = 0.5

# 2026-08-24 audit: data/sentinels/ has no pruning policy and grows forever
# (~600 dated JSON files by 2026-08, one per (job, asof) pair, roughly 5-6/
# day across 9+ labels). Each file is tiny (a few KB) so this is a file-count
# problem, not a disk-space one, but nothing bounds it. The watchdog and
# preflight only ever read TODAY's sentinel (sma.watchdog.check,
# sma.live.preflight.run_preflight) — no code path looks back further than
# the current asof — so 60 days of history is generous headroom, not a
# functional risk to the watchdog's re-kick logic.
SENTINEL_RETAIN_DAYS = 60

# Never pruned, whatever their age. Reconcile's batch sentinel
# ("com.sma.live.reconcile.daily-<decide date>.json") is its only record that
# a batch was reconciled, and its resolver (sma.live.__main__.
# _unreconciled_batches) DOES look back past 60 days: a pruned batch sentinel
# made every afternoon's reconcile re-drain that batch and re-send its old
# drift alerts (flaw hunt 2026-10-01 A4). One file per trading day, a few
# hundred bytes each. The ".ran" liveness sentinel is a different label and is
# still pruned.
_SENTINEL_PRUNE_EXEMPT_PREFIXES = ("com.sma.live.reconcile.daily-",)

# 2026-08-24 audit: ~/Library/Logs/sma/*.log has no rotation and several
# files were multi-MB and still growing after 4 months (ingest.err.log
# 3.5MB, agents.err.log 2.7MB, autoresearch.nightly.err.log 2.6MB). Trim any
# oversized log down to its most recent tail — the tail is what's useful for
# debugging a job that just ran; the head is months-old debris.
DEFAULT_LOG_DIR = Path(
    os.environ.get("SMA_LOG_DIR", "~/Library/Logs/sma")
).expanduser()
LOG_ROTATE_MAX_BYTES = 2_000_000
LOG_ROTATE_KEEP_BYTES = 500_000

# com.sma.dashboard is a long-lived daemon (not a daily/weekly batch job) that
# keeps its stdout/stderr file descriptor open continuously. Rewriting the
# file out from under an actively-open fd via atomic replace() would orphan
# the daemon's future writes onto the old, now-unlinked inode — nothing would
# ever read them again, and the path would stay stuck at whatever length our
# rotation left it. The batch-job logs this targets are all closed between
# runs, so a fresh file at the same path is picked up cleanly on next launch.
_LOG_ROTATE_EXCLUDE = frozenset({
    "dashboard.log", "dashboard.out.log", "dashboard.err.log",
})


def _is_last_day_of_month(d: date) -> bool:
    return d.day == calendar.monthrange(d.year, d.month)[1]


def _is_first_saturday_of_month(d: date) -> bool:
    """True for the FIRST Saturday of `d`'s month only (day 1-7 inclusive
    AND a Saturday) -- gates the monthly DB compaction below to once/month,
    not every Saturday."""
    return d.weekday() == 5 and d.day <= 7


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


def _compact_database(db_path: Path) -> dict:
    """Rewrite db_path into a fresh, defragmented file via `COPY FROM
    DATABASE ... TO ...`, reclaiming free-block bloat in place.

    DuckDB 1.5.2 (the version this repo pins) does NOT reclaim free block
    space via CHECKPOINT or VACUUM -- both were empirically verified
    (2026-08-25, on a full 381,169,664-byte copy of the production DB) to
    leave file size AND pragma_database_size()'s free_blocks count (586 of
    1722 blocks, 34.03% free -- matching the audit's 34.7% finding)
    completely unchanged. The only mechanism that actually shrinks the file
    is attaching a brand-new empty database and using COPY FROM DATABASE to
    rewrite every table into contiguous fresh pages, then swapping it in --
    what Postgres calls VACUUM FULL; DuckDB just doesn't fold that behavior
    into the VACUUM keyword. Measured on that same copy: 381,169,664 ->
    281,030,656 bytes (reclaimed 100,139,008 bytes, 26.3%); free_blocks
    586 -> 0; all 19 table row counts identical before/after (verified via
    information_schema.tables + COUNT(*) on every table).

    Must run while writer_lock is held -- goes through Store like
    _checkpoint_db, which asserts that (raises WriterLockNotHeld
    otherwise). Callers wrap this in try/except: a compaction failure must
    never fail the backup -- db_path is untouched until the final atomic
    `tmp_path.replace(db_path)`, so a failure partway through (e.g. a full
    disk) leaves the original file exactly as it was.
    """
    tmp_path = db_path.with_name(db_path.name + ".compact.tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    size_before = db_path.stat().st_size

    store = Store(path=db_path).connect(read_only=False)
    try:
        main_name = store.conn.execute(
            "SELECT database_name FROM pragma_database_size()"
        ).fetchone()[0]
        store.conn.execute(f"ATTACH '{tmp_path}' AS compact_target (READ_ONLY FALSE)")
        try:
            store.conn.execute(f'COPY FROM DATABASE "{main_name}" TO compact_target')
        finally:
            store.conn.execute("DETACH compact_target")
    finally:
        store.close()

    tmp_path.replace(db_path)
    size_after = db_path.stat().st_size
    return {
        "ran": True,
        "size_before_bytes": size_before,
        "size_after_bytes": size_after,
        "reclaimed_bytes": size_before - size_after,
    }


def _verify_backup(backup_path: Path) -> bool:
    """Open the backup read-only and run a minimal query.

    Returns True if the file is a valid DuckDB with an _schema_version table,
    False on any error.
    """
    try:
        # read_only_connect (2026-08-05 audit): the backup file itself is a
        # standalone copy (no concurrent writer), but route through the same
        # helper for consistency/defence-in-depth.
        conn = read_only_connect(backup_path)
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


def _check_backup_size(
    backup_size: int,
    db_path: Path,
    backup_dir: Path,
    asof: date,
    min_ratio: float = MIN_BACKUP_SIZE_RATIO,
) -> str | None:
    """Return a reason string if `backup_size` looks implausibly small, else None.

    Two independent references, either of which can fail this check:

    1. The source DB's own current size — catches a literal mid-write/
       truncated copy (the file `_copy_offsite`'s tmp+rename makes atomic-
       looking but whose *content* was captured incomplete).
    2. The most recent prior backup already in `backup_dir` — catches a
       faithful, complete copy of the WRONG (much smaller) source entirely.
       This is the one that actually fired on 2026-08-24: the backup matched
       its source byte-for-byte, so check 1 alone would have missed it.

    Best-effort: a missing/unreadable reference is skipped (never raises),
    so a first-ever backup with no prior history is not penalized.
    """
    reasons: list[str] = []

    try:
        source_size = db_path.stat().st_size
        if source_size > 0 and backup_size < min_ratio * source_size:
            reasons.append(
                f"backup ({backup_size:,} bytes) is under {min_ratio:.0%} of "
                f"its source {db_path} ({source_size:,} bytes)"
            )
    except OSError:
        pass

    try:
        prior_dates = [d for d in _backup_dates_in_dir(backup_dir) if d < asof]
        if prior_dates:
            prior_path = backup_dir / f"sma-{max(prior_dates).isoformat()}.duckdb"
            prior_size = prior_path.stat().st_size
            if prior_size > 0 and backup_size < min_ratio * prior_size:
                reasons.append(
                    f"backup ({backup_size:,} bytes) is under {min_ratio:.0%} "
                    f"of the most recent prior backup {prior_path.name} "
                    f"({prior_size:,} bytes)"
                )
    except OSError:
        pass

    return "; ".join(reasons) if reasons else None


def _copy_offsite(backup_path: Path, offsite_dir: Path, retain: int = OFFSITE_RETAIN_COUNT) -> dict:
    """Copy the already-verified local backup file to `offsite_dir`.

    2026-06-04 context: backups were moved OFF iCloud after iCloud's sync
    daemon corrupted a backup written directly into the synced folder
    mid-write. This is the safe direction: `backup_path` is only ever called
    with a file `_verify_backup` has already confirmed complete on local
    disk, and the write into `offsite_dir` itself goes through a temp name
    then `os.rename()` into place (atomic within the same volume) so the
    sync daemon only ever observes a finished file, never a partial one.

    Also prunes offsite_dir to the `retain` most recent dated backups (each
    is ~330MB; 7 bounds iCloud usage to ~2.3GB).

    Raises on failure — the caller is responsible for catching so an offsite
    problem never fails the main (local) backup.
    """
    offsite_dir.mkdir(parents=True, exist_ok=True)
    dest = offsite_dir / backup_path.name
    tmp = offsite_dir / f"{backup_path.name}.tmp"
    shutil.copy2(backup_path, tmp)
    os.rename(tmp, dest)  # atomic within the same volume
    dest.chmod(0o600)

    existing = sorted(
        f for f in offsite_dir.glob("sma-*.duckdb") if _BACKUP_FILENAME_RE.match(f.name)
    )
    for old in existing[:-retain] if retain > 0 else existing:
        with contextlib.suppress(OSError):
            old.unlink()
            logger.info("backup: pruned old offsite copy {}", old.name)

    return {"attempted": True, "copied": True, "path": str(dest), "error": None}


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


def _prune_old_sentinels(asof: date, retain_days: int = SENTINEL_RETAIN_DAYS) -> int:
    """Delete dated sentinel files older than `retain_days` before `asof`.

    Best-effort: a per-file OSError is logged and skipped rather than
    aborting the backup job. Returns the count of files actually deleted.
    """
    target_dir = _sentinel_dir()
    if not target_dir.is_dir():
        return 0
    cutoff = asof - timedelta(days=retain_days)
    pruned = 0
    for f in target_dir.iterdir():
        if not f.is_file():
            continue
        m = _SENTINEL_FILENAME_RE.search(f.name)
        if not m:
            continue
        if f.name.startswith(_SENTINEL_PRUNE_EXEMPT_PREFIXES):
            continue
        try:
            file_date = date.fromisoformat(m.group(1))
        except ValueError:
            continue
        if file_date < cutoff:
            try:
                f.unlink()
                pruned += 1
            except OSError as exc:
                logger.warning("backup: could not prune sentinel {}: {}", f.name, exc)
    if pruned:
        logger.info("backup: pruned {} sentinel(s) older than {}", pruned, cutoff)
    return pruned


def _rotate_large_logs(
    log_dir: Path = DEFAULT_LOG_DIR,
    max_bytes: int = LOG_ROTATE_MAX_BYTES,
    keep_bytes: int = LOG_ROTATE_KEEP_BYTES,
) -> dict[str, int]:
    """Trim any `*.log` file in `log_dir` over `max_bytes` to its last `keep_bytes`.

    Rotation writes the kept tail to a `.tmp` file, then atomically renames
    it over the original — never an in-place truncate — so a process that
    opens the path fresh mid-rotation always sees a complete file, old or
    new, never a torn write. That does NOT protect a writer that already has
    the path open with an append fd across the rename: its writes keep
    landing on the old, now-unlinked inode and are never seen at the path
    again. `_LOG_ROTATE_EXCLUDE` exists for exactly that case — the one
    process in this system that holds such a long-lived fd (com.sma.
    dashboard). Every other log here belongs to a daily/weekly batch job
    that has exited long before the 22:00 backup runs.

    Best-effort: per-file errors are logged and skipped, never raised.
    Returns {filename: bytes_trimmed} for files actually rotated.
    """
    if not log_dir.is_dir():
        return {}
    trimmed: dict[str, int] = {}
    for f in sorted(log_dir.glob("*.log")):
        if f.name in _LOG_ROTATE_EXCLUDE:
            continue
        try:
            size = f.stat().st_size
            if size <= max_bytes:
                continue
            with f.open("rb") as fh:
                fh.seek(size - keep_bytes)
                tail = fh.read()
            tmp = f.with_name(f.name + ".tmp")
            tmp.write_bytes(tail)
            tmp.replace(f)
            trimmed[f.name] = size - len(tail)
            logger.info(
                "backup: rotated {} ({:.1f}MB -> {:.1f}MB)",
                f.name, size / 1_048_576, len(tail) / 1_048_576,
            )
        except OSError as exc:
            logger.warning("backup: could not rotate {}: {}", f.name, exc)
    return trimmed


def run_backup(
    *,
    db_path: Path,
    backup_dir: Path = DEFAULT_BACKUP_DIR,
    offsite_dir: Path = DEFAULT_OFFSITE_BACKUP_DIR,
    asof: date,
    retain_days: int = 30,
    retain_months: int = 12,
    offsite_retain: int = OFFSITE_RETAIN_COUNT,
    models_dir: Path | None = Path("models_artifacts"),
    notify_fn=notify_failure,
    sentinel_retain_days: int = SENTINEL_RETAIN_DAYS,
    log_dir: Path = DEFAULT_LOG_DIR,
    log_rotate_max_bytes: int = LOG_ROTATE_MAX_BYTES,
    log_rotate_keep_bytes: int = LOG_ROTATE_KEEP_BYTES,
) -> dict:
    """Perform one daily backup run.

    Steps:
    1. Acquire writer_lock(label="backup").
    2. CHECKPOINT the source DB so the WAL is flushed.
    2a. On the FIRST SATURDAY of each month only, compact the DB in place
        via COPY FROM DATABASE (see _compact_database) -- reclaims the free-
        block bloat CHECKPOINT/VACUUM can't touch. Non-fatal: a compaction
        failure is logged and recorded in the sentinel, never raised.
    3. Atomic copy: copy2 to a .tmp then rename to final destination.
    4. Verify the backup is readable.
    4a. If verified, copy the finished file offsite (see _copy_offsite).
    5. Apply retention policy (delete old files).
    5a. Prune sentinel files older than `sentinel_retain_days` (2026-08-24:
        data/sentinels/ had no pruning policy and grew unbounded).
    5b. Rotate any oversized log file under `log_dir` (2026-08-24: same
        finding for ~/Library/Logs/sma/*.log).
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

    # 15 min, not the 30s default: the 22:00 backup can queue behind a long
    # agents/evening run (it LOST that race once — 2026-06 logs — and the
    # night's backup was skipped). A backup is never urgent; waiting beats
    # failing. The watchdog/monitoring still catch a true wedge.
    with writer_lock(label="backup", timeout_s=900.0):
        # Step 2: flush WAL
        _checkpoint_db(db_path)

        # Step 2a: monthly compaction (first Saturday only). Runs BEFORE the
        # backup copy so the smaller, defragmented file is what gets backed
        # up too. Non-fatal: never blocks the backup itself.
        compaction: dict = {"ran": False}
        if _is_first_saturday_of_month(asof):
            try:
                compaction = _compact_database(db_path)
                logger.info(
                    "backup: monthly compaction reclaimed {:,} bytes ({:,} -> {:,})",
                    compaction["reclaimed_bytes"],
                    compaction["size_before_bytes"],
                    compaction["size_after_bytes"],
                )
            except Exception as exc:
                compaction = {"ran": False, "error": repr(exc)}
                logger.exception("backup: monthly compaction FAILED (non-fatal): {}", exc)

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

        # Step 4c: size sanity (2026-08-24 incident — see MIN_BACKUP_SIZE_RATIO).
        # Runs regardless of `verified`: a truncated copy can independently
        # fail schema verification too, but a faithful copy of the wrong
        # (much smaller) source passes verification cleanly and needs this
        # check to be caught at all.
        size_reason = _check_backup_size(backup_size, db_path, backup_dir, asof)
        size_sane = size_reason is None
        if not size_sane:
            logger.warning(
                "backup: size sanity check FAILED for {}: {}", backup_path, size_reason
            )
            notify_fn(
                title="sma: backup size looks implausibly small",
                message=(
                    f"The {asof} backup at {backup_path} is {backup_size:,} "
                    f"bytes — {size_reason}. This backup will NOT be copied "
                    "offsite or treated as healthy; investigate before "
                    "relying on it for recovery."
                ),
            )

        # Step 4a: offsite copy (2026-07-30). Only a VERIFIED, size-sane,
        # finished local file is ever copied — an unverified/corrupt/
        # implausibly-small backup must not be trusted offsite either.
        # Offsite failure (iCloud unreachable, disk full, permission denied,
        # ...) must NOT fail the main backup: log + notify a human, then
        # continue with retention/sentinel below.
        offsite_result: dict = {
            "attempted": False, "copied": False, "path": None, "error": None,
        }
        if verified and size_sane:
            offsite_result["attempted"] = True
            try:
                offsite_result.update(
                    _copy_offsite(backup_path, Path(offsite_dir), retain=offsite_retain)
                )
                logger.info("backup: offsite copy -> {}", offsite_result["path"])
            except Exception as e:
                offsite_result["error"] = repr(e)
                logger.exception("backup: offsite copy FAILED: {}", e)
                notify_fn(
                    title="sma: offsite backup copy FAILED",
                    message=(
                        f"copy of {backup_path.name} to offsite dir "
                        f"{offsite_dir} failed: {e!r}. The local verified "
                        "backup is unaffected."
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

        # Step 5a: sentinel pruning — best-effort, never fails the backup.
        try:
            pruned_sentinels = _prune_old_sentinels(
                asof=asof, retain_days=sentinel_retain_days
            )
        except Exception as exc:
            pruned_sentinels = 0
            logger.exception("backup: sentinel pruning FAILED: {}", exc)

        # Step 5b: log rotation — best-effort, never fails the backup.
        try:
            rotated_logs = _rotate_large_logs(
                log_dir=log_dir,
                max_bytes=log_rotate_max_bytes,
                keep_bytes=log_rotate_keep_bytes,
            )
        except Exception as exc:
            rotated_logs = {}
            logger.exception("backup: log rotation FAILED: {}", exc)

        # Step 6: sentinel (written while lock is still held)
        completed_at = datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        payload = {
            "label": "com.sma.backup.daily",
            "asof": asof.isoformat(),
            "completed_at": completed_at,
            "backup_path": str(backup_path),
            "backup_size_bytes": backup_size,
            "verified": verified,
            "size_check": {"passed": size_sane, "reason": size_reason},
            # Quality block so downstream readiness/monitoring fail closed on a
            # bad backup instead of treating any written sentinel as success.
            "models_files": models_files,
            "models_bytes": models_bytes,
            "quality": {
                "passed": verified and size_sane and models_ok,
                "blocking_failures": (
                    ([] if verified else ["backup_verification_failed"])
                    + ([] if size_sane else ["backup_size_implausible"])
                    + ([] if models_ok else ["models_backup_failed"])
                ),
            },
            "retained_dailies": retained_dailies,
            "retained_monthlies": retained_monthlies,
            # Not part of `quality` on purpose: an offsite failure must not
            # fail the (already-verified, already-local) main backup.
            "offsite": offsite_result,
            # Housekeeping, also deliberately outside `quality`: none of
            # these ever block the backup itself.
            "pruned_sentinels": pruned_sentinels,
            "rotated_logs": rotated_logs,
            "compaction": compaction,
        }
        write_sentinel(label="com.sma.backup.daily", asof=asof, payload=payload)
        logger.info("backup: sentinel written for {}", asof)

    return payload
