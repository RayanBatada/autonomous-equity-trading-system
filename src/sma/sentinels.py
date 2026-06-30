"""Atomic readiness sentinels: per-(job, asof) JSON files signaling completion.

Use these instead of querying DuckDB to avoid the no-mixed-reader-with-writer
constraint. Sentinel writes are atomic via tempfile + os.replace (POSIX rename
is atomic on the same filesystem).

Ordering contract: each scheduled job writes its sentinel WHILE STILL HOLDING
the writer_lock (after closing DuckDB but before releasing the flock). This
serializes sentinel writes for the same (label, asof) so that a slow late
writer cannot overwrite a faster newer one.

Defense in depth: write_sentinel ALSO checks for monotonicity. If the new
payload has a run_id less than the existing sentinel's run_id, the write is
suppressed (with a warning). Same for completed_at timestamps when run_id is
absent. This catches edge cases where a job bypasses the writer_lock (manual
edits, recovery scripts) AND protects against bugs where a job forgets to
hold the lock.

The sentinel includes whatever the job's downstream consumers need to determine
readiness (sources_ok, quality verdict, run_id, etc.). Recommended fields:
- run_id: int (monotonic per-job counter)
- completed_at: ISO-8601 timestamp
- quality: dict with at least {"passed": bool, "blocking_failures": list[str]}

Public surface:
- write_sentinel(label, asof, payload) -> Path | None: atomically write; returns
  None if suppressed by monotonicity check
- read_sentinel(label, asof) -> dict | None: read, None if missing
- sentinel_path(label, asof) -> Path: the canonical path
- ingest_succeeded_today(asof) -> bool: shorthand for the most-common check
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from datetime import date
from pathlib import Path

from loguru import logger

__all__ = [
    "SENTINEL_DIR",
    "ingest_succeeded_today",
    "read_sentinel",
    "sentinel_dir",
    "sentinel_path",
    "write_sentinel",
]


SENTINEL_DIR = Path("data/sentinels")


def sentinel_dir() -> Path:
    """Resolve the sentinel directory: $SMA_SENTINEL_DIR if set, else SENTINEL_DIR.

    The env override exists so tests and ad-hoc tooling can fully isolate
    sentinel state. The CWD-relative default let the integration suite write
    REAL sentinels into data/sentinels/ when run from the repo root
    (2026-06-09); tests now always run with the env var pointed at a temp dir
    (autouse fixture in tests/conftest.py), and subprocess-spawned CLIs
    inherit it.
    """
    env = os.environ.get("SMA_SENTINEL_DIR")
    return Path(env) if env else SENTINEL_DIR


def sentinel_path(*, label: str, asof: date) -> Path:
    return sentinel_dir() / f"{label}-{asof.isoformat()}.json"


def _is_strictly_newer(new_payload: dict, existing_payload: dict) -> bool:
    """True iff `new_payload` should replace `existing_payload` per the
    monotonicity contract. Compares run_id first (numeric), falling back to
    completed_at (ISO-8601 string lexicographic). Returns True if neither
    payload has either field (callers responsible for ordering).

    Critical edge case: if the existing payload has a real run_id (non-None)
    and the new payload has run_id=None (e.g. a holiday-skip sentinel), the
    new payload must NOT replace the real one. A None run_id is treated as
    "no run_id" and falls through to the completed_at comparison only when
    the existing payload also has no run_id. If the existing payload has a
    run_id and the new payload does not, we return False unconditionally.
    """
    new_run = new_payload.get("run_id")
    old_run = existing_payload.get("run_id")
    if new_run is not None and old_run is not None:
        return new_run > old_run
    # If existing has a real run_id but new does not, new is never newer.
    if old_run is not None and new_run is None:
        return False
    # Both have no run_id: fall back to completed_at comparison — parsed as
    # datetimes, NOT strings. Lexicographic compare suppressed genuinely-newer
    # same-second fractional timestamps ('.': 46 < 'Z': 90), e.g.
    # 20:00:01.9Z read as OLDER than 20:00:01Z (Codex module review
    # 2026-06-11). Unparseable forms fall back to the string compare.
    new_ts = new_payload.get("completed_at")
    old_ts = existing_payload.get("completed_at")
    if new_ts is not None and old_ts is not None:
        try:
            from datetime import datetime as _dt

            new_dt = _dt.fromisoformat(str(new_ts).replace("Z", "+00:00"))
            old_dt = _dt.fromisoformat(str(old_ts).replace("Z", "+00:00"))
            return new_dt > old_dt
        except ValueError:
            return str(new_ts) > str(old_ts)
    return True


def _jsonsafe(obj):
    """Recursively replace NaN/inf floats with None so the sentinel is strict
    JSON (json.dump's default emits bare NaN — invalid for jq/strict parsers;
    the weekly-retrain sentinel hit this with cv_rmse=NaN on tiny folds)."""
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _jsonsafe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonsafe(v) for v in obj]
    return obj


def write_sentinel(*, label: str, asof: date, payload: dict) -> Path | None:
    """Atomically write `data/sentinels/<label>-<asof>.json`.

    Atomicity: write to a tempfile in the same directory, then os.replace().
    POSIX rename within the same filesystem is atomic.

    Monotonicity: if an existing sentinel has a strictly newer run_id (or
    completed_at when run_id is absent), the write is SUPPRESSED and the
    function returns None with a warning logged. Callers normally serialize
    on the writer_lock so this branch should be rare; treat it as a guard
    against bugs.

    Returns the target Path on success, or None if the write was suppressed.
    """
    target_dir = sentinel_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    target = sentinel_path(label=label, asof=asof)

    if target.exists():
        try:
            existing = json.loads(target.read_text())
        except (OSError, json.JSONDecodeError):
            existing = {}
        if existing and not _is_strictly_newer(payload, existing):
            new_run = payload.get("run_id", payload.get("completed_at"))
            old_run = existing.get("run_id", existing.get("completed_at"))
            logger.warning(
                f"sentinel write suppressed: {target.name} new={new_run} <= existing={old_run}"
            )
            return None

    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{label}-{asof.isoformat()}-",
        suffix=".json",
        dir=target_dir,
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(
                _jsonsafe(payload), f, sort_keys=True, separators=(",", ":"),
                allow_nan=False,
            )
        os.replace(tmp_name, target)
    except Exception:
        Path(tmp_name).unlink(missing_ok=True)
        raise
    return target


def read_sentinel(*, label: str, asof: date) -> dict | None:
    """Return the parsed sentinel JSON, or None if it doesn't exist yet.

    Does NOT raise on missing file; callers use None to mean "not yet written".
    Raises on JSON parse error or filesystem read error (those are bugs, not
    expected states).
    """
    p = sentinel_path(label=label, asof=asof)
    if not p.exists():
        return None
    return json.loads(p.read_text())


def ingest_succeeded_today(*, asof: date) -> bool:
    """Shorthand: did com.sma.ingest.daily complete with a passing quality verdict for `asof`?

    Replaces the legacy `ingest_succeeded_today` that queried DuckDB. Reads only
    the sentinel file; never opens DuckDB. Returns False on missing sentinel OR
    `quality.passed = False`.
    """
    sentinel = read_sentinel(label="com.sma.ingest.daily", asof=asof)
    if sentinel is None:
        return False
    return bool(sentinel.get("quality", {}).get("passed", False))
