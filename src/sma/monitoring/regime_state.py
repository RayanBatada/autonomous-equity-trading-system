"""Tiny persisted state for the regime-turn crossing detector
(sma.monitoring.check_regime_turn).

Deliberately NOT a per-day job-completion sentinel like sma.sentinels
(data/sentinels/<label>-<date>.json, one new file every day forever, pruned
monthly by the backup job) -- this is a single, continuously-overwritten
file recording only the LAST-OBSERVED trailing-IC regime level, so the
crossing detector can tell "just turned positive" apart from "still
positive from yesterday" without scanning history. One small file, not one
per day.

Atomic write: tempfile in the same directory + os.replace() (POSIX rename
is atomic on the same filesystem) -- same pattern as sma.sentinels.write_sentinel.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

__all__ = ["DEFAULT_STATE_PATH", "read_regime_state", "state_path", "write_regime_state"]

DEFAULT_STATE_PATH = Path("data/state/regime_ic_10d.json")


def state_path() -> Path:
    """Resolve the state file path: $SMA_REGIME_STATE_PATH if set, else
    DEFAULT_STATE_PATH. The env override exists so tests can fully isolate
    state (same convention as sma.sentinels.sentinel_dir's SMA_SENTINEL_DIR)."""
    env = os.environ.get("SMA_REGIME_STATE_PATH")
    return Path(env) if env else DEFAULT_STATE_PATH


def read_regime_state() -> dict | None:
    """Return the parsed state JSON, or None if it doesn't exist yet (first
    run, or a fresh SMA_REGIME_STATE_PATH)."""
    p = state_path()
    if not p.exists():
        return None
    return json.loads(p.read_text())


def write_regime_state(payload: dict) -> None:
    """Atomically overwrite the state file with `payload`."""
    p = state_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".regime-state-", suffix=".json", dir=p.parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, sort_keys=True, separators=(",", ":"))
        os.replace(tmp_name, p)
    except Exception:
        Path(tmp_name).unlink(missing_ok=True)
        raise
