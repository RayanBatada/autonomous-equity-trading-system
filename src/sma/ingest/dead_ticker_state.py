"""Tiny persisted state for no_dead_or_frozen_tickers's per-ticker dedup
(sma.ingest.quality.notify_new_dead_or_frozen_tickers).

Same convention as sma.monitoring.regime_state: one small file, atomically
OVERWRITTEN every ingest run with the CURRENTLY-flagged ticker set -- not a
new dated file appended forever. A ticker present in the prior state with the
SAME kind ("frozen"/"stale") is not re-notified (pages once, not nightly). A
ticker that heals (no longer flagged) is simply dropped from the new state,
so a later recurrence pages again rather than staying silent off a stale
record.

Atomic write: tempfile in the same directory + os.replace() (POSIX rename is
atomic on the same filesystem) -- same pattern as sma.sentinels.write_sentinel
and sma.monitoring.regime_state.write_regime_state.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

__all__ = [
    "DEFAULT_STATE_PATH",
    "read_dead_ticker_state",
    "state_path",
    "write_dead_ticker_state",
]

DEFAULT_STATE_PATH = Path("data/state/dead_frozen_tickers.json")


def state_path() -> Path:
    """Resolve the state file path: $SMA_DEAD_TICKER_STATE_PATH if set, else
    DEFAULT_STATE_PATH. The env override exists so tests can fully isolate
    state (same convention as sma.monitoring.regime_state.state_path's
    SMA_REGIME_STATE_PATH)."""
    env = os.environ.get("SMA_DEAD_TICKER_STATE_PATH")
    return Path(env) if env else DEFAULT_STATE_PATH


def read_dead_ticker_state() -> dict:
    """Return the parsed state JSON: {ticker: {kind, last_price_date,
    run_length, first_flagged_asof}}. {} if the file doesn't exist yet (first
    run, or a fresh SMA_DEAD_TICKER_STATE_PATH)."""
    p = state_path()
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def write_dead_ticker_state(payload: dict) -> None:
    """Atomically overwrite the state file with `payload`."""
    p = state_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".dead-ticker-state-", suffix=".json", dir=p.parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, sort_keys=True, separators=(",", ":"))
        os.replace(tmp_name, p)
    except Exception:
        Path(tmp_name).unlink(missing_ok=True)
        raise
