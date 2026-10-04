"""Build the DEMEAN multi-regime walk-forward cache (faithful to live since
6/16), then run the strategy-config sweep on it.

This is the honest version of revalidate_parallel.py's RAW run: the live model
is demean + 2018 multi-regime, so the config decision must be re-checked on that
surface. Builds 7 boundary models (--demean-labels, every ~21 sessions) with a
small process pool, then execs revalidate_parallel.py against the demean cache.

Isolated on the DB copy. VAL only. Long-running (~1h) -- run in background.
"""

from __future__ import annotations

import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import duckdb

from sma.backtest.windows import window_dates
from sma.eval.walkforward import retrain_boundaries

SCRATCH = Path(
    "/private/tmp/claude-501/-Users-youruser-SecondBrain/"
    "f0fae5bb-6213-478f-b88e-e8747330c663/scratchpad"
)
DB = SCRATCH / "sma-eval.duckdb"
CACHE = SCRATCH / "wf-demean-models"
BUILD_WORKERS = 2  # memory-bound on a 16GB machine


def _train(iso: str):
    # Resumable: skip a boundary whose demean model already exists in CACHE
    # (CACHE is demean-only, so any model trained AT this boundary is ours).
    from datetime import date as _date

    from sma.model.persistence import latest_model_for_date

    try:
        existing = latest_model_for_date(CACHE, _date.fromisoformat(iso), "ret_30d_forward")
        if existing.stem.split("_")[-2] == iso:
            return iso, 0, "(cached — skipped)"
    except FileNotFoundError:
        pass
    cmd = [
        sys.executable, "-m", "sma.model", "train",
        "--asof", iso, "--no-cv", "--demean-labels",
        "--db-path", str(DB), "--models-dir", str(CACHE),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return iso, r.returncode, (r.stderr or "")[-300:]


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    start, end = window_dates("val")
    con = duckdb.connect(str(DB), read_only=True)
    sessions = [
        r[0] for r in con.execute(
            "SELECT DISTINCT date FROM prices WHERE date BETWEEN ? AND ? ORDER BY date",
            [start, end],
        ).fetchall()
    ]
    con.close()
    boundaries = [b.isoformat() for b in retrain_boundaries(sessions, every=21)]
    print(f"building {len(boundaries)} demean boundary models "
          f"({BUILD_WORKERS} workers): {boundaries}", flush=True)

    with ProcessPoolExecutor(max_workers=BUILD_WORKERS) as ex:
        futs = {ex.submit(_train, b): b for b in boundaries}
        for fut in as_completed(futs):
            iso, rc, err = fut.result()
            print(f"  built {iso} rc={rc}" + (f" ERR {err}" if rc else ""), flush=True)

    print("\n=== running sweep on demean cache ===", flush=True)
    env = dict(os.environ)
    env["SMA_REVAL_CACHE"] = str(CACHE)
    env["SMA_REVAL_LABEL"] = "DEMEAN"
    env["SMA_REVAL_WORKERS"] = "4"
    subprocess.run(
        [sys.executable, "scripts/revalidate_parallel.py"], env=env, check=False
    )


if __name__ == "__main__":
    main()
