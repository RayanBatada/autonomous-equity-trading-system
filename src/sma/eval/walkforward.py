"""Walk-forward evaluation: retrain at boundaries through the window.

2026-06-12 (builder upgrade): the corrected-eval campaign used ONE model
trained at the window start, so late-window predictions came from a
months-stale model — unlike production, which retrains weekly. This harness
pre-trains a model at every Nth trading session of the window into a cache
dir; Predictor.latest_model_for_date then resolves the right boundary model
per asof automatically, and the standard evaluate_strategy run becomes a
true walk-forward.

Models are cached by (asof, target) filename — re-runs train nothing.
Training runs in a subprocess (the train CLI), --no-cv for speed.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import date
from pathlib import Path

from loguru import logger

__all__ = ["retrain_boundaries", "ensure_boundary_models"]


def retrain_boundaries(sessions: list[date], *, every: int) -> list[date]:
    """Every Nth trading session, always including the first. Empty in, empty out."""
    if not sessions:
        return []
    return list(sessions[::every])


def ensure_boundary_models(
    boundaries: list[date],
    *,
    models_dir: Path,
    db_path: Path = Path("data/sma.duckdb"),
    demean_labels: bool = False,
    objective: str | None = None,
    run_fn=None,
) -> int:
    """Train a --no-cv model at each boundary missing from `models_dir`.

    Returns the number of models trained. `run_fn` is injectable for tests;
    defaults to subprocess invocation of the train CLI.
    """
    from sma.model.persistence import latest_model_for_date

    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    trained = 0
    for b in boundaries:
        try:
            existing = latest_model_for_date(models_dir, b, "ret_30d_forward")
            # An exact boundary model is one trained AT b (filename carries
            # the train_end date); an older model means this boundary is
            # missing and the window would silently reuse stale weights.
            if existing.stem.split("_")[-2] == b.isoformat():
                continue
        except FileNotFoundError:
            pass
        cmd = [
            sys.executable, "-m", "sma.model", "train",
            "--asof", b.isoformat(), "--no-cv",
            "--db-path", str(db_path), "--models-dir", str(models_dir),
        ]
        if demean_labels:
            cmd.append("--demean-labels")
        if objective:
            cmd += ["--objective", objective]
        logger.info("walkforward: training boundary model {}", b)
        if run_fn is not None:
            run_fn(cmd)
        else:
            subprocess.run(cmd, check=True)
        trained += 1
    return trained
