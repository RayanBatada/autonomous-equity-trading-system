"""Model artifact persistence + predictions table writes."""

import json
import math
import pickle
from datetime import UTC, date, datetime
from pathlib import Path

import xgboost as xgb
from loguru import logger

from sma.model.ensemble import EnsembleModel, ensemble_random_states, ensemble_size

# 2026-08-24 audit: models_artifacts/ has no pruning policy and grows
# unboundedly — 13+ weekly pickles (~355KB-800KB each) with nothing bounding
# the count. prune_old_artifacts() keeps this many of the most-recently-
# created promoted artifacts, plus whichever one is currently serving (never
# deleted, however old — see prune_old_artifacts).
DEFAULT_ARTIFACT_KEEP_RECENT = 8

# A retrain is auto-deployed only if its out-of-sample (walk-forward CV) RMSE is
# no worse than this ratio of the currently-deployed model's. Week-to-week noise
# is a few %, so 1.10 (10% worse) blocks real degradation without tripping on
# normal variation. Tune via should_promote(..., max_ratio=).
GATE_MAX_CV_RMSE_RATIO = 1.10
# Held-out rank-IC NOISE BAND. A single walk-forward CV-IC estimate has a
# standard error of ~σ_date/√(n_date-ICs) ≈ 0.07/√100 ≈ 0.007, so this -0.02
# floor is ~3 SE below zero: it blocks models that are STATISTICALLY worse
# than chance (clear ranking inversion — the 2026-06-13 regime-luck failure)
# while tolerating ordinary measurement noise around true-zero edge. It is a
# noise band, NOT "any negative IS blocked" — a model in [-0.02, 0) deploys
# (statistically indistinguishable from chance). RMSE can't see ranking
# inversion at all, which is why this gate exists alongside it.
GATE_MIN_CV_IC = -0.02


def write_predictions(
    db_path: Path,
    asof_date: date,
    target: str,
    model_id: str,
    predictions: dict[str, float],
) -> int:
    """Write predictions for one (asof_date, target, model_id) batch.

    Routes all writes through Store so the writer_lock assertion in
    Store.connect(read_only=False) fires when no lock is held. Callers MUST
    hold writer_lock before calling this function.

    UPSERTs on the composite PK (asof_date, ticker, target, model_id) so
    re-running the same prediction job is idempotent. Returns number of rows
    written. Auto-migrates the DB to the current schema if needed (so a fresh
    ingest-only DB grows the predictions table on first write).
    """
    # Rebind to a name that doesn't collide with the SQL table name.
    # DuckDB's auto-replacement-scan walks the calling frame for Python
    # objects matching SQL identifiers; `predictions` would otherwise be
    # treated as a table source and error out.
    pred_map = predictions
    del predictions

    if not pred_map:
        return 0

    from sma.ingest.store import Store

    rows = [
        {
            "asof_date": asof_date,
            "ticker": t,
            "target": target,
            "predicted_value": float(v),
            "model_id": model_id,
        }
        for t, v in pred_map.items()
    ]

    # Store.connect(read_only=False) applies migrations (creating predictions
    # table if missing) and raises WriterLockNotHeld if the caller skipped
    # writer_lock. Single open for both migrate + write.
    store = Store(path=str(db_path))
    store.connect(read_only=False)
    try:
        for r in rows:
            store.conn.execute(
                "DELETE FROM predictions WHERE asof_date=$asof_date "
                "AND ticker=$ticker AND target=$target AND model_id=$model_id",
                {k: r[k] for k in ("asof_date", "ticker", "target", "model_id")},
            )
        store.conn.executemany(
            "INSERT INTO predictions (asof_date, ticker, target, predicted_value, model_id) "
            "VALUES ($asof_date, $ticker, $target, $predicted_value, $model_id)",
            rows,
        )
    finally:
        store.close()
    return len(rows)


def _library_versions() -> dict[str, str]:
    """Versions of the libraries whose numerics determine a model's output.

    Read at save time from the running interpreter — never hard-coded, or the
    provenance would drift from reality on the first upgrade. Kept to the
    libraries that actually affect scoring (plus the interpreter itself);
    a full `pip freeze` would bury the signal.
    """
    import platform

    import numpy as _np
    import pandas as _pd
    import sklearn as _sklearn

    return {
        "xgboost": xgb.__version__,
        "scikit-learn": _sklearn.__version__,
        "numpy": _np.__version__,
        "pandas": _pd.__version__,
        "python": platform.python_version(),
    }


def save_model(
    model: xgb.XGBRegressor | xgb.XGBRanker | EnsembleModel,
    *,
    hyperparams: dict,
    feature_names: list[str],
    train_end_date: date,
    train_rows: int,
    train_rmse: float,
    code_commit: str,
    training_duration_seconds: float,
    output_dir: Path,
    target: str = "ret_30d_forward",
    cv_rmse: float | None = None,
    cv_ic: float | None = None,
    train_start: date | None = None,
    promoted: bool = True,
    objective: str = "reg",
    label_type: str = "raw",
) -> tuple[Path, Path]:
    """Pickle the model + write a JSON metadata sidecar.

    `model` may be a single estimator or an EnsembleModel holding N boosters;
    an ensemble is ONE pickle like any other artifact, so the file layout,
    naming, and discovery are unchanged. `ensemble_seeds` /
    `ensemble_random_states` in the sidecar say how many boosters are inside
    and which seeds built them.

    Returns (pkl_path, json_path). Filenames:
        xgb_<target>_<train_end_date>_<commit_sha8>.pkl
        xgb_<target>_<train_end_date>_<commit_sha8>.json
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    sha8 = code_commit[:8] if code_commit else "nocommit"
    base = f"xgb_{target}_{train_end_date.isoformat()}_{sha8}"
    # Uniquify on collision: the Monday retrain and the autoresearch search can
    # both save the same date at the same commit — the second writer used to
    # silently OVERWRITE the first (deployed!) artifact (review 2026-07-01).
    # A "-N" suffix on the sha segment keeps both; latest_model_for_date parses
    # the date from parts[-2] (unaffected) and breaks same-date ties by
    # created_at, so the newest promoted model still serves.
    n = 2
    while (output_dir / f"{base}.pkl").exists() or (output_dir / f"{base}.json").exists():
        base = f"xgb_{target}_{train_end_date.isoformat()}_{sha8}-{n}"
        n += 1
    pkl_path = output_dir / f"{base}.pkl"
    json_path = output_dir / f"{base}.json"

    with pkl_path.open("wb") as f:
        pickle.dump(model, f)

    metadata = {
        "model_id": base,
        "target": target,
        "objective": objective,
        # raw 30d return vs cross-sectionally demeaned (alpha) target — the
        # deploy gate must not compare RMSE across types (2026-06-13).
        "label_type": label_type,
        # Training-window start — RMSE isn't comparable across different
        # training distributions, so a change here force-promotes on the IC
        # floor like a label-type change (2026-06-16 multi-regime).
        "train_start": train_start.isoformat() if train_start else None,
        "train_end_date": train_end_date.isoformat(),
        "train_rows": train_rows,
        # train_rmse = in-sample fit error; cv_rmse = walk-forward out-of-sample
        # error (the one that says how the model GENERALIZES). The gap between
        # them is the overfit signal. cv_rmse is None for --no-cv / legacy runs.
        "train_rmse": train_rmse,
        "cv_rmse": cv_rmse,
        # Held-out rank-IC (out-of-sample ranking edge) — RMSE misses ranking
        # ability; this is the metric that tracks actual edge (2026-06-13).
        "cv_ic": cv_ic,
        # Self-describing deploy status so discovery can filter on the flag, not
        # only on directory position (rejected models live in a subdir today).
        "promoted": promoted,
        "feature_names": feature_names,
        "hyperparams": hyperparams,
        # Seed-ensemble provenance (2026-08-24). N boosters live inside the ONE
        # pickle above; these say how many and which random_states built them,
        # so a deployed artifact self-describes its ensemble instead of leaving
        # it to be inferred from the pickle. ABSENT on every artifact written
        # before this date — a reader must treat "missing" as 1, which is what
        # ensemble_size() returns for a bare estimator.
        "ensemble_seeds": ensemble_size(model),
        "ensemble_random_states": ensemble_random_states(model),
        "code_commit": code_commit,
        # Library provenance. `code_commit` says which SOURCE produced this
        # model; this says which LIBRARIES did. Models are persisted with
        # `pickle.dump`, and an xgboost pickle is not guaranteed to load — or to
        # score identically — under a different xgboost build, so a deployed
        # artifact is only as reproducible as the stack that made it. Two
        # project-specific reasons this is load-bearing:
        #   1. The pyproject floor was xgboost >=2.0 until 2026-07-30, when it
        #      was raised to >=3.2 (commit 817a441) to match what actually
        #      trained the deployed models (3.2.0). Before that raise, a
        #      resolve that ignored uv.lock could have pulled a different
        #      MAJOR version and failed to load the deployed .pkl on a live
        #      book — the floor now matches, but this metadata remains the
        #      record of what actually built each artifact.
        #   2. Backtest and live share one code path — "what gets measured is
        #      what trades". Silent numerical drift from a library upgrade
        #      breaks that parity; without this there is nothing to diagnose it
        #      against. (Added 2026-07-30.)
        "library_versions": _library_versions(),
        "training_duration_seconds": training_duration_seconds,
        "created_at": datetime.utcnow().isoformat() + "Z",
    }
    with json_path.open("w") as f:
        json.dump(metadata, f, indent=2)

    return pkl_path, json_path


def load_model(pkl_path: Path) -> xgb.XGBRegressor | xgb.XGBRanker | EnsembleModel:
    """Unpickle a saved model. Raises FileNotFoundError if missing.

    Backward compatible by construction: artifacts written before 2026-08-24
    (the currently promoted one included) pickled a bare estimator, and they
    still unpickle to exactly that — no shim, no migration, no rewrite. Newer
    ensemble artifacts unpickle to an EnsembleModel. Callers only ever need
    `.predict(X)` and the sklearn metadata attributes, which both shapes
    answer identically.
    """
    if not pkl_path.exists():
        raise FileNotFoundError(f"Model not found: {pkl_path}")
    with pkl_path.open("rb") as f:
        return pickle.load(f)


def load_metadata(json_path: Path) -> dict:
    """Read the JSON sidecar."""
    if not json_path.exists():
        raise FileNotFoundError(f"Metadata not found: {json_path}")
    return json.loads(json_path.read_text())


def _model_creation_dt(pkl: Path) -> datetime:
    """When a model was created, for breaking same-train-date ties. Prefers the
    metadata `created_at`; falls back to file mtime. ALWAYS returns a tz-aware
    datetime, and never raises on a missing/corrupt/non-dict sidecar or an odd
    timestamp — a bad sidecar must not crash live model selection (Codex review
    2026-06-19)."""
    try:
        ca = load_metadata(pkl.with_suffix(".json")).get("created_at")
        if ca:
            dt = datetime.fromisoformat(str(ca).replace("Z", "+00:00"))
            # A created_at without an offset is naive; treat it as UTC so it
            # compares against the aware mtime fallback without raising.
            return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)
    except Exception:  # noqa: BLE001 - any bad sidecar -> mtime, never crash selection
        pass
    return datetime.fromtimestamp(pkl.stat().st_mtime, tz=UTC)


def latest_model_for_date(
    models_dir: Path,
    asof_date: date,
    target: str = "ret_30d_forward",
) -> Path:
    """Find the most recent saved model whose train_end_date <= asof_date.

    Returns the .pkl path. Raises FileNotFoundError if no eligible model exists.
    Same-train-date ties are broken by `created_at` (latest wins): on Mondays the
    04:00 retrain and the 07:00 autoresearch both produce a model dated today, and
    autoresearch's IC-gated winner (created later) must serve — date-only sorting
    left this to arbitrary glob order (2026-06-19).
    """
    if not models_dir.exists():
        raise FileNotFoundError(f"Models directory not found: {models_dir}")

    candidates: list[tuple[date, Path]] = []
    for pkl in models_dir.glob(f"xgb_{target}_*.pkl"):
        # filename: xgb_<target>_<YYYY-MM-DD>_<sha8>.pkl
        # target may contain underscores, so the date is always parts[-2].
        parts = pkl.stem.split("_")
        try:
            train_end = date.fromisoformat(parts[-2])
        except (ValueError, IndexError):
            continue
        if train_end <= asof_date:
            candidates.append((train_end, pkl))

    if not candidates:
        raise FileNotFoundError(
            f"No model found in {models_dir} with train_end_date <= {asof_date}"
        )

    max_date = max(d for d, _ in candidates)
    tied = [pkl for d, pkl in candidates if d == max_date]
    if len(tied) == 1:
        return tied[0]
    # Latest-created wins; ties on created_at break by filename for determinism.
    return max(tied, key=lambda p: (_model_creation_dt(p), p.name))


def prune_old_artifacts(
    models_dir: Path,
    asof_date: date,
    target: str = "ret_30d_forward",
    keep_recent: int = DEFAULT_ARTIFACT_KEEP_RECENT,
) -> list[str]:
    """Delete promoted artifacts beyond a retention window.

    Keeps the union of two sets, never deletes anything else:
      1. Whichever artifact `latest_model_for_date(models_dir, asof_date,
         target)` would currently serve — regardless of how old it is. This
         is the hard constraint: the live/serving model is never pruned.
      2. The `keep_recent` most-recently-created artifacts (by the same
         `created_at`-preferred, mtime-fallback ordering `latest_model_for_
         date` already uses for tie-breaks).

    Only scans `models_dir` directly (glob(), not rglob()) — exactly like
    `latest_model_for_date` — so `models_dir/"rejected"` (quarantined,
    non-promoted candidates) is never touched. Each deletion removes the
    `.pkl` and its `.json` sidecar together. Best-effort per file: an OSError
    on one delete is logged and skipped, never raised, so a retrain job's
    pruning step can't fail the job over a locked/already-gone file.

    Returns the model_ids (pkl stems) actually deleted.
    """
    if not models_dir.is_dir():
        return []

    candidates: list[Path] = []
    for pkl in models_dir.glob(f"xgb_{target}_*.pkl"):
        parts = pkl.stem.split("_")
        try:
            date.fromisoformat(parts[-2])
        except (ValueError, IndexError):
            continue
        candidates.append(pkl)

    if len(candidates) <= keep_recent:
        return []

    try:
        serving = latest_model_for_date(models_dir, asof_date, target)
    except FileNotFoundError:
        serving = None

    ranked = sorted(candidates, key=lambda p: (_model_creation_dt(p), p.name), reverse=True)
    keep = set(ranked[:keep_recent])
    if serving is not None:
        keep.add(serving)

    deleted: list[str] = []
    for pkl in candidates:
        if pkl in keep:
            continue
        for f in (pkl, pkl.with_suffix(".json")):
            try:
                f.unlink(missing_ok=True)
            except OSError as exc:
                logger.warning("prune_old_artifacts: could not delete {}: {}", f, exc)
        deleted.append(pkl.stem)

    if deleted:
        logger.info(
            "prune_old_artifacts: deleted {} artifact(s): {}", len(deleted), deleted
        )
    return deleted


def incumbent_label_type(
    models_dir: Path,
    asof_date: date,
    target: str = "ret_30d_forward",
) -> str | None:
    """label_type ('raw'/'demean') of the model deployed for `asof_date`, or
    None if no eligible model. Legacy artifacts (no field) read as 'raw'.
    The deploy gate force-promotes on a label_type transition since RMSE is
    not comparable across targets (2026-06-13)."""
    try:
        pkl = latest_model_for_date(models_dir, asof_date, target)
        meta = load_metadata(pkl.with_suffix(".json"))
    except FileNotFoundError:
        return None
    return meta.get("label_type", "raw")


def incumbent_train_start(
    models_dir: Path, asof_date: date, target: str = "ret_30d_forward",
) -> str | None:
    """train_start (ISO) of the deployed model for asof, or None if absent
    (legacy / no model). A change vs the new run means the RMSE comparison is
    invalid (different training distribution) — gate on the IC floor instead."""
    try:
        pkl = latest_model_for_date(models_dir, asof_date, target)
        meta = load_metadata(pkl.with_suffix(".json"))
    except FileNotFoundError:
        return None
    return meta.get("train_start")


def incumbent_cv_rmse(
    models_dir: Path,
    asof_date: date,
    target: str = "ret_30d_forward",
) -> float | None:
    """Out-of-sample CV RMSE of the model currently deployed for `asof_date`.

    Returns None when no eligible model exists. Prefers the `cv_rmse` field;
    falls back to legacy `train_rmse` (older artifacts stored the CV value
    there before train_rmse/cv_rmse were split). Used by the retrain deploy
    gate to compare a fresh model against the incumbent.
    """
    try:
        pkl = latest_model_for_date(models_dir, asof_date, target)
    except FileNotFoundError:
        return None
    try:
        meta = load_metadata(pkl.with_suffix(".json"))
    except FileNotFoundError:
        return None
    val = meta.get("cv_rmse")
    if val is None:
        val = meta.get("train_rmse")  # legacy artifacts stored CV here
    if val is None:
        return None
    val = float(val)
    return None if math.isnan(val) else val


def incumbent_cv_ic(
    models_dir: Path,
    asof_date: date,
    target: str = "ret_30d_forward",
) -> float | None:
    """Out-of-sample rank-IC (`cv_ic`) of the model currently deployed for
    `asof_date`, or None when there's no eligible model or the field is
    missing/NaN. The autoresearch search gate compares a candidate's CV-IC
    against this so it only promotes a config that ranks MEASURABLY better OOS
    than what's live — promoting on RMSE would re-introduce the ranking-blindness
    the IC work removed (2026-06-17)."""
    try:
        pkl = latest_model_for_date(models_dir, asof_date, target)
        meta = load_metadata(pkl.with_suffix(".json"))
    except FileNotFoundError:
        return None
    val = meta.get("cv_ic")
    if val is None:
        return None
    val = float(val)
    return None if math.isnan(val) else val


def incumbent_hyperparams(
    models_dir: Path,
    asof_date: date,
    target: str = "ret_30d_forward",
) -> dict | None:
    """Hyperparameters of the model deployed for `asof_date`, or None when there
    is no eligible model or none were recorded. Lets the autoresearch search
    re-measure the incumbent's OWN config on the current data (an apples-to-apples
    comparison instead of a metric stored when it was trained on older data), and
    lets the weekly retrain KEEP a good config across weeks instead of re-rolling
    it (2026-06-23)."""
    try:
        pkl = latest_model_for_date(models_dir, asof_date, target)
        meta = load_metadata(pkl.with_suffix(".json"))
    except FileNotFoundError:
        return None
    hp = meta.get("hyperparams")
    return hp if isinstance(hp, dict) and hp else None


def passes_ic_floor(
    cv_ic: float | None, *, cv_ran: bool = True, floor: float = GATE_MIN_CV_IC
) -> bool:
    """True unless the held-out rank-IC is statistically below chance.

    The top-level deploy safety: a model that ranks worse than chance OOS must
    not go live even if RMSE looks fine or it's a clean label-type transition.

    `cv_ran` distinguishes two sources of a missing IC (Codex review 2026-06-15):
    - cv_ran=False (operator chose --no-cv): no IC evidence is REQUIRED, pass.
    - cv_ran=True but cv_ic is None/nan: CV ran yet produced NO usable IC
      (degenerate cross-sections, constant predictions, all dates skipped) —
      that is missing EVIDENCE, not a passing grade, so FAIL CLOSED.
    A finite IC is compared to the noise-band floor.
    """
    finite = cv_ic is not None and not math.isnan(cv_ic)
    if not finite:
        return not cv_ran  # no-cv passes; failed-CV fails closed
    return cv_ic >= floor


def should_promote(
    new_cv_rmse: float | None,
    incumbent_cv_rmse: float | None,
    *,
    max_ratio: float = GATE_MAX_CV_RMSE_RATIO,
) -> bool:
    """Deploy gate: should a freshly trained model replace the incumbent?

    Promote unless the new model is materially WORSE out-of-sample than the
    incumbent (new CV RMSE > incumbent * max_ratio). Promote when:
      - there's no incumbent or its metric is unusable (nothing to compare), or
      - the new model has no CV number (e.g. --no-cv) — that's an explicit
        operator choice, not something the automated gate should block.
    """
    if new_cv_rmse is None or math.isnan(new_cv_rmse):
        return True
    if (
        incumbent_cv_rmse is None
        or math.isnan(incumbent_cv_rmse)
        or incumbent_cv_rmse <= 0
    ):
        return True
    return new_cv_rmse <= incumbent_cv_rmse * max_ratio
