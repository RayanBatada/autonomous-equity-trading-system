"""Tests for write_predictions, save_model, load_model, load_metadata,
and latest_model_for_date in sma.model.persistence."""

import json
from datetime import date
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from sma.ingest.store import Store
from sma.model.persistence import (
    incumbent_cv_rmse,
    latest_model_for_date,
    load_metadata,
    load_model,
    save_model,
    should_promote,
    write_predictions,
)


def _fresh_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "test.duckdb"
    store = Store(path=str(db_path))
    store.connect()
    store.close()
    return db_path


def test_write_predictions_inserts_rows(tmp_path):
    db_path = _fresh_db(tmp_path)

    n = write_predictions(
        db_path,
        date(2025, 7, 1),
        "ret_30d_forward",
        "test_model_v1",
        {"AAA": 0.05, "BBB": -0.02},
    )

    assert n == 2

    con = duckdb.connect(str(db_path), read_only=True)
    rows = con.execute(
        "SELECT asof_date, ticker, target, predicted_value, model_id "
        "FROM predictions ORDER BY ticker"
    ).fetchall()
    con.close()

    assert len(rows) == 2
    assert rows[0] == (date(2025, 7, 1), "AAA", "ret_30d_forward", 0.05, "test_model_v1")
    assert rows[1] == (date(2025, 7, 1), "BBB", "ret_30d_forward", -0.02, "test_model_v1")


def test_write_predictions_is_idempotent(tmp_path):
    db_path = _fresh_db(tmp_path)

    write_predictions(
        db_path,
        date(2025, 7, 1),
        "ret_30d_forward",
        "test_model_v1",
        {"AAA": 0.05, "BBB": -0.02},
    )
    # Second call with different values for AAA: should overwrite.
    write_predictions(
        db_path,
        date(2025, 7, 1),
        "ret_30d_forward",
        "test_model_v1",
        {"AAA": 0.99, "BBB": -0.02},
    )

    con = duckdb.connect(str(db_path), read_only=True)
    rows = con.execute(
        "SELECT ticker, predicted_value FROM predictions ORDER BY ticker"
    ).fetchall()
    con.close()

    # Only 2 rows, not 4 (no duplicates).
    assert len(rows) == 2
    aaa_val = {t: v for t, v in rows}["AAA"]
    assert aaa_val == pytest.approx(0.99)


def test_write_predictions_empty_returns_zero(tmp_path):
    db_path = _fresh_db(tmp_path)

    n = write_predictions(
        db_path,
        date(2025, 7, 1),
        "ret_30d_forward",
        "test_model_v1",
        {},
    )

    assert n == 0

    con = duckdb.connect(str(db_path), read_only=True)
    count = con.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    con.close()
    assert count == 0


# ---------------------------------------------------------------------------
# Helpers for save/load tests
# ---------------------------------------------------------------------------

_FEAT_COLS = [f"f{i}" for i in range(12)]


def _tiny_model(n: int = 40) -> tuple[xgb.XGBRegressor, pd.DataFrame]:
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.standard_normal((n, 12)), columns=_FEAT_COLS)
    y = pd.Series(rng.standard_normal(n))
    model = xgb.XGBRegressor(n_estimators=10, max_depth=2, random_state=0)
    model.fit(X, y)
    return model, X


def _save_kwargs(model, X, train_end: date, tmp_path: Path, sha: str = "abc123def456") -> dict:
    return dict(
        model=model,
        hyperparams={"max_depth": 2, "n_estimators": 10},
        feature_names=_FEAT_COLS,
        train_end_date=train_end,
        train_rows=len(X),
        train_rmse=0.123,
        code_commit=sha,
        training_duration_seconds=1.5,
        output_dir=tmp_path / "models",
    )


# ---------------------------------------------------------------------------
# Task 12: save_model / load_model / load_metadata
# ---------------------------------------------------------------------------


def test_save_and_load_round_trip_returns_same_predictions(tmp_path):
    model, X = _tiny_model()
    kwargs = _save_kwargs(model, X, date(2025, 6, 1), tmp_path)
    pkl_path, _ = save_model(**kwargs)

    loaded = load_model(pkl_path)
    np.testing.assert_array_equal(model.predict(X), loaded.predict(X))


def test_save_writes_metadata_sidecar(tmp_path):
    model, X = _tiny_model()
    kwargs = _save_kwargs(model, X, date(2025, 6, 1), tmp_path, sha="deadbeef1234")
    pkl_path, json_path = save_model(**kwargs)

    assert json_path.exists()
    meta = json.loads(json_path.read_text())

    required_keys = {
        "model_id", "target", "train_end_date", "train_rows", "train_rmse",
        "feature_names", "hyperparams", "code_commit",
        "training_duration_seconds", "created_at",
    }
    assert required_keys <= set(meta.keys())
    assert meta["train_end_date"] == "2025-06-01"
    assert meta["code_commit"] == "deadbeef1234"
    assert meta["train_rows"] == len(X)
    assert meta["feature_names"] == _FEAT_COLS


def test_load_metadata_reads_json(tmp_path):
    model, X = _tiny_model()
    kwargs = _save_kwargs(model, X, date(2025, 3, 15), tmp_path)
    _, json_path = save_model(**kwargs)
    meta = load_metadata(json_path)
    assert meta["train_end_date"] == "2025-03-15"


def test_load_model_raises_if_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "nonexistent.pkl")


def test_load_metadata_raises_if_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_metadata(tmp_path / "nonexistent.json")


# ---------------------------------------------------------------------------
# A3 (2026-06-01): OOS metric recording + deployment gate
# ---------------------------------------------------------------------------


def test_save_model_records_cv_rmse_and_in_sample_train_rmse(tmp_path):
    """The artifact must record BOTH the out-of-sample CV RMSE (how it
    generalizes) and the in-sample train RMSE (to see the overfit gap)."""
    model, X = _tiny_model()
    kwargs = _save_kwargs(model, X, date(2025, 6, 1), tmp_path)
    kwargs["train_rmse"] = 0.090   # in-sample
    kwargs["cv_rmse"] = 0.140       # out-of-sample
    _, json_path = save_model(**kwargs)
    meta = json.loads(json_path.read_text())
    assert meta["cv_rmse"] == 0.140
    assert meta["train_rmse"] == 0.090


def test_should_promote_blocks_materially_worse_model():
    assert should_promote(0.150, 0.130) is False   # >10% worse OOS → block
    assert should_promote(0.135, 0.130) is True    # within tolerance → ok
    assert should_promote(0.100, 0.130) is True    # better → ok


def test_should_promote_allows_when_no_incumbent():
    assert should_promote(0.20, None) is True       # first model ever → promote


def test_should_promote_allows_when_new_cv_unmeasured():
    import math
    # --no-cv leaves cv_rmse NaN; that's an explicit operator choice, don't block.
    assert should_promote(math.nan, 0.13) is True


def test_incumbent_cv_rmse_reads_newest_eligible(tmp_path):
    model, X = _tiny_model()
    md = tmp_path / "models"
    k1 = _save_kwargs(model, X, date(2025, 6, 1), tmp_path); k1["cv_rmse"] = 0.20
    save_model(**k1)
    k2 = _save_kwargs(model, X, date(2025, 6, 8), tmp_path); k2["cv_rmse"] = 0.12
    save_model(**k2)
    assert incumbent_cv_rmse(md, date(2025, 6, 30)) == 0.12   # newest eligible
    assert incumbent_cv_rmse(md, date(2025, 1, 1)) is None    # none eligible


def test_incumbent_cv_rmse_includes_same_date_model(tmp_path):
    """A model dated exactly asof is the one actually serving (latest_model_for_date
    uses <= asof). The incumbent lookup must see it — otherwise a same-day retrain
    re-run would gate against nothing (Codex-caught off-by-one)."""
    model, X = _tiny_model()
    md = tmp_path / "models"
    k = _save_kwargs(model, X, date(2025, 6, 8), tmp_path); k["cv_rmse"] = 0.11
    save_model(**k)
    assert incumbent_cv_rmse(md, date(2025, 6, 8)) == 0.11


def test_save_model_records_promoted_flag(tmp_path):
    """Rejected artifacts must be self-describing (promoted=False) so future
    discovery can filter on the flag, not just on directory position."""
    model, X = _tiny_model()
    k = _save_kwargs(model, X, date(2025, 6, 1), tmp_path); k["promoted"] = False
    _, json_path = save_model(**k)
    assert json.loads(json_path.read_text())["promoted"] is False
    # Default is promoted=True (the normal deploy path).
    k2 = _save_kwargs(model, X, date(2025, 6, 2), tmp_path)
    _, jp2 = save_model(**k2)
    assert json.loads(jp2.read_text())["promoted"] is True


def test_incumbent_cv_rmse_legacy_fallback_to_train_rmse(tmp_path):
    """Old artifacts stored the CV value under `train_rmse` and have no
    `cv_rmse` key — the incumbent lookup must still find a baseline."""
    model, X = _tiny_model()
    md = tmp_path / "models"
    k = _save_kwargs(model, X, date(2025, 6, 1), tmp_path)  # no cv_rmse passed
    k["train_rmse"] = 0.150
    save_model(**k)
    assert incumbent_cv_rmse(md, date(2025, 6, 30)) == 0.150


# ---------------------------------------------------------------------------
# Task 12: latest_model_for_date
# ---------------------------------------------------------------------------


def test_latest_model_for_date_picks_most_recent_eligible(tmp_path):
    model, X = _tiny_model()
    models_dir = tmp_path / "models"

    for d_str in ["2025-01-01", "2025-06-01", "2025-12-01"]:
        save_model(**_save_kwargs(model, X, date.fromisoformat(d_str), tmp_path))

    # asof 2025-09-01 -> eligible: 2025-01-01, 2025-06-01; pick 2025-06-01
    best = latest_model_for_date(models_dir, date(2025, 9, 1))
    assert "2025-06-01" in best.stem

    # asof 2024-12-31 -> no eligible model
    with pytest.raises(FileNotFoundError):
        latest_model_for_date(models_dir, date(2024, 12, 31))


def test_latest_model_for_date_target_with_underscores(tmp_path):
    """Filename parsing must handle targets like 'ret_30d_forward' (multi-underscore)."""
    model, X = _tiny_model()
    models_dir = tmp_path / "models"

    # save_model default target is "ret_30d_forward"
    save_model(**_save_kwargs(model, X, date(2025, 4, 10), tmp_path))

    best = latest_model_for_date(models_dir, date(2025, 12, 31), target="ret_30d_forward")
    assert "2025-04-10" in best.stem


def test_latest_model_for_date_raises_no_models_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        latest_model_for_date(tmp_path / "nonexistent_dir", date(2025, 1, 1))


def test_save_model_records_label_type(tmp_path):
    """2026-06-13: demean-label models train on a different target (cross-
    sectional alpha) — metadata must record label_type so the deploy gate
    never compares incompatible RMSE scales across a transition."""
    import xgboost as xgb

    from sma.model.persistence import load_metadata, save_model

    m = xgb.XGBRegressor(n_estimators=2, max_depth=2)
    import numpy as np
    m.fit(np.random.rand(20, 3), np.random.rand(20))
    _, jp = save_model(
        m, hyperparams={}, feature_names=["a", "b", "c"],
        train_end_date=date(2025, 6, 30), train_rows=20, train_rmse=0.1,
        code_commit="abc", training_duration_seconds=1.0, output_dir=tmp_path,
        cv_rmse=0.12, label_type="demean",
    )
    assert load_metadata(jp)["label_type"] == "demean"


def test_save_model_label_type_defaults_raw(tmp_path):
    import numpy as np
    import xgboost as xgb

    from sma.model.persistence import load_metadata, save_model
    m = xgb.XGBRegressor(n_estimators=2, max_depth=2)
    m.fit(np.random.rand(20, 3), np.random.rand(20))
    _, jp = save_model(
        m, hyperparams={}, feature_names=["a", "b", "c"],
        train_end_date=date(2025, 6, 30), train_rows=20, train_rmse=0.1,
        code_commit="abc", training_duration_seconds=1.0, output_dir=tmp_path,
        cv_rmse=0.12,
    )
    assert load_metadata(jp)["label_type"] == "raw"


def test_passes_ic_floor_rejects_worse_than_chance():
    """2026-06-15: completing the IC gate. A model that ranks meaningfully
    WORSE than chance out-of-sample (negative held-out CV-IC) must never
    deploy, regardless of RMSE — RMSE can't see ranking inversion, which was
    the whole regime-luck trap. Conservative floor: only block clearly-
    negative IC, not thin-positive (real edge here is ~+0.03)."""
    from sma.model.persistence import GATE_MIN_CV_IC, passes_ic_floor

    assert passes_ic_floor(0.03) is True       # real edge
    assert passes_ic_floor(0.0) is True         # break-even — don't block
    assert passes_ic_floor(-0.005) is True      # noise-band — don't block
    assert passes_ic_floor(GATE_MIN_CV_IC - 0.01) is False  # clearly worse than chance
    assert passes_ic_floor(-0.10) is False      # strongly inverted — reject


def test_passes_ic_floor_no_cv_passes_but_failed_cv_fails_closed():
    """Codex review 2026-06-15: a missing IC from --no-cv (operator choice)
    passes, but a missing IC when CV ACTUALLY RAN (degenerate folds) is missing
    EVIDENCE and must FAIL CLOSED — else the gate silently bypasses."""
    import math

    from sma.model.persistence import passes_ic_floor

    # operator chose no-cv: no IC required
    assert passes_ic_floor(None, cv_ran=False) is True
    assert passes_ic_floor(math.nan, cv_ran=False) is True
    # CV ran but produced no usable IC: fail closed
    assert passes_ic_floor(None, cv_ran=True) is False
    assert passes_ic_floor(math.nan, cv_ran=True) is False
    # default cv_ran=True is the safe assumption
    assert passes_ic_floor(math.nan) is False


def test_passes_ic_floor_noise_band_boundary():
    """The floor is a calibrated noise band, not 'any negative blocked'. A
    model just inside the band (statistically indistinguishable from chance)
    deploys; one clearly below is blocked (Codex review: state this honestly)."""
    from sma.model.persistence import GATE_MIN_CV_IC, passes_ic_floor

    assert passes_ic_floor(GATE_MIN_CV_IC + 0.001) is True   # inside band -> deploy
    assert passes_ic_floor(GATE_MIN_CV_IC) is True            # exactly at floor
    assert passes_ic_floor(GATE_MIN_CV_IC - 0.001) is False   # below band -> block


def test_save_model_records_train_start(tmp_path):
    """2026-06-16: train_start in metadata so the gate can detect a training-
    WINDOW change (RMSE isn't comparable across different training
    distributions — multi-regime 2018+ has higher RMSE than 2023-only yet
    better OOS IC)."""
    import numpy as np
    import xgboost as xgb

    from sma.model.persistence import incumbent_train_start, load_metadata, save_model

    m = xgb.XGBRegressor(n_estimators=2, max_depth=2)
    m.fit(np.random.rand(20, 3), np.random.rand(20))
    _, jp = save_model(
        m, hyperparams={}, feature_names=["a", "b", "c"],
        train_end_date=date(2025, 6, 30), train_rows=20, train_rmse=0.1,
        code_commit="abc", training_duration_seconds=1.0, output_dir=tmp_path,
        cv_rmse=0.12, train_start=date(2018, 1, 1),
    )
    assert load_metadata(jp)["train_start"] == "2018-01-01"
    # incumbent_train_start reads it back; legacy (absent) -> None
    assert incumbent_train_start(tmp_path, date(2025, 7, 1)) == "2018-01-01"
