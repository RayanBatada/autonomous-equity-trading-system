"""Tests for the 10-seed prediction ensemble (sma.model.ensemble + trainer wiring).

TDD: written before the implementation.

The load-bearing property in every one of these is BACKWARD COMPATIBILITY.
`ensemble_seeds=1` must reproduce the pre-2026-08-24 behaviour exactly — same
object type, same predictions, same artifact shape — because the currently
promoted model was trained that way and must keep serving through the change.
"""

import pickle
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from sma.model.ensemble import (
    DEFAULT_BASE_SEED,
    DEFAULT_ENSEMBLE_SEEDS,
    EnsembleModel,
    ensemble_random_states,
    ensemble_seed_list,
    ensemble_size,
    seeds_are_inert,
)
from sma.model.persistence import load_metadata, load_model, save_model
from sma.model.trainer import (
    _walk_forward_folds,
    cv_information_coefficient,
    select_hyperparams,
    select_hyperparams_keep_better_ic,
    train_xgb,
    walk_forward_cv_rmse,
)

N_FEATURES = 6
FEATURE_COLS = [f"f{i}" for i in range(N_FEATURES)]
# Tiny trees keep the whole module fast; the properties under test are
# structural, not statistical.
TINY = {"n_estimators": 6, "max_depth": 2, "n_jobs": 1}


def _make_xy(n_rows: int = 120, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.standard_normal((n_rows, N_FEATURES)), columns=FEATURE_COLS)
    # A weak real signal so trees actually differ between seeds.
    y = pd.Series(X["f0"] * 0.3 + rng.standard_normal(n_rows), name="ret_30d_forward")
    return X, y


def _make_asof_dates(n_rows: int = 120, n_per_date: int = 6) -> pd.Series:
    from datetime import timedelta

    base = date(2023, 1, 1)
    dates = []
    for i in range(n_rows // n_per_date):
        dates.extend([base + timedelta(days=i)] * n_per_date)
    return pd.Series(dates, name="asof_date")


# ---------------------------------------------------------------------------
# Seed list — the study's exact seeds
# ---------------------------------------------------------------------------

def test_seed_list_reproduces_the_study_seeds():
    """The study ran SEEDS = tuple(range(42, 52)). Production must train on
    those same 10 seeds, with production's incumbent seed (42) first."""
    assert ensemble_seed_list(DEFAULT_BASE_SEED, DEFAULT_ENSEMBLE_SEEDS) == list(range(42, 52))


def test_seed_list_of_one_is_just_the_base_seed():
    assert ensemble_seed_list(42, 1) == [42]


def test_seed_list_rejects_empty_ensemble():
    with pytest.raises(ValueError):
        ensemble_seed_list(42, 0)


def test_default_ensemble_size_is_ten():
    assert DEFAULT_ENSEMBLE_SEEDS == 10


# ---------------------------------------------------------------------------
# ensemble_seeds=1 is the OLD behaviour, exactly
# ---------------------------------------------------------------------------

def test_one_seed_returns_a_bare_estimator_not_a_wrapper():
    """A 1-seed artifact must be shaped exactly like every artifact written
    before this change — a plain XGBRegressor, not an EnsembleModel."""
    X, y = _make_xy()
    model = train_xgb(X, y, hyperparams=TINY, ensemble_seeds=1)
    assert isinstance(model, xgb.XGBRegressor)
    assert not isinstance(model, EnsembleModel)


def test_one_seed_bit_matches_the_pre_change_default_call():
    """`train_xgb(X, y)` (no ensemble argument at all — how autoresearch and
    every existing caller invoke it) and an explicit ensemble_seeds=1 must
    produce bit-identical predictions."""
    X, y = _make_xy(seed=3)
    before = train_xgb(X, y, hyperparams=TINY)
    after = train_xgb(X, y, hyperparams=TINY, ensemble_seeds=1)
    np.testing.assert_array_equal(before.predict(X), after.predict(X))


def test_one_seed_bit_matches_for_the_rank_objective_too():
    X, y = _make_xy(seed=4)
    asof = _make_asof_dates()
    before = train_xgb(X, y, hyperparams=TINY, asof_dates=asof, objective="rank")
    after = train_xgb(
        X, y, hyperparams=TINY, asof_dates=asof, objective="rank", ensemble_seeds=1,
    )
    assert isinstance(after, xgb.XGBRanker)
    np.testing.assert_array_equal(before.predict(X), after.predict(X))


def test_default_ensemble_seeds_argument_is_one():
    """The FUNCTION default stays 1 so autoresearch and the research scripts
    keep their single-model behaviour; only the retrain CLI opts in to 10."""
    import inspect

    assert inspect.signature(train_xgb).parameters["ensemble_seeds"].default == 1


def test_ensemble_seeds_must_be_a_positive_int():
    X, y = _make_xy()
    for bad in (0, -1):
        with pytest.raises(ValueError):
            train_xgb(X, y, hyperparams=TINY, ensemble_seeds=bad)


# ---------------------------------------------------------------------------
# N > 1: members differ ONLY in random_state
# ---------------------------------------------------------------------------

def test_ensemble_fits_n_members_differing_only_in_seed():
    X, y = _make_xy(seed=5)
    ens = train_xgb(X, y, hyperparams=TINY, ensemble_seeds=3)

    assert isinstance(ens, EnsembleModel)
    assert len(ens) == 3
    assert ens.random_states == [42, 43, 44]

    # Every other hyperparameter is identical across members.
    params = [dict(m.get_params()) for m in ens.models]
    for p in params:
        p.pop("random_state")
    assert params[0] == params[1] == params[2]


def test_ensemble_members_are_actually_different_models():
    """Guards against a bug where the seed never reaches XGBoost and the
    'ensemble' is 10 copies of one model — which would look like it passed
    every other test here while delivering zero variance reduction."""
    X, y = _make_xy(seed=6)
    ens = train_xgb(X, y, hyperparams=TINY, ensemble_seeds=3)
    p0 = ens.models[0].predict(X)
    p1 = ens.models[1].predict(X)
    assert not np.array_equal(p0, p1)


def test_ensemble_predict_is_the_arithmetic_mean_of_members():
    """The study's PRIMARY combination rule (PREREGISTRATION §2.2): mean of
    the per-seed RAW predictions."""
    X, y = _make_xy(seed=7)
    ens = train_xgb(X, y, hyperparams=TINY, ensemble_seeds=4)

    expected = np.stack(
        [np.asarray(m.predict(X), dtype=np.float64) for m in ens.models]
    ).mean(axis=0)
    np.testing.assert_array_equal(ens.predict(X), expected)


def test_ensemble_predict_returns_one_value_per_row():
    X, y = _make_xy(seed=8)
    ens = train_xgb(X, y, hyperparams=TINY, ensemble_seeds=3)
    assert ens.predict(X).shape == (len(X),)


def test_ensemble_of_one_member_returns_that_member_exactly():
    """float32 -> float64 widening is exact, so wrapping a single model must
    not perturb its values."""
    X, y = _make_xy(seed=9)
    single = train_xgb(X, y, hyperparams=TINY, ensemble_seeds=1)
    np.testing.assert_array_equal(EnsembleModel([single]).predict(X), single.predict(X))


def test_ensemble_rejects_zero_members():
    with pytest.raises(ValueError):
        EnsembleModel([])


def test_ensemble_exposes_feature_names_in():
    """The predictor reads `feature_names_in_` off the model to pick the
    column set the model was trained on. An ensemble artifact must answer it."""
    X, y = _make_xy(seed=10)
    ens = train_xgb(X, y, hyperparams=TINY, ensemble_seeds=3)
    assert list(ens.feature_names_in_) == FEATURE_COLS


def test_ensemble_rank_objective_builds_rankers():
    X, y = _make_xy(seed=11)
    asof = _make_asof_dates()
    ens = train_xgb(
        X, y, hyperparams=TINY, asof_dates=asof, objective="rank", ensemble_seeds=3,
    )
    assert isinstance(ens, EnsembleModel)
    assert all(isinstance(m, xgb.XGBRanker) for m in ens.models)
    assert ens.random_states == [42, 43, 44]


def test_seeds_are_inert_without_subsampling():
    """random_state only bites through XGBoost's stochastic parts. At
    subsample=colsample=1.0 an N-seed ensemble is N identical models — 10x the
    cost, zero variance reduction, and silent. The retrain warns on this."""
    assert seeds_are_inert({"subsample": 1.0, "colsample_bytree": 1.0}) is True
    assert seeds_are_inert({}) is True
    assert seeds_are_inert({"subsample": 0.8, "colsample_bytree": 1.0}) is False
    assert seeds_are_inert({"colsample_bynode": 0.9}) is False


def test_production_defaults_keep_the_ensemble_effective():
    """The whole adoption rests on this: DEFAULT_HYPERPARAMS must stay
    stochastic, or the 10-seed ensemble quietly becomes 1 model."""
    from sma.model.trainer import DEFAULT_HYPERPARAMS

    assert seeds_are_inert(DEFAULT_HYPERPARAMS) is False


def test_ensemble_size_and_random_states_helpers():
    X, y = _make_xy(seed=12)
    single = train_xgb(X, y, hyperparams=TINY, ensemble_seeds=1)
    ens = train_xgb(X, y, hyperparams=TINY, ensemble_seeds=3)

    assert ensemble_size(single) == 1
    assert ensemble_size(ens) == 3
    assert ensemble_random_states(single) == [42]
    assert ensemble_random_states(ens) == [42, 43, 44]


# ---------------------------------------------------------------------------
# Artifact round-trip
# ---------------------------------------------------------------------------

def _save(model, tmp_path: Path, **kw) -> Path:
    pkl_path, _ = save_model(
        model,
        hyperparams=dict(TINY),
        feature_names=FEATURE_COLS,
        train_end_date=date(2026, 8, 24),
        train_rows=120,
        train_rmse=0.5,
        code_commit="deadbeef",
        training_duration_seconds=0.1,
        output_dir=tmp_path / "models",
        **kw,
    )
    return pkl_path


def test_three_member_artifact_round_trips(tmp_path):
    X, y = _make_xy(seed=13)
    ens = train_xgb(X, y, hyperparams=TINY, ensemble_seeds=3)

    pkl_path = _save(ens, tmp_path)
    loaded = load_model(pkl_path)

    assert isinstance(loaded, EnsembleModel)
    assert len(loaded) == 3
    assert loaded.random_states == [42, 43, 44]
    np.testing.assert_array_equal(loaded.predict(X), ens.predict(X))


def test_artifact_metadata_records_the_ensemble(tmp_path):
    X, y = _make_xy(seed=14)
    ens = train_xgb(X, y, hyperparams=TINY, ensemble_seeds=3)
    meta = load_metadata(_save(ens, tmp_path).with_suffix(".json"))

    assert meta["ensemble_seeds"] == 3
    assert meta["ensemble_random_states"] == [42, 43, 44]


def test_artifact_metadata_keeps_library_versions_provenance(tmp_path):
    """library_versions is what lets a deployed pickle be diagnosed after a
    library upgrade (persistence.py's own note). Persisting N boosters must
    not drop it."""
    X, y = _make_xy(seed=15)
    ens = train_xgb(X, y, hyperparams=TINY, ensemble_seeds=3)
    meta = load_metadata(_save(ens, tmp_path).with_suffix(".json"))

    libs = meta["library_versions"]
    assert libs["xgboost"] == xgb.__version__
    for key in ("scikit-learn", "numpy", "pandas", "python"):
        assert libs[key]


def test_single_model_artifact_records_one_seed(tmp_path):
    X, y = _make_xy(seed=16)
    single = train_xgb(X, y, hyperparams=TINY, ensemble_seeds=1)
    meta = load_metadata(_save(single, tmp_path).with_suffix(".json"))

    assert meta["ensemble_seeds"] == 1
    assert meta["ensemble_random_states"] == [42]


def test_artifact_is_deterministic_same_inputs_same_output(tmp_path):
    """Same artifact + same inputs => bit-identical output, across separate
    loads (i.e. nothing about the mean depends on process state or ordering)."""
    X, y = _make_xy(seed=17)
    ens = train_xgb(X, y, hyperparams=TINY, ensemble_seeds=5)
    pkl_path = _save(ens, tmp_path)

    first = load_model(pkl_path).predict(X)
    second = load_model(pkl_path).predict(X)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first, load_model(pkl_path).predict(X))


# ---------------------------------------------------------------------------
# Backward compatibility with artifacts written BEFORE this change
# ---------------------------------------------------------------------------

def _pre_change_artifact(tmp_path: Path, X: pd.DataFrame, y: pd.Series) -> Path:
    """Write an artifact the way the code did before ensembles existed: a bare
    estimator through plain pickle.dump, and a sidecar with no ensemble fields.

    Built by hand rather than through save_model on purpose — a fixture built
    by the NEW save_model could not catch a regression in save_model itself.
    """
    model = xgb.XGBRegressor(**TINY, random_state=42)
    model.fit(X, y)
    out = tmp_path / "legacy"
    out.mkdir(parents=True, exist_ok=True)
    pkl_path = out / "xgb_ret_30d_forward_2026-08-17_dbb6660f.pkl"
    with pkl_path.open("wb") as f:
        pickle.dump(model, f)
    pkl_path.with_suffix(".json").write_text(
        '{"model_id": "xgb_ret_30d_forward_2026-08-17_dbb6660f", '
        '"target": "ret_30d_forward", "objective": "reg", "label_type": "demean", '
        '"cv_rmse": 0.11510518192167933, "cv_ic": 0.0009967960520898048, '
        '"promoted": true, "created_at": "2026-08-17T10:24:19.296707Z"}'
    )
    return pkl_path


def test_pre_change_artifact_still_loads_and_scores(tmp_path):
    """The currently promoted model (xgb_..._2026-08-17_dbb6660f) is a single
    pickled XGBRegressor with no ensemble fields anywhere. It must keep
    serving, unchanged, after this code lands."""
    X, y = _make_xy(seed=18)
    pkl_path = _pre_change_artifact(tmp_path, X, y)

    loaded = load_model(pkl_path)
    assert isinstance(loaded, xgb.XGBRegressor)
    assert not isinstance(loaded, EnsembleModel)

    reference = xgb.XGBRegressor(**TINY, random_state=42)
    reference.fit(X, y)
    np.testing.assert_array_equal(loaded.predict(X), reference.predict(X))


def test_pre_change_artifact_metadata_reads_as_one_seed(tmp_path):
    """A sidecar with no `ensemble_seeds` key must not crash a reader, and
    must mean one booster."""
    X, y = _make_xy(seed=19)
    pkl_path = _pre_change_artifact(tmp_path, X, y)

    meta = load_metadata(pkl_path.with_suffix(".json"))
    assert "ensemble_seeds" not in meta
    assert ensemble_size(load_model(pkl_path)) == 1


def test_real_promoted_artifacts_still_load():
    """Belt-and-braces against the ACTUAL artifacts on this machine. Skipped in
    CI, where models_artifacts/ is gitignored and absent."""
    models_dir = Path("models_artifacts")
    pkls = sorted(models_dir.glob("xgb_ret_30d_forward_*.pkl")) if models_dir.exists() else []
    if not pkls:
        pytest.skip("no local model artifacts to check")
    for pkl in pkls:
        model = load_model(pkl)
        assert hasattr(model, "predict"), f"{pkl.name} did not load as a scorable model"


# ---------------------------------------------------------------------------
# The GATES compute on ensemble-mean predictions
# ---------------------------------------------------------------------------

def _reference_cv_rmse(X, y, asof, params, seeds, n_folds=2, purge_days=1) -> float:
    """Hand-rolled walk-forward CV RMSE on ensemble-mean out-of-fold
    predictions: fit one model per seed per fold, average the fold's
    predictions, then RMSE. This is the number the gate must see."""
    rmses = []
    for train_idx, val_idx in _walk_forward_folds(asof, n_folds, purge_days):
        preds = [
            np.asarray(
                train_xgb(
                    X.iloc[train_idx], y.iloc[train_idx],
                    hyperparams={**params, "random_state": s},
                    asof_dates=asof.iloc[train_idx],
                ).predict(X.iloc[val_idx]),
                dtype=np.float64,
            )
            for s in seeds
        ]
        mean_pred = np.stack(preds).mean(axis=0)
        rmses.append(float(np.sqrt(np.mean((mean_pred - y.iloc[val_idx].to_numpy()) ** 2))))
    return float(np.mean(rmses))


def test_cv_rmse_gate_is_computed_on_ensemble_mean_predictions():
    X, y = _make_xy(n_rows=180, seed=20)
    asof = _make_asof_dates(n_rows=180)

    got = walk_forward_cv_rmse(
        X, y, asof, TINY, n_folds=2, purge_days=1, ensemble_seeds=3,
    )
    expected = _reference_cv_rmse(X, y, asof, TINY, seeds=[42, 43, 44])
    assert got == pytest.approx(expected, rel=1e-12)


def test_cv_rmse_gate_at_one_seed_is_unchanged():
    X, y = _make_xy(n_rows=180, seed=21)
    asof = _make_asof_dates(n_rows=180)

    before = walk_forward_cv_rmse(X, y, asof, TINY, n_folds=2, purge_days=1)
    after = walk_forward_cv_rmse(
        X, y, asof, TINY, n_folds=2, purge_days=1, ensemble_seeds=1,
    )
    assert before == after


def test_cv_rmse_gate_actually_moves_with_the_ensemble():
    """If ensemble_seeds silently failed to reach the fold fits, the N=3 gate
    number would equal the N=1 one and every apples-to-apples claim would be
    false."""
    X, y = _make_xy(n_rows=180, seed=22)
    asof = _make_asof_dates(n_rows=180)

    one = walk_forward_cv_rmse(X, y, asof, TINY, n_folds=2, purge_days=1, ensemble_seeds=1)
    three = walk_forward_cv_rmse(X, y, asof, TINY, n_folds=2, purge_days=1, ensemble_seeds=3)
    assert one != three


def test_cv_ic_gate_is_computed_on_ensemble_mean_predictions():
    """The IC floor is the top-level deploy safety. It must score the ensemble
    that will actually trade, not seed 42 alone."""
    from scipy.stats import spearmanr

    X, y = _make_xy(n_rows=180, seed=23)
    asof = _make_asof_dates(n_rows=180)

    got = cv_information_coefficient(
        X, y, asof, TINY, n_folds=2, purge_days=1, ensemble_seeds=3,
    )

    ics = []
    for train_idx, val_idx in _walk_forward_folds(asof, 2, 1):
        preds = [
            np.asarray(
                train_xgb(
                    X.iloc[train_idx], y.iloc[train_idx],
                    hyperparams={**TINY, "random_state": s},
                    asof_dates=asof.iloc[train_idx],
                ).predict(X.iloc[val_idx]),
                dtype=np.float64,
            )
            for s in (42, 43, 44)
        ]
        va = pd.DataFrame({
            "a": asof.iloc[val_idx].to_numpy(),
            "p": np.stack(preds).mean(axis=0),
            "y": y.iloc[val_idx].to_numpy(),
        })
        for _, g in va.groupby("a"):
            if len(g) < 5 or g["p"].std() < 1e-12:
                continue
            ic = spearmanr(g["p"], g["y"]).correlation
            if ic == ic:
                ics.append(float(ic))
    assert got == pytest.approx(float(np.mean(ics)), rel=1e-12)


def test_cv_ic_gate_at_one_seed_is_unchanged():
    X, y = _make_xy(n_rows=180, seed=24)
    asof = _make_asof_dates(n_rows=180)

    before = cv_information_coefficient(X, y, asof, TINY, n_folds=2, purge_days=1)
    after = cv_information_coefficient(
        X, y, asof, TINY, n_folds=2, purge_days=1, ensemble_seeds=1,
    )
    assert before == after


def test_hyperparam_grid_search_scores_ensembles():
    """select_hyperparams' winning RMSE becomes the artifact's cv_rmse — the
    number should_promote() compares against the incumbent. So the grid has to
    score ensembles too, or the gate would compare a single-seed RMSE against
    an ensemble that trades."""
    X, y = _make_xy(n_rows=180, seed=25)
    asof = _make_asof_dates(n_rows=180)

    _p1, rmse1 = select_hyperparams(X, y, asof, 2, 1, ensemble_seeds=1)
    _p3, rmse3 = select_hyperparams(X, y, asof, 2, 1, ensemble_seeds=3)
    assert rmse1 != rmse3


def test_grid_search_at_one_seed_is_unchanged():
    X, y = _make_xy(n_rows=180, seed=26)
    asof = _make_asof_dates(n_rows=180)

    before = select_hyperparams(X, y, asof, 2, 1)
    after = select_hyperparams(X, y, asof, 2, 1, ensemble_seeds=1)
    assert before == after


def test_retrain_gate_numbers_are_ensemble_numbers(tmp_path):
    """End of the gate chain: the (cv_ic, cv_rmse) the retrain writes into the
    artifact and the sentinel must be the ensemble's, so the deploy gate
    compares ensemble-vs-incumbent apples-to-apples (study PART 8)."""
    X, y = _make_xy(n_rows=180, seed=27)
    asof = _make_asof_dates(n_rows=180)
    empty_models_dir = tmp_path / "models"  # no incumbent to keep

    _hp, cv_ic, cv_rmse = select_hyperparams_keep_better_ic(
        X, y, asof,
        models_dir=empty_models_dir, asof_date=date(2026, 8, 24),
        n_folds=2, purge_days=1, ensemble_seeds=3,
    )

    assert cv_rmse == pytest.approx(
        _reference_cv_rmse(X, y, asof, _hp, seeds=[42, 43, 44]), rel=1e-9,
    )
    assert cv_ic == pytest.approx(
        cv_information_coefficient(
            X, y, asof, _hp, n_folds=2, purge_days=1, ensemble_seeds=3,
        ),
        rel=1e-12,
    )


# ---------------------------------------------------------------------------
# Config knob + CLI resolution
# ---------------------------------------------------------------------------

def test_model_config_defaults_to_ten_seeds():
    from sma.config import ModelConfig

    assert ModelConfig().ensemble_seeds == DEFAULT_ENSEMBLE_SEEDS


def test_model_config_rejects_a_zero_ensemble():
    from pydantic import ValidationError

    from sma.config import ModelConfig

    with pytest.raises(ValidationError):
        ModelConfig(ensemble_seeds=0)


def test_repo_config_yaml_carries_the_knob():
    """The shipped config.yaml must actually set model.ensemble_seeds, or the
    scheduled retrain would silently fall back to a code default."""
    import yaml

    cfg = Path("config.yaml")
    if not cfg.exists():
        pytest.skip("config.yaml not in the working directory")
    raw = yaml.safe_load(cfg.read_text())
    assert raw["model"]["ensemble_seeds"] == 10


def test_load_model_config_reads_the_yaml_without_needing_secrets(tmp_path, monkeypatch):
    """The retrain CLI has never loaded Settings (and so has never needed API
    keys in its environment). Reading this one knob must not change that."""
    import yaml

    from sma.config import load_model_config

    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump({"model": {"ensemble_seeds": 4}}))
    for var in ("FINNHUB_API_KEY", "NEWSAPI_KEY", "ALPACA_API_KEY",
                "ALPACA_API_SECRET", "ALPACA_BASE_URL", "EDGAR_USER_AGENT"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(tmp_path)

    assert load_model_config(cfg).ensemble_seeds == 4


def test_load_model_config_falls_back_to_default_when_absent(tmp_path):
    from sma.config import load_model_config

    assert load_model_config(tmp_path / "nope.yaml").ensemble_seeds == DEFAULT_ENSEMBLE_SEEDS


def test_load_model_config_survives_a_malformed_file(tmp_path):
    """A broken config.yaml must not take the weekly retrain down over an
    optional knob."""
    from sma.config import load_model_config

    bad = tmp_path / "config.yaml"
    bad.write_text("model: [this is not a mapping\n")
    assert load_model_config(bad).ensemble_seeds == DEFAULT_ENSEMBLE_SEEDS


def test_cli_flag_overrides_the_config(tmp_path):
    import yaml

    from sma.model.__main__ import _resolve_ensemble_seeds

    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump({"model": {"ensemble_seeds": 10}}))
    assert _resolve_ensemble_seeds(1, config_path=cfg) == 1
    assert _resolve_ensemble_seeds(None, config_path=cfg) == 10


def test_cli_rejects_a_nonsense_ensemble_size(tmp_path):
    from sma.model.__main__ import _resolve_ensemble_seeds

    with pytest.raises(ValueError):
        _resolve_ensemble_seeds(0, config_path=tmp_path / "config.yaml")
