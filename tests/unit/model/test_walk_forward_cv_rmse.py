"""Tests for walk_forward_cv_rmse: the point-error sibling of
cv_information_coefficient, used to give an autoresearch search model a real
CV-RMSE (so it doesn't disable the next retrain's RMSE gate)."""
import numpy as np
import pandas as pd

from sma.model.trainer import walk_forward_cv_rmse


def _synthetic(n_dates=40, n_names=20, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_dates).date
    rows, ys, asofs = [], [], []
    for d in dates:
        f0 = rng.normal(size=n_names)
        for v in f0:
            rows.append({"f0": float(v), "f1": float(rng.normal())})
            ys.append(float(v + rng.normal(scale=0.5)))
            asofs.append(d)
    return pd.DataFrame(rows), pd.Series(ys), pd.Series(asofs)


def test_returns_finite_nonnegative_rmse_on_learnable_data():
    X, y, asof = _synthetic()
    rmse = walk_forward_cv_rmse(
        X, y, asof, {"max_depth": 3, "n_estimators": 50},
        n_folds=3, purge_days=2,
    )
    assert rmse == rmse  # not NaN
    assert rmse >= 0.0


def test_returns_nan_when_no_folds_possible():
    # purge_days larger than the data window → no folds → NaN, no exception
    X, y, asof = _synthetic(n_dates=8)
    rmse = walk_forward_cv_rmse(
        X, y, asof, {"max_depth": 3, "n_estimators": 30},
        n_folds=3, purge_days=99,
    )
    assert rmse != rmse  # NaN


def test_reproducible_given_same_inputs():
    X, y, asof = _synthetic()
    params = {"max_depth": 3, "n_estimators": 50}
    a = walk_forward_cv_rmse(X, y, asof, params, n_folds=3, purge_days=2)
    b = walk_forward_cv_rmse(X, y, asof, params, n_folds=3, purge_days=2)
    assert a == b
