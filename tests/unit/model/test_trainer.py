"""Tests for sma.model.trainer: single-shot trainer + walk-forward CV."""

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from sma.model.trainer import (
    _prepare_rank_training_data,
    _walk_forward_folds,
    select_hyperparams,
    train_xgb,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_FEATURES = 12
FEATURE_COLS = [f"f{i}" for i in range(N_FEATURES)]


def _make_xy(n_rows: int = 80, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.standard_normal((n_rows, N_FEATURES)), columns=FEATURE_COLS)
    y = pd.Series(rng.standard_normal(n_rows), name="ret_30d_forward")
    return X, y


def _make_asof_dates(n_rows: int = 100, n_tickers: int = 2) -> pd.Series:
    """Produce a Series of date objects: each date appears n_tickers times."""
    from datetime import date, timedelta

    base = date(2023, 1, 1)
    n_dates = n_rows // n_tickers
    dates = []
    for i in range(n_dates):
        d = base + timedelta(days=i)
        dates.extend([d] * n_tickers)
    return pd.Series(dates, name="asof_date")


# ---------------------------------------------------------------------------
# Task 10: train_xgb
# ---------------------------------------------------------------------------


def test_train_xgb_returns_fitted_model():
    X, y = _make_xy(80)
    model = train_xgb(X, y)
    assert isinstance(model, xgb.XGBRegressor)
    assert model.get_xgb_params()["objective"] == "reg:squarederror"
    preds = model.predict(X)
    assert preds.shape == (len(X),)


def test_train_xgb_deterministic_with_seed():
    X, y = _make_xy(80, seed=7)
    m1 = train_xgb(X, y, hyperparams={"random_state": 42})
    m2 = train_xgb(X, y, hyperparams={"random_state": 42})
    np.testing.assert_array_equal(m1.predict(X), m2.predict(X))


def test_train_xgb_raises_on_empty():
    X_empty = pd.DataFrame(columns=FEATURE_COLS)
    y_empty = pd.Series(dtype=float)
    with pytest.raises(ValueError, match="empty"):
        train_xgb(X_empty, y_empty)


def test_train_xgb_raises_on_mismatched_lengths():
    X, y = _make_xy(80)
    with pytest.raises(ValueError):
        train_xgb(X, y.iloc[:40])


def test_train_xgb_hyperparams_override():
    """Passing overrides should not raise and should produce a valid model."""
    X, y = _make_xy(50)
    model = train_xgb(X, y, hyperparams={"max_depth": 3, "n_estimators": 50})
    assert model.predict(X).shape == (50,)


def test_prepare_rank_training_data_sorts_by_asof_and_builds_groups_stably():
    X = pd.DataFrame({"f0": [10, 11, 12, 13, 14], "f1": [20, 21, 22, 23, 24]})
    y = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5], name="ret_30d_forward")
    asof_dates = pd.Series(
        pd.to_datetime([
            "2025-01-03",
            "2025-01-01",
            "2025-01-03",
            "2025-01-02",
            "2025-01-01",
        ]).date,
        name="asof_date",
    )

    X_sorted, y_sorted, groups = _prepare_rank_training_data(X, y, asof_dates)

    assert X_sorted["f0"].tolist() == [11, 14, 13, 10, 12]
    assert y_sorted.tolist() == [0.2, 0.5, 0.4, 0.1, 0.3]
    assert groups == [2, 1, 2]


def test_train_xgb_rank_returns_fitted_ranker():
    X, y = _make_xy(30)
    asof_dates = pd.Series(
        [date for date in pd.date_range("2025-01-01", periods=10).date for _ in range(3)]
    )

    model = train_xgb(
        X.sample(frac=1, random_state=3).reset_index(drop=True),
        y.sample(frac=1, random_state=3).reset_index(drop=True),
        hyperparams={"n_estimators": 5, "max_depth": 2},
        asof_dates=asof_dates.sample(frac=1, random_state=3).reset_index(drop=True),
        objective="rank",
    )

    assert isinstance(model, xgb.XGBRanker)
    assert model.get_xgb_params()["objective"] == "rank:pairwise"
    assert model.predict(X).shape == (len(X),)


def test_train_xgb_rank_requires_asof_dates():
    X, y = _make_xy(20)
    with pytest.raises(ValueError, match="asof_dates"):
        train_xgb(X, y, objective="rank")


# ---------------------------------------------------------------------------
# Task 11: _walk_forward_folds
# ---------------------------------------------------------------------------


def test_walk_forward_folds_produces_n_folds():
    asof_dates = _make_asof_dates(n_rows=100, n_tickers=2)  # 50 unique dates
    folds = _walk_forward_folds(asof_dates, n_folds=5, purge_days=5)
    # Should produce up to 5 folds; data is large enough
    assert len(folds) == 5


def test_walk_forward_folds_purge_gap_respected():
    """val_dates min should be > train_dates max + purge_days (strictly later)."""
    asof_dates = _make_asof_dates(n_rows=120, n_tickers=2)  # 60 unique dates
    purge = 3
    folds = _walk_forward_folds(asof_dates, n_folds=4, purge_days=purge)
    assert folds, "Expected at least one fold"
    sorted_unique = sorted(asof_dates.unique())

    for train_idx, val_idx in folds:
        train_dates = asof_dates.iloc[train_idx]
        val_dates = asof_dates.iloc[val_idx]
        # The gap in *sorted_dates positions* between train end and val start
        # must be at least purge_days.
        train_end_date = max(train_dates)
        val_start_date = min(val_dates)
        train_end_pos = sorted_unique.index(train_end_date)
        val_start_pos = sorted_unique.index(val_start_date)
        assert val_start_pos - train_end_pos >= purge


def test_walk_forward_folds_raises_too_few_dates():
    # 3 unique dates, n_folds=5 needs at least 6 unique dates
    dates = pd.Series([1, 1, 2, 2, 3, 3], name="asof_date")
    with pytest.raises(ValueError, match="unique dates"):
        _walk_forward_folds(dates, n_folds=5, purge_days=0)


def test_walk_forward_folds_no_overlap_train_val():
    """Train and val index sets must be disjoint."""
    asof_dates = _make_asof_dates(n_rows=100, n_tickers=2)
    folds = _walk_forward_folds(asof_dates, n_folds=4, purge_days=2)
    for train_idx, val_idx in folds:
        assert len(set(train_idx) & set(val_idx)) == 0


# ---------------------------------------------------------------------------
# Task 11: select_hyperparams
# ---------------------------------------------------------------------------

VALID_GRID = [
    {"max_depth": md, "learning_rate": lr}
    for md in [3, 5, 7]
    for lr in [0.05, 0.1]
]


def test_select_hyperparams_returns_grid_member():
    X, y = _make_xy(100, seed=1)
    # Use more unique asof_dates so CV has enough folds
    asof_dates = _make_asof_dates(n_rows=100, n_tickers=2)  # 50 unique dates

    best_params, best_rmse = select_hyperparams(
        X, y, asof_dates, n_folds=3, purge_days=3
    )

    assert best_params in VALID_GRID
    assert isinstance(best_rmse, float)
    assert best_rmse >= 0.0


def test_select_hyperparams_rmse_is_finite():
    X, y = _make_xy(80, seed=5)
    asof_dates = _make_asof_dates(n_rows=80, n_tickers=2)

    _, best_rmse = select_hyperparams(X, y, asof_dates, n_folds=2, purge_days=2)
    assert np.isfinite(best_rmse)


def test_cv_information_coefficient_high_for_predictive_features():
    """2026-06-13: CV gates on RMSE, which doesn't capture RANKING ability —
    the exact gap that let regime-luck configs look good. cv_information_
    coefficient measures held-out rank-IC (predictions vs realized, per asof).
    A feature that IS the label must yield strongly positive IC."""
    import numpy as np
    import pandas as pd

    from sma.model.trainer import cv_information_coefficient

    rng = np.random.default_rng(0)
    dates = pd.Series(np.repeat(pd.date_range("2024-01-01", periods=40).date, 25))
    n = len(dates)
    signal = rng.normal(size=n)
    X = pd.DataFrame({"signal": signal, "noise": rng.normal(size=n)})
    y = pd.Series(signal + rng.normal(scale=0.1, size=n))  # y ~ signal
    ic = cv_information_coefficient(
        X, y, dates, {"max_depth": 3, "learning_rate": 0.1}, n_folds=3, purge_days=2,
    )
    assert ic > 0.3, f"predictive features must give clearly positive CV-IC, got {ic}"


def test_cv_information_coefficient_near_zero_for_noise():
    import numpy as np
    import pandas as pd

    from sma.model.trainer import cv_information_coefficient

    rng = np.random.default_rng(1)
    dates = pd.Series(np.repeat(pd.date_range("2024-01-01", periods=40).date, 25))
    n = len(dates)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = pd.Series(rng.normal(size=n))  # independent of X
    ic = cv_information_coefficient(
        X, y, dates, {"max_depth": 3, "learning_rate": 0.1}, n_folds=3, purge_days=2,
    )
    assert abs(ic) < 0.15, f"noise must give ~0 CV-IC, got {ic}"
