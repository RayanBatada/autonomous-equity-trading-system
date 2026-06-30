"""XGBoost training: single-shot trainer + walk-forward CV hyperparameter search."""

import numpy as np
import pandas as pd
import xgboost as xgb

DEFAULT_HYPERPARAMS: dict = {
    "max_depth": 5,
    "n_estimators": 300,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1,
    "tree_method": "hist",
    "objective": "reg:squarederror",
}


def train_xgb(
    X: pd.DataFrame,  # noqa: N803
    y: pd.Series,
    hyperparams: dict | None = None,
    *,
    asof_dates: pd.Series | None = None,
    objective: str = "reg",
) -> xgb.XGBRegressor | xgb.XGBRanker:
    """Train one XGBoost model on (X, y).

    Args:
        X: feature DataFrame.
        y: regression target or ranking relevance Series.
        hyperparams: overrides for DEFAULT_HYPERPARAMS. None means use defaults.
        asof_dates: per-row asof dates. Required for objective="rank" so
            XGBoost can receive one query group per date.
        objective: "reg" for XGBRegressor, "rank" for XGBRanker.

    Returns the fitted model.
    """
    if X.empty or len(y) == 0:
        raise ValueError("Cannot train on empty data.")
    if len(X) != len(y):
        raise ValueError(f"X has {len(X)} rows but y has {len(y)} values.")
    if objective not in {"reg", "rank"}:
        raise ValueError("objective must be 'reg' or 'rank'.")

    params = {**DEFAULT_HYPERPARAMS, **(hyperparams or {})}
    if objective == "rank":
        X_fit, y_fit, groups = _prepare_rank_training_data(X, y, asof_dates)  # noqa: N806
        params["objective"] = "rank:pairwise"
        model = xgb.XGBRanker(**params)
        model.fit(X_fit, y_fit, group=groups)
        return model

    model = xgb.XGBRegressor(**params)
    model.fit(X, y)
    return model


def _prepare_rank_training_data(
    X: pd.DataFrame,  # noqa: N803
    y: pd.Series,
    asof_dates: pd.Series | None,
) -> tuple[pd.DataFrame, pd.Series, list[int]]:
    """Sort rows by asof date and return XGBoost ranker query group sizes."""
    if asof_dates is None:
        raise ValueError("asof_dates is required when objective='rank'.")
    if len(asof_dates) != len(X):
        raise ValueError(
            f"asof_dates has {len(asof_dates)} rows but X has {len(X)} rows."
        )

    sort_order = np.argsort(pd.Series(asof_dates).to_numpy(), kind="mergesort")
    sorted_dates = pd.Series(asof_dates).iloc[sort_order].reset_index(drop=True)
    groups = sorted_dates.value_counts(sort=False).tolist()
    X_sorted = X.iloc[sort_order].reset_index(drop=True)  # noqa: N806
    y_sorted = y.iloc[sort_order].reset_index(drop=True)
    return X_sorted, y_sorted, groups


def _walk_forward_folds(
    asof_dates: pd.Series,
    n_folds: int,
    purge_days: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate (train_idx, val_idx) pairs for expanding-window walk-forward CV.

    Sorts unique dates ascending, splits into n_folds + 1 blocks. Each fold
    trains on dates up to block boundary (with purge_days gap) and validates
    on the next block.

    purge_days creates a gap between train and val to prevent label leakage:
    since labels look forward 30 days, the last 30 days of train would
    overlap with the first val labels.

    NOTE: purge_days is counted in POSITIONS of the sorted unique asof-dates, not
    calendar days. This equals trading days (and so matches a 30-session forward
    label) ONLY when asof dates are one-per-trading-day (label_stride=1, the
    default). If label_stride > 1, pass purge_days = ceil(forward_horizon /
    label_stride) to keep the gap >= the label horizon.
    """
    sorted_dates = sorted(asof_dates.unique())
    if len(sorted_dates) < n_folds + 1:
        raise ValueError(
            f"Need at least {n_folds + 1} unique dates for {n_folds} folds; "
            f"got {len(sorted_dates)}."
        )

    block_size = len(sorted_dates) // (n_folds + 1)
    folds: list[tuple[np.ndarray, np.ndarray]] = []

    for fold_i in range(n_folds):
        train_end_pos = block_size * (fold_i + 1)
        val_start_pos = train_end_pos + purge_days
        val_end_pos = val_start_pos + block_size

        if val_start_pos >= len(sorted_dates):
            break  # not enough data for this fold

        train_dates = set(sorted_dates[:train_end_pos])
        val_dates = set(sorted_dates[val_start_pos:val_end_pos])
        if not val_dates:
            break

        train_idx = asof_dates[asof_dates.isin(train_dates)].index.to_numpy()
        val_idx = asof_dates[asof_dates.isin(val_dates)].index.to_numpy()
        folds.append((train_idx, val_idx))

    return folds


def cv_information_coefficient(
    X: pd.DataFrame,  # noqa: N803
    y: pd.Series,
    asof_dates: pd.Series,
    params: dict,
    *,
    n_folds: int = 5,
    purge_days: int = 30,
    objective: str = "reg",
) -> float:
    """Held-out walk-forward rank-IC: mean per-asof Spearman correlation of
    out-of-fold predictions vs realized targets.

    The deploy gate has always used CV-RMSE, which measures point-error, not
    RANKING ability — and this is a ranker (decide only sorts by score). A
    model can have great RMSE and rank no better than chance, or invert in a
    regime, and RMSE would not see it (2026-06-13 IC analysis). This is the
    metric that actually tracks edge; reported in model metadata + the retrain
    sentinel so every retrain self-describes its out-of-sample ranking quality.
    Returns nan when no fold yields a usable IC.
    """
    from scipy.stats import spearmanr

    folds = _walk_forward_folds(asof_dates, n_folds, purge_days)
    ics: list[float] = []
    for train_idx, val_idx in folds:
        if len(train_idx) == 0 or len(val_idx) == 0:
            continue
        model = train_xgb(
            X.iloc[train_idx], y.iloc[train_idx], hyperparams=params,
            asof_dates=asof_dates.iloc[train_idx], objective=objective,
        )
        preds = model.predict(X.iloc[val_idx])
        va = pd.DataFrame({
            "a": asof_dates.iloc[val_idx].to_numpy(),
            "p": preds, "y": y.iloc[val_idx].to_numpy(),
        })
        for _, g in va.groupby("a"):
            if len(g) < 5 or g["p"].std() < 1e-12:
                continue
            ic = spearmanr(g["p"], g["y"]).correlation
            if ic == ic:  # not nan
                ics.append(float(ic))
    return float(np.mean(ics)) if ics else float("nan")


def walk_forward_cv_rmse(
    X: pd.DataFrame,  # noqa: N803
    y: pd.Series,
    asof_dates: pd.Series,
    params: dict,
    *,
    n_folds: int = 5,
    purge_days: int = 30,
    objective: str = "reg",
) -> float:
    """Held-out walk-forward CV RMSE for ONE param set — the point-error sibling
    of cv_information_coefficient.

    The autoresearch config search selects by IC, so it knows the winner's CV-IC
    but not its CV-RMSE. Persisting a real cv_rmse (instead of NaN) keeps the
    weekly retrain's incumbent-RMSE gate working: a NaN incumbent RMSE reads as
    None and makes should_promote pass unconditionally, silently disabling that
    safety gate for the next retrain (Codex review 2026-06-17). Returns nan when
    no fold yields a prediction.
    """
    folds = _walk_forward_folds(asof_dates, n_folds, purge_days)
    rmses: list[float] = []
    for train_idx, val_idx in folds:
        if len(train_idx) == 0 or len(val_idx) == 0:
            continue
        model = train_xgb(
            X.iloc[train_idx], y.iloc[train_idx], hyperparams=params,
            asof_dates=asof_dates.iloc[train_idx], objective=objective,
        )
        preds = model.predict(X.iloc[val_idx])
        rmse = float(np.sqrt(np.mean((preds - y.iloc[val_idx].to_numpy()) ** 2)))
        rmses.append(rmse)
    return float(np.mean(rmses)) if rmses else float("nan")


def select_hyperparams(
    X: pd.DataFrame,  # noqa: N803
    y: pd.Series,
    asof_dates: pd.Series,
    n_folds: int = 5,
    purge_days: int = 30,
    *,
    objective: str = "reg",
) -> tuple[dict, float]:
    """Run walk-forward CV across a small grid; return (best_params, best_rmse).

    Grid: max_depth in {3, 5, 7}, learning_rate in {0.05, 0.1}. n_estimators
    fixed at 300. 6 combinations total.
    """
    grid = []
    for max_depth in [3, 5, 7]:
        for lr in [0.05, 0.1]:
            grid.append({"max_depth": max_depth, "learning_rate": lr})

    folds = _walk_forward_folds(asof_dates, n_folds, purge_days)
    if not folds:
        raise ValueError("No folds produced; check n_folds and data size.")

    best_rmse = float("inf")
    best_params: dict = grid[0]

    for params in grid:
        rmses: list[float] = []
        for train_idx, val_idx in folds:
            X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]  # noqa: N806
            y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]
            if len(X_tr) == 0 or len(X_va) == 0:
                continue
            model = train_xgb(
                X_tr,
                y_tr,
                hyperparams=params,
                asof_dates=asof_dates.iloc[train_idx],
                objective=objective,
            )
            preds = model.predict(X_va)
            rmse = float(np.sqrt(np.mean((preds - y_va.to_numpy()) ** 2)))
            rmses.append(rmse)
        if not rmses:
            continue
        mean_rmse = float(np.mean(rmses))
        if mean_rmse < best_rmse:
            best_rmse = mean_rmse
            best_params = params

    return best_params, best_rmse


def select_hyperparams_keep_better_ic(
    X: pd.DataFrame,  # noqa: N803
    y: pd.Series,
    asof_dates: pd.Series,
    *,
    models_dir,
    asof_date,
    target: str = "ret_30d_forward",
    objective: str = "reg",
    n_folds: int = 5,
    purge_days: int = 30,
) -> tuple[dict, float, float]:
    """Pick retrain hyperparameters: the RMSE-grid winner, UNLESS the currently
    deployed model's OWN config scores a higher walk-forward CV-IC on this data —
    then keep the incumbent's config. This keeps a good config (from a past
    retrain or an autoresearch win) across weeks instead of re-rolling it by RMSE
    every time (2026-06-23). Returns (hyperparams, cv_ic, cv_rmse).
    """
    from sma.model.persistence import incumbent_hyperparams

    best_params, grid_rmse = select_hyperparams(
        X, y, asof_dates, n_folds, purge_days, objective=objective,
    )
    grid_full = {**DEFAULT_HYPERPARAMS, **best_params}
    grid_ic = cv_information_coefficient(
        X, y, asof_dates, best_params,
        n_folds=n_folds, purge_days=purge_days, objective=objective,
    )
    chosen, cv_ic, cv_rmse = grid_full, grid_ic, grid_rmse

    inc_params = incumbent_hyperparams(models_dir, asof_date, target)
    if inc_params is not None and inc_params != grid_full:
        try:
            inc_ic = cv_information_coefficient(
                X, y, asof_dates, inc_params,
                n_folds=n_folds, purge_days=purge_days, objective=objective,
            )
        except Exception:  # noqa: BLE001 - a bad deployed config must not abort the retrain
            inc_ic = float("nan")  # ignore it; keep the grid pick
        if not np.isnan(inc_ic) and (np.isnan(grid_ic) or inc_ic > grid_ic):
            chosen = inc_params
            cv_ic = inc_ic
            cv_rmse = walk_forward_cv_rmse(
                X, y, asof_dates, inc_params,
                n_folds=n_folds, purge_days=purge_days, objective=objective,
            )
    return chosen, cv_ic, cv_rmse
