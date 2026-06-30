"""select_hyperparams_keep_better_ic: the retrain keeps whichever config — its
RMSE-grid winner or the currently-deployed model's — ranks better on CV-IC, so a
good config (from a past retrain or an autoresearch win) persists across weeks
instead of being re-rolled by RMSE each time.
"""
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from sma.model.trainer import DEFAULT_HYPERPARAMS, select_hyperparams_keep_better_ic

_X = pd.DataFrame({"f0": [0.0]})
_Y = pd.Series([0.0])
_A = pd.Series([date(2024, 1, 1)])
_GRID = {"max_depth": 3}
_INC = {"max_depth": 7, "n_estimators": 500}


def _call():
    return select_hyperparams_keep_better_ic(
        _X, _Y, _A, models_dir=Path("x"), asof_date=date(2026, 6, 22),
        target="ret_30d_forward",
    )


def test_keeps_incumbent_when_its_ic_is_higher():
    def fake_ic(X, y, a, params, **kw):  # noqa: N803
        return 0.05 if params == _INC else 0.01
    with (
        patch("sma.model.trainer.select_hyperparams", return_value=(_GRID, 0.20)),
        patch("sma.model.trainer.cv_information_coefficient", side_effect=fake_ic),
        patch("sma.model.trainer.walk_forward_cv_rmse", return_value=0.25),
        patch("sma.model.persistence.incumbent_hyperparams", return_value=_INC),
    ):
        params, cv_ic, cv_rmse = _call()
    assert params == _INC          # kept the incumbent's better-IC config
    assert cv_ic == 0.05
    assert cv_rmse == 0.25         # its re-measured CV-RMSE, not the grid's


def test_keeps_grid_when_incumbent_ic_lower():
    def fake_ic(X, y, a, params, **kw):  # noqa: N803
        return 0.05 if params == _GRID else 0.01
    with (
        patch("sma.model.trainer.select_hyperparams", return_value=(_GRID, 0.20)),
        patch("sma.model.trainer.cv_information_coefficient", side_effect=fake_ic),
        patch("sma.model.persistence.incumbent_hyperparams", return_value=_INC),
    ):
        params, cv_ic, cv_rmse = _call()
    assert params == {**DEFAULT_HYPERPARAMS, **_GRID}
    assert cv_ic == 0.05
    assert cv_rmse == 0.20


def test_keeps_grid_when_incumbent_eval_raises():
    # Codex: a bad deployed config must not abort the weekly retrain
    def fake_ic(X, y, a, params, **kw):  # noqa: N803
        if params == _INC:
            raise ValueError("bad deployed config")
        return 0.03
    with (
        patch("sma.model.trainer.select_hyperparams", return_value=(_GRID, 0.20)),
        patch("sma.model.trainer.cv_information_coefficient", side_effect=fake_ic),
        patch("sma.model.persistence.incumbent_hyperparams", return_value=_INC),
    ):
        params, cv_ic, cv_rmse = _call()
    assert params == {**DEFAULT_HYPERPARAMS, **_GRID}  # fell back to grid, no crash
    assert cv_ic == 0.03
    assert cv_rmse == 0.20


def test_uses_grid_when_no_incumbent():
    with (
        patch("sma.model.trainer.select_hyperparams", return_value=(_GRID, 0.20)),
        patch("sma.model.trainer.cv_information_coefficient", return_value=0.03),
        patch("sma.model.persistence.incumbent_hyperparams", return_value=None),
    ):
        params, cv_ic, cv_rmse = _call()
    assert params == {**DEFAULT_HYPERPARAMS, **_GRID}
    assert cv_ic == 0.03
    assert cv_rmse == 0.20
