"""Seed ensemble: N XGBoost models that differ ONLY in random_state.

Adopted 2026-08-24 from the ensemble-rank study (`~/.sma-pit/
ensemble-rank-study-2026-08-21`), arm **B_ens** (`ens_reg_mean`), verdict
ADOPT-CANDIDATE-VARIANCE. What that verdict does and does not say:

- It did NOT pass the mean-IC route. The paired per-date IC delta vs the
  single-seed control was +0.0009..+0.0014 with Newey-West t of 1.1..1.6 —
  under the pre-registered bar (delta > ~0.002, |t| > ~1.6..2.0) at every
  horizon. The reason to ship this is not "it predicts better".
- It passed the VARIANCE route: no worse on mean IC (V1), and the variance of
  mean IC across sub-ensembles fell to 0.12-0.17x the single-seed variance
  (V2, bar was <= 0.50), with top-15 selection quality no worse (V3, +0.0015).

So the deliverable is SEED-LOTTERY REMOVAL: one seed's model is a draw from a
distribution whose spread is large next to the edge itself, and the weekly
retrain re-rolls that die every Monday. Averaging 10 seeds cuts ~85% of that
variance for 10x the (trivial: ~2s) fit cost. The study's sibling arms, both
of which changed the OBJECTIVE to `rank:pairwise` (C_rank, D_both), were
REJECTED — this module changes the seed count and nothing else, so the model
stays `reg:squarederror` and its scores stay in return units.

Averaging semantics are the study's PRIMARY rule (PREREGISTRATION.md §2.2),
copied exactly: the **arithmetic mean of the per-seed RAW predictions**, per
(asof_date, ticker). Not a median, not a mean of within-date ranks — those
were pre-registered sensitivities, not the adopted rule. The mean is computed
in one place, `EnsembleModel.predict`, and that same code runs in the
walk-forward CV that feeds the deploy gates AND in the live predictor, so the
gate measures exactly what trades.

Seeds are `base, base+1, ..., base+N-1` with base = the hyperparams'
`random_state` (production: 42). At N=10 that is 42..51 — the study's own
`SEEDS = tuple(range(42, 52))`, with 42 (production's incumbent seed) first.
N=1 is not this class at all: `train_xgb` returns the bare estimator, so a
single-seed artifact is byte-identical in shape to every artifact written
before this module existed.

TRAP, and the reason `seeds_are_inert` exists: an XGBoost fit is only
seed-dependent if something in it is actually stochastic. With `subsample=1.0`
AND `colsample_bytree=1.0`, `random_state` changes NOTHING — ten seeds give
ten bit-identical models, and the "ensemble" is one model at 10x the cost with
exactly zero variance reduction, silently. Production is safe today only
because DEFAULT_HYPERPARAMS carries `subsample=0.8, colsample_bytree=0.8`
(and the study's FROZEN_HYPERPARAMS carried the same). Anyone who retunes
those to 1.0 must know they are also turning this feature off.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import xgboost as xgb

# The study's ensemble size. 10 seeds is what arm B_ens actually ran; the
# variance numbers quoted above are estimated from 5-seed sub-ensembles of it
# (a 10-seed ensemble has exactly one realisation — PREREGISTRATION §9.2).
DEFAULT_ENSEMBLE_SEEDS = 10

# The base seed production has always trained on, and the study's CONTROL_SEED.
DEFAULT_BASE_SEED = 42

FittedModel = xgb.XGBRegressor | xgb.XGBRanker


def ensemble_seed_list(base_random_state: int, n: int) -> list[int]:
    """The n consecutive seeds an n-member ensemble trains on.

    `ensemble_seed_list(42, 10) == [42, 43, ..., 51]`, matching the study's
    `SEEDS = tuple(range(42, 52))` exactly, and putting production's incumbent
    seed (42) first so a 1-member ensemble is the incumbent model.
    """
    if n < 1:
        raise ValueError(f"ensemble size must be >= 1; got {n}")
    return [int(base_random_state) + i for i in range(n)]


class EnsembleModel:
    """N fitted XGBoost models; `predict` returns the mean of their outputs.

    Deliberately NOT an sklearn estimator: it is only ever built from
    already-fitted members (by `train_xgb`), and the only interface the rest
    of the system asks of a model is `.predict(X)` plus the sklearn metadata
    attributes the predictor reads off it. Keeping it a plain picklable object
    means an artifact is still `pickle.dump(model, f)` and nothing about the
    artifact contract changes.
    """

    def __init__(self, models: list[FittedModel]) -> None:
        members = list(models)
        if not members:
            raise ValueError("EnsembleModel needs at least one fitted model.")
        self.models = members

    def __len__(self) -> int:
        return len(self.models)

    def __repr__(self) -> str:
        inner = type(self.models[0]).__name__
        return f"EnsembleModel({len(self.models)}x {inner}, seeds={self.random_states})"

    @property
    def random_states(self) -> list[Any]:
        """Each member's `random_state`, in fit order. Recorded in artifact
        metadata so a deployed ensemble self-describes which seeds built it."""
        return [m.get_params().get("random_state") for m in self.models]

    def predict(self, X, **kwargs) -> np.ndarray:  # noqa: N803
        """Arithmetic mean of the members' raw predictions (study §2.2).

        Members are summed in fit order over a stacked float64 array, so the
        result is a pure function of (artifact, X) — same artifact and same
        inputs give a bit-identical array on every call. Members' own
        `predict` returns float32; the float32 -> float64 widening is exact,
        so a 1-member ensemble returns exactly its member's values.
        """
        stacked = np.stack(
            [np.asarray(m.predict(X, **kwargs), dtype=np.float64) for m in self.models]
        )
        return stacked.mean(axis=0)

    def __getattr__(self, name: str) -> Any:
        """Delegate sklearn metadata (`feature_names_in_`, `n_features_in_`,
        `get_xgb_params`, ...) to the first member.

        Every member was fit on the same X with the same params, so they agree
        on all of it except `random_state`. This is what lets the predictor's
        `getattr(model, "feature_names_in_", FEATURE_NAMES)` keep working
        unchanged against an ensemble artifact.

        Reads `self.__dict__` rather than `self.models`: __getattr__ runs
        during UNPICKLING (for `__setstate__`, `__reduce_ex__`, ...) before
        __init__ has put anything on the instance, and a plain `self.models`
        access there recurses until the stack blows. Dunder lookups are
        refused outright for the same reason.
        """
        if name.startswith("__"):
            raise AttributeError(name)
        models = self.__dict__.get("models")
        if not models:
            raise AttributeError(name)
        return getattr(models[0], name)


def seeds_are_inert(params: dict) -> bool:
    """True when `random_state` cannot change the fit, so an ensemble of N
    seeds would be N identical models.

    XGBoost's seed only bites through its stochastic parts: row subsampling
    (`subsample`) and column subsampling (`colsample_bytree` / `_bylevel` /
    `_bynode`). If every one of those is >= 1.0, the fit is deterministic and
    the ensemble is pure cost. The retrain CLI calls this once on the chosen
    hyperparameters and warns loudly rather than quietly training ten copies.
    """
    knobs = ("subsample", "colsample_bytree", "colsample_bylevel", "colsample_bynode")
    for knob in knobs:
        value = params.get(knob)
        if value is not None and float(value) < 1.0:
            return False
    return True


def ensemble_size(model: FittedModel | EnsembleModel) -> int:
    """Number of boosters behind a model. 1 for a plain (pre-2026-08-24 or
    `ensemble_seeds=1`) estimator."""
    return len(model) if isinstance(model, EnsembleModel) else 1


def ensemble_random_states(model: FittedModel | EnsembleModel) -> list[Any]:
    """The `random_state` of every booster behind a model, in fit order."""
    members = model.models if isinstance(model, EnsembleModel) else [model]
    return [m.get_params().get("random_state") for m in members]
