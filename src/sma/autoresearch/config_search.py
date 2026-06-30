"""Autoresearch config search: find the training config with the best held-out
rank-IC.

Replaces the old tilt()-rewriting LLM loop (post-mortem in the vault at
1-Projects/Stock-Market-Predictor-Agents/autoresearch-diagnosis.md), which
optimized a post-selection reweighter against noisy short-window Sharpe and never
contributed. This searches the VALIDATED edge metric — walk-forward CV rank-IC
(trainer.cv_information_coefficient) — over a bounded, deterministic space, so
there is no API/LLM dependency to fail (the old loop's reliability sink).
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from datetime import date

import pandas as pd
from loguru import logger

from sma.autoresearch.promotion import PromotionDecision, evaluate_search_promotion
from sma.model.persistence import (
    incumbent_cv_ic,
    incumbent_hyperparams,
    incumbent_label_type,
    incumbent_train_start,
)
from sma.model.trainer import DEFAULT_HYPERPARAMS, cv_information_coefficient

# Bounded search space. Wider than select_hyperparams' 6-combo RMSE grid, and
# scored on IC instead of RMSE. Every value must be xgboost-valid; random search
# beats grid at a fixed evaluation budget (Bergstra & Bengio 2012), and lets us
# cap cost by choosing n_configs.
SEARCH_SPACE: dict[str, list] = {
    "max_depth": [3, 4, 5, 6],
    "learning_rate": [0.03, 0.05, 0.1],
    "n_estimators": [200, 300, 500],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0],
    "min_child_weight": [1, 3, 5],
}


@dataclass(frozen=True)
class ConfigResult:
    """One evaluated config. rank 0 = best (highest cv_ic); NaN-IC sort last."""

    params: dict
    cv_ic: float
    rank: int


def _baseline_config() -> dict:
    """The DEFAULT hyperparams projected onto the searched keys. Always config 0
    so the search result can never be worse than what the retrain would use."""
    return {k: DEFAULT_HYPERPARAMS[k] for k in SEARCH_SPACE if k in DEFAULT_HYPERPARAMS}


def sample_configs(space: dict[str, list], n: int, seed: int) -> list[dict]:
    """n distinct configs sampled from `space`, reproducible by seed.

    Config 0 is always the DEFAULT baseline; the rest are deduped random draws.
    May return fewer than n if the space is smaller than n (deterministic cap).
    ALWAYS returns at least the baseline (n<=0 clamps to 1) so search_configs
    never indexes an empty list — a bad --n-configs fails safe (evaluates the
    baseline + records a hold) instead of crashing before the sentinel.
    """
    rng = random.Random(seed)
    baseline = _baseline_config()
    configs: list[dict] = [baseline]
    seen = {tuple(sorted(baseline.items()))}
    max_attempts = n * 50
    attempts = 0
    while len(configs) < n and attempts < max_attempts:
        attempts += 1
        cfg = {k: rng.choice(v) for k, v in space.items()}
        key = tuple(sorted(cfg.items()))
        if key not in seen:
            seen.add(key)
            configs.append(cfg)
    return configs[: max(1, n)]


def search_configs(
    X: pd.DataFrame,  # noqa: N803
    y: pd.Series,
    asof_dates: pd.Series,
    *,
    n_configs: int,
    seed: int = 0,
    objective: str = "reg",
    n_folds: int = 5,
    purge_days: int = 30,
) -> list[ConfigResult]:
    """Evaluate `n_configs` sampled configs by walk-forward CV rank-IC.

    Returns ConfigResults sorted best-first (highest cv_ic; NaN last), with rank
    set to position. Pure: no DB, no I/O, deterministic given the inputs + seed.
    """
    configs = sample_configs(SEARCH_SPACE, n_configs, seed)
    scored: list[tuple[dict, float]] = []
    for cfg in configs:
        params = {**DEFAULT_HYPERPARAMS, **cfg}
        # A single config that can't be evaluated (degenerate folds, too few
        # dates, an xgboost error) must NOT crash the whole search — that
        # whole-run fragility is what killed the old loop. Record NaN and move
        # on; NaN configs sort last and are filtered before promotion.
        try:
            ic = cv_information_coefficient(
                X,
                y,
                asof_dates,
                params,
                n_folds=n_folds,
                purge_days=purge_days,
                objective=objective,
            )
        except Exception as e:  # noqa: BLE001 - intentional: isolate per-config failure
            logger.warning("config_search: config {} failed to evaluate: {}", cfg, e)
            ic = float("nan")
        scored.append((cfg, ic))
    # Finite IC descending; NaN sinks to the bottom (treated as -inf for sort).
    scored.sort(
        key=lambda t: (t[1] if not math.isnan(t[1]) else float("-inf")),
        reverse=True,
    )
    return [ConfigResult(params=c, cv_ic=ic, rank=i) for i, (c, ic) in enumerate(scored)]


@dataclass(frozen=True)
class SearchOutcome:
    """Result of one search: the winning config, the promotion decision, the
    incumbent's CV-IC it was compared against, how many configs ran, and EVERY
    evaluated config (best-first) so the run is fully reviewable — the
    counterfactual record of what each config would have scored, held or not."""

    best: ConfigResult
    decision: PromotionDecision
    incumbent_cv_ic: float | None
    n_configs_evaluated: int
    results: list[ConfigResult]


def _incumbent_ic(
    X: pd.DataFrame,  # noqa: N803
    y: pd.Series,
    asof_dates: pd.Series,
    *,
    models_dir,
    asof_date: date,
    target: str,
    objective: str,
    n_folds: int,
    purge_days: int,
) -> float | None:
    """The incumbent's CV-IC to gate against, re-measured on the CURRENT data for
    an apples-to-apples comparison. Returns:
      - the stored cv_ic when the incumbent has no recorded hyperparams (legacy);
      - None when the incumbent's own config can't be measured here (the eval
        raises, or returns NaN) — the gate then DEFERS rather than fall back to a
        stale, inconsistent stored value (Codex review 2026-06-23);
      - otherwise the freshly measured CV-IC.
    """
    inc_params = incumbent_hyperparams(models_dir, asof_date, target)
    if inc_params is None:
        return incumbent_cv_ic(models_dir, asof_date, target)
    try:
        measured = cv_information_coefficient(
            X, y, asof_dates, inc_params,
            n_folds=n_folds, purge_days=purge_days, objective=objective,
        )
    except Exception as e:  # noqa: BLE001 - a bad deployed config must not crash the run
        logger.warning("incumbent re-measure failed ({}); deferring", e)
        return None
    return None if math.isnan(measured) else measured


def select_and_gate(
    X: pd.DataFrame,  # noqa: N803
    y: pd.Series,
    asof_dates: pd.Series,
    *,
    models_dir,
    asof_date: date,
    train_start: date,
    label_type: str,
    target: str = "ret_30d_forward",
    objective: str = "reg",
    n_configs: int = 16,
    seed: int = 0,
    n_folds: int = 5,
    purge_days: int = 30,
) -> SearchOutcome:
    """Run the CV-IC search, take the best config, look up the deployed
    incumbent, and decide promotion via the IC-improvement gate. No side effects
    (no train/save), so it is testable with synthetic data + a tmp models_dir.
    The caller trains + saves the winner only when ``decision.promote`` is True.
    """
    results = search_configs(
        X, y, asof_dates,
        n_configs=n_configs, seed=seed, objective=objective,
        n_folds=n_folds, purge_days=purge_days,
    )
    best = results[0]
    # The incumbent's CV-IC, re-measured on the CURRENT data for a fair
    # comparison (guarded; defers rather than using a stale value on failure).
    inc_ic = _incumbent_ic(
        X, y, asof_dates, models_dir=models_dir, asof_date=asof_date,
        target=target, objective=objective, n_folds=n_folds, purge_days=purge_days,
    )
    decision = evaluate_search_promotion(
        new_cv_ic=best.cv_ic,
        incumbent_cv_ic=inc_ic,
        inc_label_type=incumbent_label_type(models_dir, asof_date, target),
        new_label_type=label_type,
        inc_train_start=incumbent_train_start(models_dir, asof_date, target),
        new_train_start=train_start.isoformat(),
    )
    logger.info(
        "config search: {} configs, best CV-IC {:+.4f} (incumbent {}); {} — {}",
        len(results), best.cv_ic,
        f"{inc_ic:+.4f}" if inc_ic is not None else "none",
        "PROMOTE" if decision.promote else "HOLD", decision.reason,
    )
    return SearchOutcome(
        best=best, decision=decision,
        incumbent_cv_ic=inc_ic, n_configs_evaluated=len(results),
        results=results,
    )
