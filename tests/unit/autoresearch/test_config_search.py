"""Unit tests for the autoresearch config search core.

Synthetic data only (no DB): a learnable cross-sectional signal where y rank
tracks feature f0 within each date, so a working ranker earns positive CV-IC.
"""
import json
from datetime import date

import numpy as np
import pandas as pd

from sma.autoresearch.config_search import (
    SEARCH_SPACE,
    ConfigResult,
    SearchOutcome,
    sample_configs,
    search_configs,
    select_and_gate,
)
from sma.model.trainer import DEFAULT_HYPERPARAMS


def _synthetic(n_dates=40, n_names=30, seed=0):
    """Mirror build_training_set's shape: (X, y, asof_dates) with signal in f0."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_dates).date
    rows, ys, asofs = [], [], []
    for d in dates:
        f0 = rng.normal(size=n_names)
        f1 = rng.normal(size=n_names)
        y = f0 * 1.0 + rng.normal(scale=0.5, size=n_names)  # signal in f0
        for i in range(n_names):
            rows.append({"f0": f0[i], "f1": f1[i]})
            ys.append(float(y[i]))
            asofs.append(d)
    return pd.DataFrame(rows), pd.Series(ys), pd.Series(asofs)


def test_sample_configs_is_reproducible_and_distinct():
    a = sample_configs(SEARCH_SPACE, n=8, seed=42)
    b = sample_configs(SEARCH_SPACE, n=8, seed=42)
    assert a == b  # reproducible
    assert len(a) == 8
    # config 0 is always the DEFAULT baseline so the search can't do worse than it
    for k in ("max_depth", "learning_rate"):
        assert a[0][k] == DEFAULT_HYPERPARAMS[k]
    # every config only uses allowed values
    for cfg in a:
        for k, v in cfg.items():
            assert v in SEARCH_SPACE[k]


def test_sample_configs_clamps_to_baseline_for_nonpositive_n():
    # n<=0 must still yield the baseline so search_configs never indexes [] (and
    # the scheduled job fails safe with a recorded hold, not a pre-sentinel crash)
    got = sample_configs(SEARCH_SPACE, n=0, seed=0)
    assert len(got) == 1
    for k in ("max_depth", "learning_rate"):
        assert got[0][k] == DEFAULT_HYPERPARAMS[k]


def test_search_returns_results_sorted_by_ic_best_first():
    X, y, asof = _synthetic()
    results = search_configs(X, y, asof, n_configs=4, seed=1, n_folds=3, purge_days=2)
    assert len(results) == 4
    assert all(isinstance(r, ConfigResult) for r in results)
    # sorted by cv_ic descending (finite only); rank matches position
    ics = [r.cv_ic for r in results if r.cv_ic == r.cv_ic]
    assert ics == sorted(ics, reverse=True)
    assert results[0].rank == 0
    # the learnable signal yields a positive best IC
    assert results[0].cv_ic > 0.0


def test_search_handles_all_nan_without_crashing():
    # too few names per date → every per-date IC skipped → NaN, must not raise
    X = pd.DataFrame({"f0": [1.0, 2.0], "f1": [3.0, 4.0]})
    y = pd.Series([0.1, 0.2])
    asof = pd.Series(pd.bdate_range("2024-01-01", periods=2).date)
    results = search_configs(X, y, asof, n_configs=2, seed=0, n_folds=2, purge_days=0)
    assert all(r.cv_ic != r.cv_ic for r in results)  # all NaN, no exception


# ---- select_and_gate: search + incumbent lookup + promotion decision ----


def _write_incumbent(models_dir, train_end, cv_ic, label_type="demean",
                     train_start="2018-01-01", sha="dead1234", hyperparams=None):
    base = f"xgb_ret_30d_forward_{train_end}_{sha}"
    (models_dir / f"{base}.pkl").write_bytes(b"x")
    meta = {
        "cv_ic": cv_ic, "label_type": label_type,
        "train_start": train_start, "train_end_date": train_end,
    }
    if hyperparams is not None:
        meta["hyperparams"] = hyperparams
    (models_dir / f"{base}.json").write_text(json.dumps(meta))


def _gate(models_dir):
    X, y, asof = _synthetic()
    return select_and_gate(
        X, y, asof, models_dir=models_dir, asof_date=date(2026, 6, 10),
        train_start=date(2018, 1, 1), label_type="demean",
        n_configs=3, seed=1, n_folds=3, purge_days=2,
    )


def test_select_and_gate_promotes_when_no_incumbent(tmp_path):
    out = _gate(tmp_path)
    assert isinstance(out, SearchOutcome)
    assert out.best.cv_ic > 0  # learnable signal
    assert out.decision.promote is True  # nothing to beat, passes floor


def test_select_and_gate_defers_when_incumbent_ic_higher(tmp_path):
    _write_incumbent(tmp_path, "2026-06-08", cv_ic=0.99)
    out = _gate(tmp_path)
    assert out.decision.promote is False


def test_select_and_gate_defers_on_label_mismatch(tmp_path):
    _write_incumbent(tmp_path, "2026-06-08", cv_ic=0.0, label_type="raw")
    out = _gate(tmp_path)
    assert out.decision.promote is False
    assert "differ" in out.decision.reason.lower()


def test_select_and_gate_remeasures_incumbent_on_current_data(tmp_path):
    # The incumbent has real hyperparams but a STALE/overstated stored cv_ic.
    # The gate must compare against the incumbent RE-MEASURED on current data,
    # not the stale 0.99 (which would always force a HOLD).
    from sma.model.trainer import cv_information_coefficient
    X, y, asof = _synthetic()
    inc_hp = {"max_depth": 3, "n_estimators": 50}
    _write_incumbent(tmp_path, "2026-06-08", cv_ic=0.99, hyperparams=inc_hp)
    out = select_and_gate(
        X, y, asof, models_dir=tmp_path, asof_date=date(2026, 6, 10),
        train_start=date(2018, 1, 1), label_type="demean",
        n_configs=3, seed=1, n_folds=3, purge_days=2,
    )
    expected = cv_information_coefficient(X, y, asof, inc_hp, n_folds=3, purge_days=2)
    assert out.incumbent_cv_ic != 0.99                  # not the stale stored value
    assert abs(out.incumbent_cv_ic - expected) < 1e-9   # the value re-measured on current data


# ---- _incumbent_ic: guarded, no stale fallback on a bad re-measure (Codex) ----

_CS = "sma.autoresearch.config_search."


def _inc_ic_call():
    from pathlib import Path

    from sma.autoresearch.config_search import _incumbent_ic
    X, y, a = pd.DataFrame({"f": [0.0]}), pd.Series([0.0]), pd.Series([date(2024, 1, 1)])
    return _incumbent_ic(
        X, y, a, models_dir=Path("x"), asof_date=date(2026, 6, 22),
        target="ret_30d_forward", objective="reg", n_folds=3, purge_days=2,
    )


def test_incumbent_ic_legacy_no_hyperparams_uses_stored():
    from unittest.mock import patch
    with (
        patch(_CS + "incumbent_hyperparams", return_value=None),
        patch(_CS + "incumbent_cv_ic", return_value=0.0159),
    ):
        assert _inc_ic_call() == 0.0159


def test_incumbent_ic_remeasures_when_hyperparams_present():
    from unittest.mock import patch
    with (
        patch(_CS + "incumbent_hyperparams", return_value={"max_depth": 5}),
        patch(_CS + "cv_information_coefficient", return_value=0.012),
    ):
        assert _inc_ic_call() == 0.012


def test_incumbent_ic_eval_error_returns_none_not_crash():
    from unittest.mock import patch
    with (
        patch(_CS + "incumbent_hyperparams", return_value={"max_depth": 5}),
        patch(_CS + "cv_information_coefficient", side_effect=ValueError("boom")),
    ):
        assert _inc_ic_call() is None


def test_incumbent_ic_nan_returns_none_not_stale():
    from unittest.mock import patch
    with (
        patch(_CS + "incumbent_hyperparams", return_value={"max_depth": 5}),
        patch(_CS + "cv_information_coefficient", return_value=float("nan")),
        patch(_CS + "incumbent_cv_ic", return_value=0.99),
    ):
        assert _inc_ic_call() is None  # NOT 0.99 — no stale fallback on a bad re-measure
