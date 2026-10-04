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
from sma.autoresearch.promotion import PromotionDecision
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


def test_search_configs_stops_early_when_deadline_already_reached():
    # config 0 (the baseline) is always evaluated regardless — the search must
    # never do worse than the baseline it's guarding against — but nothing
    # past it runs once the deadline callback trips.
    X, y, asof = _synthetic()
    results = search_configs(
        X, y, asof, n_configs=4, seed=1, n_folds=3, purge_days=2,
        deadline_reached=lambda: True,
    )
    assert len(results) == 1
    assert results[0].params == {
        k: DEFAULT_HYPERPARAMS[k] for k in SEARCH_SPACE if k in DEFAULT_HYPERPARAMS
    }


def test_search_configs_stops_after_deadline_trips_mid_search():
    X, y, asof = _synthetic()
    calls = {"n": 0}

    def _deadline():
        calls["n"] += 1
        return calls["n"] >= 2  # trips before the 3rd config

    results = search_configs(
        X, y, asof, n_configs=4, seed=1, n_folds=3, purge_days=2,
        deadline_reached=_deadline,
    )
    assert len(results) == 2


def test_search_configs_runs_all_configs_when_deadline_never_reached():
    X, y, asof = _synthetic()
    results = search_configs(
        X, y, asof, n_configs=4, seed=1, n_folds=3, purge_days=2,
        deadline_reached=lambda: False,
    )
    assert len(results) == 4


def test_search_configs_default_has_no_deadline_guard():
    X, y, asof = _synthetic()
    results = search_configs(X, y, asof, n_configs=4, seed=1, n_folds=3, purge_days=2)
    assert len(results) == 4


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


def test_select_and_gate_threads_deadline_reached_into_search(tmp_path):
    X, y, asof = _synthetic()
    outcome = select_and_gate(
        X, y, asof, models_dir=tmp_path, asof_date=date(2026, 6, 10),
        train_start=date(2018, 1, 1), label_type="demean",
        n_configs=3, seed=1, n_folds=3, purge_days=2,
        deadline_reached=lambda: True,
    )
    # Only the baseline config ran once the deadline callback tripped.
    assert outcome.n_configs_evaluated == 1


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


def test_select_and_gate_skips_expensive_cv_on_label_mismatch(tmp_path):
    """2026-07-01 review finding: evaluate_search_promotion ALWAYS defers on a
    label_type mismatch, regardless of any candidate's IC (see
    test_select_and_gate_defers_on_label_mismatch above) -- so running the
    full n_configs x n_folds walk-forward CV search when the incumbent's
    label_type is already known to differ is pure wasted compute on a 16GB
    machine. Unless force=True (the deliberate --raw-labels CLI path),
    select_and_gate must skip search_configs() entirely and return
    immediately with zero evaluated candidates."""
    _write_incumbent(tmp_path, "2026-06-08", cv_ic=0.5, label_type="demean")
    X, y, asof = _synthetic()
    out = select_and_gate(
        X, y, asof, models_dir=tmp_path, asof_date=date(2026, 6, 10),
        train_start=date(2018, 1, 1), label_type="raw",
        n_configs=3, seed=1, n_folds=3, purge_days=2,
    )
    assert out.decision.promote is False
    assert "label" in out.decision.reason.lower()
    assert out.n_configs_evaluated == 0
    assert out.results == []


def test_select_and_gate_force_raw_still_runs_full_search(tmp_path):
    """Explicit --raw-labels (force=True) is a deliberate manual research run
    and must still fully evaluate every sampled config, even though the
    incumbent is demean and the run is known upfront to defer -- it must not
    be silently truncated the way an unflagged/scheduled run now is."""
    _write_incumbent(tmp_path, "2026-06-08", cv_ic=0.5, label_type="demean")
    X, y, asof = _synthetic()
    out = select_and_gate(
        X, y, asof, models_dir=tmp_path, asof_date=date(2026, 6, 10),
        train_start=date(2018, 1, 1), label_type="raw", force=True,
        n_configs=3, seed=1, n_folds=3, purge_days=2,
    )
    assert out.decision.promote is False
    assert out.n_configs_evaluated == 3
    assert len(out.results) == 3


def test_select_and_gate_matching_label_never_skips(tmp_path):
    """No mismatch (both demean) -> full search always runs regardless of
    force, matching current/existing behavior exactly."""
    _write_incumbent(tmp_path, "2026-06-08", cv_ic=0.5, label_type="demean")
    out = _gate(tmp_path)  # label_type="demean", force defaults False
    assert out.n_configs_evaluated == 3
    assert len(out.results) == 3


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


def test_incumbent_ic_threads_ensemble_seeds():
    """885bf98 gap: the incumbent re-measurement must be scored the same way
    the promoted artifact would be -- on ensemble-mean predictions -- or the
    gate compares a single-seed incumbent number to an ensemble candidate."""
    from unittest.mock import patch

    from sma.autoresearch.config_search import _incumbent_ic

    with (
        patch(_CS + "incumbent_hyperparams", return_value={"max_depth": 5}),
        patch(_CS + "cv_information_coefficient", return_value=0.012) as fn,
    ):
        X, y, a = pd.DataFrame({"f": [0.0]}), pd.Series([0.0]), pd.Series([date(2024, 1, 1)])
        _incumbent_ic(
            X, y, a, models_dir="x", asof_date=date(2026, 6, 22),
            target="ret_30d_forward", objective="reg", n_folds=3, purge_days=2,
            ensemble_seeds=7,
        )
    assert fn.call_args.kwargs["ensemble_seeds"] == 7


# ---- ensemble_seeds wiring (2026-08-24 gap-fix): the FINAL fit and the
# gate numbers that justify its promotion must honor model.ensemble_seeds so
# a promotion is apples-to-apples with the ensemble it deploys. The N-config
# search SWEEP itself stays single-seed (cost guard) -- only the chosen
# winner and the incumbent get re-measured at ensemble_seeds. ----


def test_search_configs_has_no_ensemble_seeds_knob():
    """Cost guard at the API level: search_configs cannot multiply its
    n_configs x n_folds sweep by ensemble_seeds because the parameter does
    not exist on it. select_and_gate must never grow one either."""
    import inspect

    assert "ensemble_seeds" not in inspect.signature(search_configs).parameters


def test_select_and_gate_search_sweep_stays_single_seed(tmp_path):
    """Cost guard: even when select_and_gate is asked for a 10-seed ensemble
    gate, the n_configs candidate sweep itself must NOT multiply by N -- only
    the winner's one re-measurement and the incumbent's one re-measurement
    may pay the ensemble tax."""
    from unittest.mock import patch

    from sma.model.trainer import cv_information_coefficient as real_cv_ic

    inc_hp = {"max_depth": 3, "n_estimators": 50}
    _write_incumbent(tmp_path, "2026-06-08", cv_ic=0.5, hyperparams=inc_hp)
    X, y, asof = _synthetic()
    calls = []

    def spy(*args, **kwargs):
        calls.append(kwargs.get("ensemble_seeds", 1))
        return real_cv_ic(*args, **kwargs)

    with patch(_CS + "cv_information_coefficient", side_effect=spy):
        select_and_gate(
            X, y, asof, models_dir=tmp_path, asof_date=date(2026, 6, 10),
            train_start=date(2018, 1, 1), label_type="demean",
            n_configs=3, seed=1, n_folds=3, purge_days=2, ensemble_seeds=10,
        )
    # First 3 calls are the search sweep (one per sampled config) -- single-seed.
    assert calls[:3] == [1, 1, 1]
    # Remaining calls are the winner + incumbent re-measurements -- ensemble.
    assert calls[3:], "expected the winner and/or incumbent to be re-measured"
    assert all(c == 10 for c in calls[3:])


def test_select_and_gate_ensemble_seeds_one_is_exact_current_behavior(tmp_path):
    """At ensemble_seeds=1 (the default), the gate must reuse the search
    winner's own cv_ic with NO extra CV-IC computation -- bit-identical
    result and cost to before this wiring existed."""
    from unittest.mock import patch

    from sma.model.trainer import cv_information_coefficient as real_cv_ic

    inc_hp = {"max_depth": 3, "n_estimators": 50}
    _write_incumbent(tmp_path, "2026-06-08", cv_ic=0.5, hyperparams=inc_hp)
    X, y, asof = _synthetic()
    calls = []

    def spy(*args, **kwargs):
        calls.append(kwargs.get("ensemble_seeds", 1))
        return real_cv_ic(*args, **kwargs)

    with patch(_CS + "cv_information_coefficient", side_effect=spy):
        out = select_and_gate(
            X, y, asof, models_dir=tmp_path, asof_date=date(2026, 6, 10),
            train_start=date(2018, 1, 1), label_type="demean",
            n_configs=3, seed=1, n_folds=3, purge_days=2,
        )  # ensemble_seeds omitted -> default 1
    # 3 search-sweep calls + 1 incumbent re-measure (pre-existing behavior);
    # NO extra winner re-measure call.
    assert len(calls) == 4
    assert out.promoted_cv_ic == out.best.cv_ic


def test_select_and_gate_ensemble_gate_ic_matches_direct_ensemble_measurement(tmp_path):
    """The gate's ensemble-consistent winner IC (promoted_cv_ic) must equal a
    direct cv_information_coefficient call on the winner's own config at the
    same ensemble_seeds -- the actual apples-to-apples number, independent of
    select_and_gate's internals."""
    from sma.model.trainer import cv_information_coefficient

    inc_hp = {"max_depth": 3, "n_estimators": 50}
    _write_incumbent(tmp_path, "2026-06-08", cv_ic=0.5, hyperparams=inc_hp)
    X, y, asof = _synthetic()
    out = select_and_gate(
        X, y, asof, models_dir=tmp_path, asof_date=date(2026, 6, 10),
        train_start=date(2018, 1, 1), label_type="demean",
        n_configs=3, seed=1, n_folds=3, purge_days=2, ensemble_seeds=5,
    )
    expected = cv_information_coefficient(
        X, y, asof, out.best.params, n_folds=3, purge_days=2, ensemble_seeds=5,
    )
    assert abs(out.promoted_cv_ic - expected) < 1e-9
    # a real, non-degenerate ensemble re-measurement differs from the
    # single-seed search-stage score it started from
    assert abs(out.promoted_cv_ic - out.best.cv_ic) > 1e-9


def test_select_and_gate_gates_on_ensemble_consistent_ic_not_search_stage_ic(tmp_path):
    """The promotion DECISION itself -- not just what gets recorded after --
    must compare the ensemble-consistent winner IC to the incumbent, not the
    single-seed search-stage score, once ensemble_seeds > 1."""
    from unittest.mock import patch

    inc_hp = {"max_depth": 3, "n_estimators": 50}
    _write_incumbent(tmp_path, "2026-06-08", cv_ic=0.5, hyperparams=inc_hp)
    X, y, asof = _synthetic()
    with patch(
        _CS + "evaluate_search_promotion",
        return_value=PromotionDecision(True, "stub"),
    ) as gate_fn:
        out = select_and_gate(
            X, y, asof, models_dir=tmp_path, asof_date=date(2026, 6, 10),
            train_start=date(2018, 1, 1), label_type="demean",
            n_configs=3, seed=1, n_folds=3, purge_days=2, ensemble_seeds=5,
        )
    assert gate_fn.call_args.kwargs["new_cv_ic"] == out.promoted_cv_ic
    assert abs(gate_fn.call_args.kwargs["new_cv_ic"] - out.best.cv_ic) > 1e-9
