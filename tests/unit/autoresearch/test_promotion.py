"""Unit tests for the pure promotion gate.

Mirrors the retrain's deploy-gate sequence (src/sma/model/__main__.py): IC floor
(fail-closed) -> label-type transition -> train-window transition -> RMSE ratio.
"""
from sma.autoresearch.promotion import (
    IC_PROMOTE_MARGIN,
    evaluate_promotion,
    evaluate_search_promotion,
)


def test_ic_below_floor_fails_closed():
    d = evaluate_promotion(
        new_cv_ic=-0.05, new_cv_rmse=0.1, incumbent_cv_rmse=0.1,
        inc_label_type="raw", new_label_type="raw",
        inc_train_start="2018-01-01", new_train_start="2018-01-01", cv_ran=True,
    )
    assert d.promote is False
    assert "floor" in d.reason.lower()


def test_failed_cv_fails_closed_but_no_cv_passes():
    # cv_ran=True but IC is nan -> missing evidence -> fail closed
    d = evaluate_promotion(
        new_cv_ic=float("nan"), new_cv_rmse=0.1, incumbent_cv_rmse=0.1,
        inc_label_type="raw", new_label_type="raw",
        inc_train_start="2018-01-01", new_train_start="2018-01-01", cv_ran=True,
    )
    assert d.promote is False
    # operator chose --no-cv -> no evidence required -> pass
    d2 = evaluate_promotion(
        new_cv_ic=float("nan"), new_cv_rmse=float("nan"), incumbent_cv_rmse=0.1,
        inc_label_type="raw", new_label_type="raw",
        inc_train_start="2018-01-01", new_train_start="2018-01-01", cv_ran=False,
    )
    assert d2.promote is True


def test_label_transition_promotes_on_ic():
    d = evaluate_promotion(
        new_cv_ic=0.01, new_cv_rmse=0.5, incumbent_cv_rmse=0.1,
        inc_label_type="raw", new_label_type="demean",
        inc_train_start="2018-01-01", new_train_start="2018-01-01", cv_ran=True,
    )
    assert d.promote is True
    assert "label" in d.reason.lower()


def test_train_window_transition_promotes_on_ic():
    d = evaluate_promotion(
        new_cv_ic=0.01, new_cv_rmse=0.9, incumbent_cv_rmse=0.1,
        inc_label_type="raw", new_label_type="raw",
        inc_train_start="2023-01-01", new_train_start="2018-01-01", cv_ran=True,
    )
    assert d.promote is True
    assert "train" in d.reason.lower() or "window" in d.reason.lower()


def test_rmse_ratio_decides_when_no_transition():
    better = evaluate_promotion(
        new_cv_ic=0.03, new_cv_rmse=0.10, incumbent_cv_rmse=0.10,
        inc_label_type="raw", new_label_type="raw",
        inc_train_start="2018-01-01", new_train_start="2018-01-01", cv_ran=True,
    )
    worse = evaluate_promotion(
        new_cv_ic=0.03, new_cv_rmse=0.30, incumbent_cv_rmse=0.10,
        inc_label_type="raw", new_label_type="raw",
        inc_train_start="2018-01-01", new_train_start="2018-01-01", cv_ran=True,
    )
    assert better.promote is True
    assert worse.promote is False
    assert "rmse" in worse.reason.lower()


def test_no_incumbent_promotes():
    d = evaluate_promotion(
        new_cv_ic=0.02, new_cv_rmse=0.1, incumbent_cv_rmse=None,
        inc_label_type=None, new_label_type="demean",
        inc_train_start=None, new_train_start="2018-01-01", cv_ran=True,
    )
    assert d.promote is True


# ---- evaluate_search_promotion: the IC-improvement gate for config search ----


def _search(new_cv_ic, incumbent_cv_ic, **kw):
    """Defaults to a same-surface comparison (demean/2018 on both sides)."""
    base = dict(
        inc_label_type="demean", new_label_type="demean",
        inc_train_start="2018-01-01", new_train_start="2018-01-01", cv_ran=True,
    )
    base.update(kw)
    return evaluate_search_promotion(
        new_cv_ic=new_cv_ic, incumbent_cv_ic=incumbent_cv_ic, **base
    )


def test_search_floor_fails_closed():
    d = _search(new_cv_ic=-0.05, incumbent_cv_ic=0.03)
    assert d.promote is False
    assert "floor" in d.reason.lower()


def test_search_nan_ic_fails_closed_when_cv_ran():
    d = _search(new_cv_ic=float("nan"), incumbent_cv_ic=0.03, cv_ran=True)
    assert d.promote is False


def test_search_no_incumbent_model_promotes_on_floor():
    # genuinely no deployed model: inc_label_type is None (the existence signal)
    d = _search(new_cv_ic=0.01, incumbent_cv_ic=None,
                inc_label_type=None, inc_train_start=None)
    assert d.promote is True


def test_search_same_surface_unreadable_incumbent_ic_defers():
    # an incumbent EXISTS on the same surface but has no comparable CV-IC
    # (legacy/NaN) — can't prove an improvement, so defer rather than deploy
    d = _search(new_cv_ic=0.10, incumbent_cv_ic=None)  # _search defaults to demean/2018
    assert d.promote is False
    assert "comparable" in d.reason.lower()


def test_search_defers_on_mismatch_even_when_incumbent_ic_unreadable():
    # Codex 2026-06-17: a mismatched-surface incumbent with an unreadable IC
    # (incumbent_cv_ic=None) must still DEFER, not auto-promote on the None path.
    d = _search(new_cv_ic=0.10, incumbent_cv_ic=None, inc_label_type="raw")
    assert d.promote is False
    assert "differ" in d.reason.lower()


def test_search_requires_real_ic_lift_over_incumbent():
    up = _search(new_cv_ic=0.03 + IC_PROMOTE_MARGIN, incumbent_cv_ic=0.03)
    flat = _search(new_cv_ic=0.03 + IC_PROMOTE_MARGIN / 2, incumbent_cv_ic=0.03)
    assert up.promote is True          # meaningful lift ships
    assert flat.promote is False       # noise-sized lift does NOT churn live model


def test_search_defers_on_label_or_window_mismatch_even_with_higher_ic():
    # search compares apples-to-apples; if the incumbent's label/window differs,
    # IC isn't comparable, so defer rather than deploy (even at higher IC).
    lab = _search(new_cv_ic=0.10, incumbent_cv_ic=0.03, inc_label_type="raw")
    win = _search(new_cv_ic=0.10, incumbent_cv_ic=0.03, inc_train_start="2023-01-01")
    assert lab.promote is False
    assert win.promote is False
