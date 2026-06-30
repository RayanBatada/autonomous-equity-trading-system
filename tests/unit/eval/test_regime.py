"""Data-driven regime classifier: trend when momentum works, reversal when it
inverts (2026-06-25 — replaces the calendar-midpoint mislabel)."""


from sma.eval.regime import label_from_factor, momentum_factor_return

_PAST = {"A": 0.5, "B": 0.4, "C": 0.1, "D": -0.1, "E": -0.4, "F": -0.5}


def test_trend_when_winners_keep_winning():
    fwd = {"A": 0.20, "B": 0.15, "C": 0.0, "D": 0.0, "E": -0.10, "F": -0.15}
    fr = momentum_factor_return(_PAST, fwd, quantile=3)
    assert fr is not None and fr > 0
    assert label_from_factor(fr) == "trend"


def test_reversal_when_winners_crash_and_losers_bounce():
    fwd = {"A": -0.20, "B": -0.15, "C": 0.0, "D": 0.0, "E": 0.10, "F": 0.15}
    fr = momentum_factor_return(_PAST, fwd, quantile=3)
    assert fr is not None and fr < 0
    assert label_from_factor(fr) == "reversal"


def test_too_few_names_returns_none():
    assert momentum_factor_return({"A": 0.1, "B": 0.2}, {"A": 0.1, "B": 0.2}, quantile=3) is None


def test_nan_inputs_are_dropped_not_guessed():
    past = {**_PAST, "G": float("nan")}
    fwd = {"A": 0.2, "B": 0.15, "C": 0.0, "D": 0.0, "E": -0.1, "F": -0.15, "G": 0.9}
    fr = momentum_factor_return(past, fwd, quantile=3)
    assert fr is not None and fr > 0  # G (nan past) excluded, doesn't corrupt the factor


def test_label_edges():
    assert label_from_factor(None) is None
    assert label_from_factor(float("nan")) is None
    assert label_from_factor(0.0) == "trend"
    assert label_from_factor(-1e-9) == "reversal"
