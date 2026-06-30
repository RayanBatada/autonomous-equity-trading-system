"""Live drawdown computation for the max_drawdown rail (#1 live-path audit)."""

from sma.live.decide import _drawdown_from_peak


def test_drawdown_zero_at_new_high():
    assert _drawdown_from_peak(peak_equity=110_000, current_equity=110_000) == 0.0


def test_drawdown_zero_above_prior_peak():
    # current above the recorded peak (a fresh high) → no drawdown
    assert _drawdown_from_peak(peak_equity=100_000, current_equity=105_000) == 0.0


def test_drawdown_fraction_below_peak():
    # 110k peak, 99k now → 10% below peak
    assert _drawdown_from_peak(peak_equity=110_000, current_equity=99_000) == 0.1


def test_drawdown_zero_when_peak_nonpositive():
    # empty history / zero peak must not divide-by-zero or alarm
    assert _drawdown_from_peak(peak_equity=0.0, current_equity=0.0) == 0.0
