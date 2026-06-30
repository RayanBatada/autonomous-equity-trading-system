"""alpha_vs_benchmark + trust_level: honest performance measurement."""

from sma.eval.performance import alpha_vs_benchmark, trust_level


def test_excess_return_is_strategy_minus_benchmark():
    # strategy +10%, SPY +6% -> +4% alpha
    out = alpha_vs_benchmark([100.0, 110.0], [100.0, 106.0])
    assert round(out["strategy_return"], 4) == 0.10
    assert round(out["benchmark_return"], 4) == 0.06
    assert round(out["excess_return"], 4) == 0.04


def test_beating_absolute_but_lagging_market_is_negative_alpha():
    # +6% looks good in isolation but SPY did +10% -> NEGATIVE alpha (no edge)
    out = alpha_vs_benchmark([100.0, 106.0], [100.0, 110.0])
    assert out["excess_return"] < 0


def test_insufficient_or_degenerate_curves_return_none():
    assert alpha_vs_benchmark([100.0], [100.0, 110.0])["excess_return"] is None
    assert alpha_vs_benchmark([0.0, 110.0], [100.0, 106.0])["excess_return"] is None


def test_trust_level_scales_with_sample_size():
    assert trust_level(25)[0] == "LOW"
    assert trust_level(120)[0] == "MEDIUM"
    assert trust_level(300)[0] == "HIGH"
    # the LOW message must be blunt about noise
    assert "luck" in trust_level(25)[1].lower()
