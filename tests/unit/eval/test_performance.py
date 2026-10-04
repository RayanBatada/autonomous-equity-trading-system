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


# --- regression alpha (diagnosis 2026-06-25 quick win: the beta=1 excess
# overstates selection ~2.5pp on a 1.17-beta book; report TRUE alpha + t-stat
# and require |t|>2 before claiming skill) ----------------------------------


def test_regression_alpha_pure_beta_book_has_zero_alpha():
    """A strategy that is exactly 1.2x the benchmark has beta 1.2 and ZERO
    alpha — the naive excess-return proxy would call this 'skill' in an up
    market."""
    from sma.eval.performance import regression_alpha

    bench = [0.01, -0.005, 0.008, -0.002, 0.012, 0.003, -0.007, 0.009] * 10
    # Deterministic zero-mean micro-noise so the fit is not degenerate.
    noise = [(((i * 37) % 11) - 5) * 1e-5 for i in range(len(bench))]
    strat = [1.2 * r + e for r, e in zip(bench, noise, strict=True)]
    out = regression_alpha(strat, bench)
    assert abs(out["beta"] - 1.2) < 0.01
    assert abs(out["alpha_daily"]) < 5e-5
    assert abs(out["t_stat"]) < 2.0  # no skill claim on a pure beta book


def test_regression_alpha_constant_edge_is_significant():
    """A constant +10bp/day over the benchmark is real alpha with a huge t."""
    from sma.eval.performance import regression_alpha

    bench = [0.01, -0.005, 0.008, -0.002, 0.012, 0.003, -0.007, 0.009] * 10
    noise = [(((i * 37) % 11) - 5) * 1e-5 for i in range(len(bench))]
    strat = [r + 0.001 + e for r, e in zip(bench, noise, strict=True)]
    out = regression_alpha(strat, bench)
    assert abs(out["beta"] - 1.0) < 0.01
    assert abs(out["alpha_daily"] - 0.001) < 5e-5
    assert out["t_stat"] > 2.0
    assert out["n"] == len(bench)


def test_regression_alpha_short_series_returns_none():
    from sma.eval.performance import regression_alpha

    out = regression_alpha([0.01] * 5, [0.01] * 5)
    assert out["alpha_daily"] is None and out["t_stat"] is None


def test_regression_alpha_rejects_non_finite_inputs():
    """A NaN equity/SPY value (corrupted DB row) must return all-None, not a
    confident-looking nan banner (review 2026-07-20)."""
    from sma.eval.performance import regression_alpha

    bench = [0.01, -0.005] * 15
    strat = [0.01, -0.005] * 15
    strat[7] = float("nan")
    assert regression_alpha(strat, bench)["alpha_daily"] is None
    bench2 = list(bench)
    bench2[3] = float("inf")
    assert regression_alpha(strat[:0] or [0.01] * 30, bench2)["alpha_daily"] is None
