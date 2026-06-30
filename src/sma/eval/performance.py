"""Honest performance measurement: alpha vs a buy-and-hold benchmark, and how
much to trust it given the live sample size.

The bot is fully long, so absolute return is dominated by market beta — a +6%
return in a +6% market is ZERO skill. The only evidence of edge is EXCESS return
over buy-and-hold SPY. And with a tiny live sample, even a positive excess is
noise. These pure helpers make both explicit so early P&L isn't over-read.
"""

from __future__ import annotations


def alpha_vs_benchmark(
    equity_curve: list[float], benchmark_curve: list[float]
) -> dict[str, float | None]:
    """Strategy vs buy-and-hold benchmark over the same period.

    Returns strategy_return, benchmark_return, and excess_return (the alpha
    proxy). All None when either curve has < 2 points or a non-positive start.
    """
    none = {"strategy_return": None, "benchmark_return": None, "excess_return": None}
    if len(equity_curve) < 2 or len(benchmark_curve) < 2:
        return none
    if equity_curve[0] <= 0 or benchmark_curve[0] <= 0:
        return none
    strat = equity_curve[-1] / equity_curve[0] - 1.0
    bench = benchmark_curve[-1] / benchmark_curve[0] - 1.0
    return {
        "strategy_return": strat,
        "benchmark_return": bench,
        "excess_return": strat - bench,
    }


def trust_level(n_live_days: int) -> tuple[str, str]:
    """How much the live P&L can be trusted, by sample size. Returns
    (level, explanation). The thresholds are deliberately conservative: a few
    dozen daily returns cannot distinguish skill from luck."""
    if n_live_days < 60:
        return (
            "LOW",
            f"{n_live_days} live trading days — far too few to distinguish skill "
            "from luck. Treat all P&L as noise.",
        )
    if n_live_days < 250:
        return (
            "MEDIUM",
            f"{n_live_days} live days — suggestive but not conclusive; one good "
            "month can flatter the record.",
        )
    return (
        "HIGH",
        f"{n_live_days} live days — a meaningful track record (still judge on "
        "EXCESS return vs SPY, not absolute).",
    )
