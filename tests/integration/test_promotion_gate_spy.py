"""Promotion gate: SPY simulator vs. ground-truth 2025 return.

This test loads real SPY price data from data/sma.duckdb and verifies that
BuyAndHoldSPYStrategy(target_weight=1.0) reproduces the actual 2025 SPY
adj_close return within 1 percentage point.

Skips if the DB is missing or contains no SPY data for 2025.
"""

from pathlib import Path

import pytest

_DB_PATH = Path("data/sma.duckdb")


def _skip_if_no_data():
    if not _DB_PATH.exists():
        pytest.skip(reason="data/sma.duckdb not present; run `python -m sma.ingest run` first")


def test_promotion_gate_spy_simulator_matches_ground_truth():
    """Simulated SPY 2025 total_return must be within 1% of adj_close ground truth."""
    _skip_if_no_data()

    from sma.backtest.__main__ import _spy_ground_truth_return, _spy_simulated_return

    try:
        ground_truth, first_day, last_day = _spy_ground_truth_return(_DB_PATH)
    except ValueError as exc:
        pytest.skip(reason=f"No SPY 2025 data in DB: {exc}")

    simulated = _spy_simulated_return(_DB_PATH)
    diff = abs(simulated - ground_truth)

    print(f"\ndate_range:   {first_day} to {last_day}")
    print(f"ground_truth: {ground_truth:.4%}")
    print(f"simulated:    {simulated:.4%}")
    print(f"abs_diff:     {diff:.4%}")

    assert diff < 0.01, (
        f"Simulator return {simulated:.4%} differs from ground truth "
        f"{ground_truth:.4%} by {diff:.4%}, exceeding the 1% tolerance. "
        "Investigate: slippage drag, adj_close vs close, dividend handling."
    )
