"""Tests for the 15th model feature: days_to_next_earnings.

The model target is 30-day forward return. When a ticker reports earnings
inside that 30-day window, the target distribution is meaningfully
different (binary event risk dominates baseline drift). This feature
gives the model an explicit handle on that distinction.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from sma.features.builder import (
    _EARNINGS_PROXIMITY_CAP_DAYS,
    FEATURE_NAMES,
    _days_to_next_earnings,
    build_features,
)
from sma.model.loader import _compute_next_earnings


def _synthetic_prices(ticker: str, start: date, n_days: int):
    """Constant log-return + sin-wave volume so build_features doesn't drop it."""
    rng = np.random.default_rng(seed=hash(ticker) % 999)
    rows = []
    price = 100.0
    for i in range(n_days):
        d = start + timedelta(days=i)
        price *= np.exp(0.001)
        rows.append({
            "ticker": ticker, "date": d,
            "open": price, "high": price, "low": price, "close": price,
            "adj_close": price,
            "volume": int(1_000_000 + 100_000 * rng.standard_normal()),
            "source": "yfinance", "run_id": 1,
        })
    return pd.DataFrame(rows)


# ---- _days_to_next_earnings -------------------------------------------------


def test_feature_present_in_feature_names():
    assert "days_to_next_earnings" in FEATURE_NAMES
    # Position-stable so older models' feature_names_in_ still resolves;
    # new features always APPEND to the end.
    assert FEATURE_NAMES.index("days_to_next_earnings") == 14


def test_no_calendar_returns_cap():
    """Missing or empty calendar → every ticker gets the cap (≈ no earnings)."""
    assert _days_to_next_earnings("AAPL", date(2026, 5, 24), None) == 60.0
    assert _days_to_next_earnings("AAPL", date(2026, 5, 24), {}) == 60.0


def test_ticker_absent_from_calendar_returns_cap():
    cal = {"MSFT": date(2026, 6, 10)}
    assert _days_to_next_earnings("AAPL", date(2026, 5, 24), cal) == 60.0


def test_returns_actual_days_when_within_cap():
    cal = {"AAPL": date(2026, 5, 30)}
    assert _days_to_next_earnings("AAPL", date(2026, 5, 24), cal) == 6.0


def test_caps_when_earnings_more_than_60d_out():
    cal = {"AAPL": date(2026, 8, 1)}  # ~70 days out
    assert _days_to_next_earnings("AAPL", date(2026, 5, 24), cal) == 60.0


def test_past_earnings_returns_cap():
    """Earnings on or before asof shouldn't happen (loader filters
    report_date > asof) but guard anyway — a stale calendar must not
    produce a negative or zero feature value."""
    cal = {"AAPL": date(2026, 5, 20)}
    assert _days_to_next_earnings("AAPL", date(2026, 5, 24), cal) == 60.0
    # Exactly today: also treated as "no upcoming"
    cal = {"AAPL": date(2026, 5, 24)}
    assert _days_to_next_earnings("AAPL", date(2026, 5, 24), cal) == 60.0


def test_returns_float_not_int():
    """The training-set tests assert float dtype across all features;
    days_to_next_earnings has to be cast even when the day count is
    integer-valued."""
    val = _days_to_next_earnings("X", date(2026, 5, 24), {"X": date(2026, 6, 1)})
    assert isinstance(val, float)
    val = _days_to_next_earnings("X", date(2026, 5, 24), None)
    assert isinstance(val, float)


def test_cap_constant_matches_documented_60():
    """If someone tunes the cap, the docstring + this test should move
    together — guard against silent drift."""
    assert _EARNINGS_PROXIMITY_CAP_DAYS == 60


# ---- _compute_next_earnings (loader-side reducer) --------------------------


def test_compute_next_earnings_empty_df_returns_empty():
    out = _compute_next_earnings(pd.DataFrame(), date(2026, 5, 24))
    assert out == {}


def test_compute_next_earnings_picks_earliest_future_date():
    df = pd.DataFrame([
        {"ticker": "AAPL", "report_date": date(2026, 7, 30)},  # Q3
        {"ticker": "AAPL", "report_date": date(2026, 5, 30)},  # Q2 — closer
        {"ticker": "AAPL", "report_date": date(2026, 4, 30)},  # Q1 — past
        {"ticker": "MSFT", "report_date": date(2026, 6, 10)},
    ])
    out = _compute_next_earnings(df, date(2026, 5, 24))
    assert out == {
        "AAPL": date(2026, 5, 30),  # earliest > asof
        "MSFT": date(2026, 6, 10),
    }


def test_compute_next_earnings_excludes_past_only_tickers():
    """If every report_date for a ticker is on or before asof, it shouldn't
    appear in the calendar (builder will default to the cap)."""
    df = pd.DataFrame([
        {"ticker": "AAPL", "report_date": date(2026, 1, 30)},
        {"ticker": "AAPL", "report_date": date(2026, 4, 30)},  # still past 5/24
    ])
    out = _compute_next_earnings(df, date(2026, 5, 24))
    assert out == {}


# ---- end-to-end through build_features ------------------------------------


def test_build_features_populates_earnings_feature():
    """When earnings_calendar is provided, the feature reflects it. When
    omitted, every ticker gets the cap. Both are valid use cases."""
    asof = date(2026, 5, 24)
    start = asof - timedelta(days=400)
    tickers = ["AAPL", "MSFT", "SPY"]
    prices = pd.concat([_synthetic_prices(t, start, 401) for t in tickers], ignore_index=True)

    # Without earnings_calendar: feature defaults to cap for everyone.
    out_no_cal = build_features(prices, tickers, asof)
    for t in tickers:
        if t in out_no_cal.index:
            assert out_no_cal.loc[t, "days_to_next_earnings"] == 60.0

    # With earnings_calendar: AAPL gets actual days, MSFT gets cap.
    cal = {"AAPL": date(2026, 5, 30)}
    out_with_cal = build_features(prices, tickers, asof, earnings_calendar=cal)
    if "AAPL" in out_with_cal.index:
        assert out_with_cal.loc["AAPL", "days_to_next_earnings"] == 6.0
    if "MSFT" in out_with_cal.index:
        assert out_with_cal.loc["MSFT", "days_to_next_earnings"] == 60.0
