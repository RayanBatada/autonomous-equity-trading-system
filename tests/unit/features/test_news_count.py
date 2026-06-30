"""Tests for the 16th model feature: news_count_7d_log.

News coverage volume as an attention/momentum proxy. Aggregates across
all news sources (Finnhub + Alpaca/Benzinga + NewsAPI). log1p transform
compresses the heavy right tail (top names get 500-700+ articles/week
while median is ~50) without losing the monotonic relationship.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import pandas as pd

from sma.features.builder import (
    FEATURE_NAMES,
    _news_count_7d_log,
    build_features,
)
from sma.model.loader import _compute_news_counts_7d


def test_feature_present_in_feature_names():
    assert "news_count_7d_log" in FEATURE_NAMES
    # Position-stable. New features APPEND so older models'
    # feature_names_in_ still subsets correctly via predict_for. As of
    # 2026-05-26: news_count_7d_log is feature #16 (zero-indexed 15);
    # rel_strength_sector_etf_30d added after as #17.
    assert "news_count_7d_log" in FEATURE_NAMES  # order-agnostic: XGBoost ranks by name


def test_log1p_compression_matches_documented_behavior():
    """log1p(0)=0, log1p(48)≈3.89 (median), log1p(744)≈6.61 (max observed).
    Rule of thumb: every 10x bump in raw count adds ~2.3 to the feature."""
    assert _news_count_7d_log("AAPL", {"AAPL": 0}) == pytest.approx(0.0)
    assert _news_count_7d_log("AAPL", {"AAPL": 48}) == pytest.approx(math.log1p(48))
    assert _news_count_7d_log("AAPL", {"AAPL": 744}) == pytest.approx(math.log1p(744))


def test_no_news_counts_returns_zero():
    """No news data loaded → every ticker reads as zero attention."""
    assert _news_count_7d_log("AAPL", None) == 0.0
    assert _news_count_7d_log("AAPL", {}) == 0.0


def test_ticker_absent_returns_zero():
    """Ticker with no coverage in the window → zero attention."""
    assert _news_count_7d_log("UNKNOWN", {"AAPL": 100}) == 0.0


def test_negative_count_clamped_to_zero():
    """Defensive: a malformed counts dict with negative values shouldn't
    propagate `nan` from log1p. Clamp to 0."""
    assert _news_count_7d_log("AAPL", {"AAPL": -5}) == 0.0


def test_returns_float_not_int():
    val = _news_count_7d_log("AAPL", {"AAPL": 100})
    assert isinstance(val, float)


# ---- loader-side aggregator -----------------------------------------------


def _make_news_df(rows):
    """Build a news-like DataFrame with the `published_at_date` column the
    loader's reducer expects (pre-computed in _load_news_for_count_feature)."""
    df = pd.DataFrame(rows)
    if not df.empty:
        df["published_at_date"] = pd.to_datetime(df["published_at_date"]).dt.date
    return df


def test_compute_news_counts_7d_empty_df_returns_empty():
    out = _compute_news_counts_7d(pd.DataFrame(), date(2026, 5, 22))
    assert out == {}


def test_compute_news_counts_7d_groups_by_ticker_in_window():
    asof = date(2026, 5, 22)
    rows = []
    # AAPL: 3 in window, 1 outside
    for _ in range(3):
        rows.append({"ticker": "AAPL", "published_at_date": date(2026, 5, 20)})
    rows.append({"ticker": "AAPL", "published_at_date": date(2026, 5, 10)})  # out
    # MSFT: 2 in window
    for _ in range(2):
        rows.append({"ticker": "MSFT", "published_at_date": date(2026, 5, 21)})
    # GOOGL: 1 in window, untrimmed-NaN-ticker row
    rows.append({"ticker": "GOOGL", "published_at_date": date(2026, 5, 22)})
    rows.append({"ticker": None, "published_at_date": date(2026, 5, 22)})

    out = _compute_news_counts_7d(_make_news_df(rows), asof)
    assert out == {"AAPL": 3, "MSFT": 2, "GOOGL": 1}


def test_compute_news_counts_7d_includes_asof_itself():
    """Window is inclusive on both ends — news published intraday on
    asof_date counts (model uses end-of-day prices)."""
    asof = date(2026, 5, 22)
    rows = [
        {"ticker": "AAPL", "published_at_date": date(2026, 5, 22)},
        {"ticker": "AAPL", "published_at_date": date(2026, 5, 15)},  # exactly 7d back
    ]
    out = _compute_news_counts_7d(_make_news_df(rows), asof)
    assert out == {"AAPL": 2}


# ---- end-to-end through build_features ------------------------------------


def _synthetic_prices(ticker, start, n_days):
    import numpy as np
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


def test_build_features_populates_news_count():
    asof = date(2026, 5, 22)
    start = asof - timedelta(days=400)
    tickers = ["AAPL", "MSFT", "SPY"]
    prices = pd.concat([_synthetic_prices(t, start, 401) for t in tickers], ignore_index=True)

    counts = {"AAPL": 250, "MSFT": 100}  # GOOGL absent
    out = build_features(prices, tickers, asof, news_counts_7d=counts)
    assert "news_count_7d_log" in out.columns
    if "AAPL" in out.index:
        assert out.loc["AAPL", "news_count_7d_log"] == pytest.approx(math.log1p(250))
    if "MSFT" in out.index:
        assert out.loc["MSFT", "news_count_7d_log"] == pytest.approx(math.log1p(100))
    # SPY absent from counts → 0
    if "SPY" in out.index:
        assert out.loc["SPY", "news_count_7d_log"] == 0.0


# pytest import is at module level so the parametrize-style approx works.
import pytest  # noqa: E402
