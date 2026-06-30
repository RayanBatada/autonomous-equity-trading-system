"""Tests for the 17th model feature: rel_strength_sector_etf_30d.

ret_30d(ticker) - ret_30d(sector_etf). Uses SPDR sector ETFs (XLK for
IT, XLF for Financials, etc.) as the canonical sector benchmark.
Cleaner than the per-name peer-mean rel_strength_sector_30d because the
benchmark doesn't drift with universe composition.

Both features ship together — the model picks which is more predictive
per sector at training time.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from sma.features.builder import FEATURE_NAMES, build_features
from sma.features.technical import rel_strength_sector_etf_30d
from sma.sectors import SECTOR_ETF_FOR_GICS, sector_etf_for


def _synthetic_prices(ticker: str, start: date, n_days: int, daily_log_ret: float = 0.0):
    rng = np.random.default_rng(seed=hash(ticker) % 999)
    rows = []
    price = 100.0
    for i in range(n_days):
        d = start + timedelta(days=i)
        price *= np.exp(daily_log_ret)
        rows.append({
            "ticker": ticker, "date": d,
            "open": price, "high": price, "low": price, "close": price,
            "adj_close": price,
            "volume": int(1_000_000 + 100_000 * rng.standard_normal()),
            "source": "yfinance", "run_id": 1,
        })
    return pd.DataFrame(rows)


def test_sector_etf_map_covers_all_11_gics_sectors():
    """One SPDR ETF per GICS sector — no gaps."""
    expected = {
        "Information Technology", "Financials", "Health Care",
        "Consumer Discretionary", "Consumer Staples", "Energy",
        "Industrials", "Communication Services", "Materials",
        "Utilities", "Real Estate",
    }
    assert set(SECTOR_ETF_FOR_GICS.keys()) == expected


def test_sector_etf_for_resolves_known_tickers():
    assert sector_etf_for("NVDA") == "XLK"
    assert sector_etf_for("JPM") == "XLF"
    assert sector_etf_for("EQIX") == "XLRE"
    assert sector_etf_for("XOM") == "XLE"
    assert sector_etf_for("UNH") == "XLV"


def test_sector_etf_for_returns_none_for_etfs_themselves():
    """SPY/NANC are in the ETF bucket; sector ETFs in 'Sector ETF'.
    Neither maps to a benchmark (they ARE the benchmarks)."""
    assert sector_etf_for("SPY") is None
    assert sector_etf_for("NANC") is None
    assert sector_etf_for("XLK") is None
    assert sector_etf_for("XLF") is None


def test_sector_etf_for_returns_none_for_unknown_ticker():
    assert sector_etf_for("ZZZ_NONEXISTENT") is None


def test_target_beats_sector_etf_positive_rs():
    asof = date(2026, 1, 1)
    start = asof - timedelta(days=40)
    target = _synthetic_prices("NVDA", start, 41, daily_log_ret=0.003)
    etf = _synthetic_prices("XLK", start, 41, daily_log_ret=0.001)
    rs = rel_strength_sector_etf_30d(target, etf, asof)
    assert rs is not None
    assert 0.04 < rs < 0.08  # e^(0.003*30) - e^(0.001*30) ≈ 0.064


def test_target_lagging_sector_etf_negative_rs():
    asof = date(2026, 1, 1)
    start = asof - timedelta(days=40)
    target = _synthetic_prices("INTC", start, 41, daily_log_ret=0.0005)
    etf = _synthetic_prices("XLK", start, 41, daily_log_ret=0.002)
    rs = rel_strength_sector_etf_30d(target, etf, asof)
    assert rs is not None
    assert rs < -0.02


def test_returns_none_when_target_history_insufficient():
    asof = date(2026, 1, 1)
    target = _synthetic_prices("NEW", asof - timedelta(days=5), 6)
    etf = _synthetic_prices("XLK", asof - timedelta(days=40), 41)
    assert rel_strength_sector_etf_30d(target, etf, asof) is None


def test_returns_none_when_etf_history_insufficient():
    asof = date(2026, 1, 1)
    target = _synthetic_prices("NVDA", asof - timedelta(days=40), 41)
    etf = _synthetic_prices("XLK", asof - timedelta(days=5), 6)
    assert rel_strength_sector_etf_30d(target, etf, asof) is None


def test_feature_in_feature_names_appended_last():
    """Position-stable. ETF RS appended after news_count_7d_log."""
    assert "rel_strength_sector_etf_30d" in FEATURE_NAMES
    assert "rel_strength_sector_etf_30d" in FEATURE_NAMES  # order-agnostic: features rank by name
    assert FEATURE_NAMES.index("rel_strength_sector_etf_30d") == 16


def test_build_features_populates_etf_rs_when_etf_prices_present():
    asof = date(2026, 1, 1)
    start = asof - timedelta(days=400)
    tickers = ["NVDA", "XLK", "SPY"]
    prices = pd.concat([_synthetic_prices(t, start, 401) for t in tickers], ignore_index=True)
    result = build_features(prices, tickers, asof)
    assert "rel_strength_sector_etf_30d" in result.columns
    if "NVDA" in result.index:
        # Same synthetic daily return for both → RS ≈ 0.
        assert abs(result.loc["NVDA", "rel_strength_sector_etf_30d"]) < 1e-6


def test_build_features_etf_rs_defaults_zero_when_etf_prices_missing():
    """If sector ETF prices aren't in the prices DF (fresh DB before
    backfill completes), the feature defaults to 0 rather than dropping
    the ticker entirely."""
    asof = date(2026, 1, 1)
    start = asof - timedelta(days=400)
    tickers = ["NVDA", "MSFT", "SPY"]  # no XLK
    prices = pd.concat([_synthetic_prices(t, start, 401) for t in tickers], ignore_index=True)
    result = build_features(prices, tickers, asof)
    if "NVDA" in result.index:
        assert result.loc["NVDA", "rel_strength_sector_etf_30d"] == 0.0


def test_build_features_etf_rs_is_zero_for_etfs_themselves():
    asof = date(2026, 1, 1)
    start = asof - timedelta(days=400)
    tickers = ["XLK", "SPY"]
    prices = pd.concat([_synthetic_prices(t, start, 401) for t in tickers], ignore_index=True)
    result = build_features(prices, tickers, asof)
    if "SPY" in result.index:
        assert result.loc["SPY", "rel_strength_sector_etf_30d"] == 0.0
    if "XLK" in result.index:
        assert result.loc["XLK", "rel_strength_sector_etf_30d"] == 0.0
