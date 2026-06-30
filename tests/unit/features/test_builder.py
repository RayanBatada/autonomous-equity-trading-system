"""Tests for the batch feature builder (build_features)."""

from datetime import date, timedelta

import pandas as pd

from sma.features.builder import FEATURE_NAMES, build_features

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _multi_ticker_prices(
    tickers: list[str],
    n_days: int,
    start: date = date(2024, 1, 1),
) -> pd.DataFrame:
    """Build a deterministic multi-ticker price DataFrame with monotonic close
    prices (so all features resolve cleanly)."""
    rows = []
    for t_idx, t in enumerate(tickers):
        base = 100.0 + t_idx * 10
        d = start
        for i in range(n_days):
            c = base + i * 0.5  # gentle uptrend
            rows.append({
                "ticker": t, "date": d,
                "open": c - 0.1, "high": c + 0.1, "low": c - 0.1,
                "close": c, "adj_close": c, "volume": 1_000_000 + i * 100,
            })
            d = d + timedelta(days=1)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Test 1: basic return value shape
# ---------------------------------------------------------------------------

def test_build_features_returns_indexed_dataframe():
    """build_features returns a DataFrame indexed by ticker with all 12 features."""
    tickers = ["SPY", "AAPL", "MSFT"]
    prices = _multi_ticker_prices(tickers, n_days=310)
    asof = prices[prices["ticker"] == "SPY"]["date"].max()

    result = build_features(prices, tickers, asof)

    assert isinstance(result, pd.DataFrame)
    assert result.index.name == "ticker"
    assert list(result.columns) == FEATURE_NAMES
    assert set(tickers) == set(result.index.tolist())
    # All values must be floats, no None or NaN.
    assert result.notna().all().all()
    for col in FEATURE_NAMES:
        assert result[col].dtype == float or pd.api.types.is_float_dtype(result[col])


# ---------------------------------------------------------------------------
# Test 2: insufficient history -> ticker dropped
# ---------------------------------------------------------------------------

def test_build_features_drops_tickers_with_insufficient_history():
    """Tickers with too few rows to compute any feature are dropped."""
    good_tickers = ["SPY", "AAPL"]
    prices_good = _multi_ticker_prices(good_tickers, n_days=310)

    # Build SHORT price history for "THIN" -- only 50 rows.
    thin_prices = _multi_ticker_prices(["THIN"], n_days=50)
    prices = pd.concat([prices_good, thin_prices], ignore_index=True)

    asof = prices_good[prices_good["ticker"] == "SPY"]["date"].max()
    universe = ["SPY", "AAPL", "THIN"]

    result = build_features(prices, universe, asof)

    assert "THIN" not in result.index
    assert "SPY" in result.index
    assert "AAPL" in result.index


# ---------------------------------------------------------------------------
# Test 3: ticker not in prices at all -> no crash, not in output
# ---------------------------------------------------------------------------

def test_build_features_drops_ticker_missing_from_prices():
    """A ticker in the universe but absent from prices is silently omitted."""
    tickers = ["SPY", "AAPL"]
    prices = _multi_ticker_prices(tickers, n_days=310)
    asof = prices[prices["ticker"] == "SPY"]["date"].max()

    universe = ["SPY", "AAPL", "ZZZ"]  # ZZZ has no rows
    result = build_features(prices, universe, asof)

    assert "ZZZ" not in result.index
    assert "SPY" in result.index
    assert "AAPL" in result.index


# ---------------------------------------------------------------------------
# Test 4: empty universe -> empty DataFrame
# ---------------------------------------------------------------------------

def test_build_features_returns_empty_when_universe_empty():
    """Empty universe returns an empty DataFrame with the right columns."""
    tickers = ["SPY", "AAPL"]
    prices = _multi_ticker_prices(tickers, n_days=310)
    asof = prices[prices["ticker"] == "SPY"]["date"].max()

    result = build_features(prices, [], asof)

    assert isinstance(result, pd.DataFrame)
    assert result.empty
    assert list(result.columns) == FEATURE_NAMES
    assert result.index.name == "ticker"


# ---------------------------------------------------------------------------
# Test 5: adversarial lookahead test
# ---------------------------------------------------------------------------

def test_build_features_lookahead_clean_adversarial():
    """Adding future rows to prices must not change any feature value."""
    tickers = ["SPY", "AAPL", "MSFT"]
    n_days = 310
    start = date(2024, 1, 1)
    prices_base = _multi_ticker_prices(tickers, n_days=n_days, start=start)
    asof = prices_base[prices_base["ticker"] == "SPY"]["date"].max()

    result_base = build_features(prices_base, tickers, asof)

    # Append 20 future rows with garbage values for every ticker.
    future_rows = []
    for ticker in tickers:
        d = asof + timedelta(days=1)
        for _ in range(20):
            future_rows.append({
                "ticker": ticker, "date": d,
                "open": 9999.0, "high": 9999.0, "low": 0.01,
                "close": 9999.0, "adj_close": 9999.0, "volume": 999_999_999,
            })
            d = d + timedelta(days=1)

    prices_augmented = pd.concat(
        [prices_base, pd.DataFrame(future_rows)], ignore_index=True
    )

    result_augmented = build_features(prices_augmented, tickers, asof)

    pd.testing.assert_frame_equal(result_base, result_augmented)


# ---------------------------------------------------------------------------
# Test 6: missing SPY -> rel_strength_spy_60d returns None -> all tickers dropped
# ---------------------------------------------------------------------------

def test_build_features_handles_missing_spy():
    """When SPY rows are absent, rel_strength_spy_60d returns None for every
    ticker, so all are dropped and the result is empty."""
    # Build prices without SPY.
    tickers = ["AAPL", "MSFT"]
    prices = _multi_ticker_prices(tickers, n_days=310)
    asof = prices[prices["ticker"] == "AAPL"]["date"].max()

    result = build_features(prices, tickers, asof)

    # rel_strength_spy_60d will return None (spy_prices is empty),
    # so every ticker should be dropped.
    assert result.empty
