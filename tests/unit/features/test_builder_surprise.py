"""earnings_surprise_last: builder consumes a pre-resolved {ticker: surprise}
map (same caller-resolves pattern as politician_flows / earnings_calendar)."""
from datetime import date, timedelta

import pandas as pd

from sma.features.builder import FEATURE_NAMES, build_features


def _prices(tickers, n=300):
    rows = []
    for t in tickers:
        px = 100.0
        d = date(2025, 1, 1)
        for i in range(n):
            px *= 1.0005 * (1.002 if i % 2 else 0.998)
            rows.append({"ticker": t, "date": d, "open": px, "high": px, "low": px,
                         "close": px, "adj_close": px, "volume": 1_000_000 + i * 1_000})
            d += timedelta(days=1)
    return pd.DataFrame(rows)


def test_surprise_in_feature_names_and_passed_through():
    assert "earnings_surprise_last" in FEATURE_NAMES
    prices = _prices(["SPY", "AAPL", "MSFT"])
    asof = prices["date"].max()
    df = build_features(
        prices, ["AAPL", "MSFT"], asof,
        earnings_surprises={"AAPL": -0.25},
    )
    assert float(df.loc["AAPL", "earnings_surprise_last"]) == -0.25
    assert float(df.loc["MSFT", "earnings_surprise_last"]) == 0.0  # absent -> neutral
