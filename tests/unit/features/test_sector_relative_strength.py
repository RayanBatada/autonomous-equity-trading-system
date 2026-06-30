"""Tests for the 14th model feature: rel_strength_sector_30d.

Captures within-sector leadership — a name that beats its sector mean
30-day return is structurally different from one that's just riding the
sector wave. SPY-relative strength conflates both; this isolates the
alpha component.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from sma.features.builder import FEATURE_NAMES, build_features
from sma.features.technical import rel_strength_sector_30d


def _synthetic_prices(ticker: str, start: date, n_days: int, daily_log_ret: float = 0.0):
    """Constant log-return series so ret_30d is deterministic. Volume is
    varied (sin wave) so volume_z_20d's stddev is non-zero — otherwise
    that feature returns None and the ticker is dropped from build_features."""
    rng = np.random.default_rng(seed=hash(ticker) % 999)
    rows = []
    price = 100.0
    for i in range(n_days):
        d = start + timedelta(days=i)
        price *= np.exp(daily_log_ret)
        rows.append({
            "ticker": ticker,
            "date": d,
            "open": price, "high": price, "low": price,
            "close": price, "adj_close": price,
            "volume": int(1_000_000 + 100_000 * rng.standard_normal()),
            "source": "yfinance",
            "run_id": 1,
        })
    return pd.DataFrame(rows)


def test_feature_in_feature_names():
    """Order matters — new features append at the END so older models
    (which use model.feature_names_in_) still see the same column order
    for their known features."""
    assert "rel_strength_sector_30d" in FEATURE_NAMES
    # Sector RS is the second-newest; days_to_next_earnings was added after.
    assert FEATURE_NAMES.index("rel_strength_sector_30d") == 13
    assert len(FEATURE_NAMES) >= 14


def test_rel_strength_sector_30d_positive_alpha():
    """Target +0.2% daily, peers +0.1% daily over 30d → target beats peers."""
    asof = date(2026, 1, 1)
    start = asof - timedelta(days=40)
    target = _synthetic_prices("AAPL", start, 41, daily_log_ret=0.002)
    peers = [
        _synthetic_prices("MSFT", start, 41, daily_log_ret=0.001),
        _synthetic_prices("GOOGL", start, 41, daily_log_ret=0.001),
    ]
    rs = rel_strength_sector_30d(target, peers, asof)
    assert rs is not None
    # Target 30d ret ≈ e^(0.002*30) - 1 ≈ 0.0618
    # Peer mean ≈ e^(0.001*30) - 1 ≈ 0.0305
    # Diff ≈ 0.0313
    assert 0.02 < rs < 0.05


def test_rel_strength_sector_30d_negative_alpha():
    """Target lagging its sector → negative value."""
    asof = date(2026, 1, 1)
    start = asof - timedelta(days=40)
    target = _synthetic_prices("AAPL", start, 41, daily_log_ret=0.001)
    peers = [
        _synthetic_prices("MSFT", start, 41, daily_log_ret=0.003),
        _synthetic_prices("GOOGL", start, 41, daily_log_ret=0.003),
    ]
    rs = rel_strength_sector_30d(target, peers, asof)
    assert rs is not None
    assert rs < -0.03


def test_rel_strength_sector_30d_returns_none_when_target_has_no_history():
    asof = date(2026, 1, 1)
    target = _synthetic_prices("AAPL", asof - timedelta(days=5), 6)  # too short
    peers = [
        _synthetic_prices("MSFT", asof - timedelta(days=40), 41),
        _synthetic_prices("GOOGL", asof - timedelta(days=40), 41),
    ]
    assert rel_strength_sector_30d(target, peers, asof) is None


def test_rel_strength_sector_30d_returns_none_when_only_one_peer():
    """A 1-peer sector mean isn't statistically meaningful — neutral
    (None) is more honest than a noisy comparison against a single name."""
    asof = date(2026, 1, 1)
    start = asof - timedelta(days=40)
    target = _synthetic_prices("AAPL", start, 41, daily_log_ret=0.001)
    peers = [_synthetic_prices("MSFT", start, 41, daily_log_ret=0.001)]  # 1 peer
    assert rel_strength_sector_30d(target, peers, asof) is None


def test_rel_strength_sector_30d_skips_peers_with_no_data():
    """Peers with insufficient history get dropped; remaining ones still
    feed the mean. Robustness against the bootstrap window when a new
    ticker is added to the universe mid-quarter."""
    asof = date(2026, 1, 1)
    start = asof - timedelta(days=40)
    target = _synthetic_prices("AAPL", start, 41, daily_log_ret=0.002)
    peers = [
        _synthetic_prices("MSFT", start, 41, daily_log_ret=0.001),  # valid
        _synthetic_prices("NEW1", asof - timedelta(days=5), 6),     # too short
        _synthetic_prices("GOOGL", start, 41, daily_log_ret=0.001), # valid
    ]
    rs = rel_strength_sector_30d(target, peers, asof)
    assert rs is not None  # 2 valid peers is enough


def test_build_features_emits_sector_relative_strength_for_real_ticker():
    """End-to-end: build_features computes the new column for a real
    ticker using its actual GICS sector peers from sma.sectors."""
    asof = date(2026, 1, 1)
    start = asof - timedelta(days=400)
    # Use real tickers with real GICS sector mappings (Information Technology).
    tickers = ["AAPL", "MSFT", "GOOGL", "NVDA", "SPY"]
    prices = pd.concat([_synthetic_prices(t, start, 401) for t in tickers], ignore_index=True)
    result = build_features(prices, tickers, asof)
    assert "rel_strength_sector_30d" in result.columns
    # All synthetic, so the within-sector RS is ~0.
    for ticker in ("AAPL", "MSFT", "GOOGL", "NVDA"):
        rs = result.loc[ticker, "rel_strength_sector_30d"]
        assert abs(rs) < 1e-6, f"{ticker} expected ~0 RS, got {rs}"


def test_build_features_returns_neutral_for_etf_sector():
    """ETFs (SPY, NANC) live in the synthetic 'ETF' bucket; sector-RS
    isn't meaningful for them. Default 0.0 is honest neutrality."""
    asof = date(2026, 1, 1)
    start = asof - timedelta(days=400)
    prices = _synthetic_prices("SPY", start, 401, daily_log_ret=0.001)
    result = build_features(prices, ["SPY"], asof)
    assert result.loc["SPY", "rel_strength_sector_30d"] == 0.0


def test_build_features_neutralizes_sector_etf_bucket():
    """The SPDR sector ETFs (XLK, XLE, XLF, ...) are benchmarks, not
    within-sector members of one another. Their rel_strength_sector_30d
    must be neutral (0.0), not computed against the other sector ETFs — a
    meaningless cross-sector peer group.

    Regression: the builder's skip only excluded the 'ETF' bucket
    (SPY/NANC); the 'Sector ETF' bucket leaked into peer grouping, so each
    of the 11 sector ETFs got a polluted within-sector feature in every
    training/predict row."""
    asof = date(2026, 1, 1)
    start = asof - timedelta(days=400)
    # XLK climbs; the other sector ETFs are flat. A (buggy) cross-ETF peer
    # comparison would hand XLK a strongly non-zero within-sector RS.
    # SPY is included so rel_strength_spy_60d isn't None (else the row drops).
    prices = pd.concat(
        [
            _synthetic_prices("XLK", start, 401, daily_log_ret=0.002),
            _synthetic_prices("XLE", start, 401, daily_log_ret=0.0),
            _synthetic_prices("XLF", start, 401, daily_log_ret=0.0),
            _synthetic_prices("SPY", start, 401, daily_log_ret=0.0),
        ],
        ignore_index=True,
    )
    result = build_features(prices, ["XLK", "XLE", "XLF", "SPY"], asof)
    assert result.loc["XLK", "rel_strength_sector_30d"] == 0.0
    assert result.loc["XLE", "rel_strength_sector_30d"] == 0.0
    assert result.loc["XLF", "rel_strength_sector_30d"] == 0.0
