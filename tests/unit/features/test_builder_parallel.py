"""build_features must be BIT-IDENTICAL at any worker count.

The parallel path splits the universe into contiguous chunks and concatenates
their results in chunk order; the cross-ticker input (each peer's 30d return,
which feeds the sector-leadership mean) is precomputed serially before the
fan-out. This file pins both halves of that: same values, same ROW ORDER.

Real tickers are used deliberately — synthetic symbols are absent from
sma.sectors.SECTORS, which would leave rel_strength_sector_30d and
rel_strength_sector_etf_30d at their 0.0 defaults and never exercise the
cross-ticker paths that chunking could break.
"""

from datetime import date, timedelta

import pandas as pd
from pandas.testing import assert_frame_equal

from sma.features.builder import build_features

# Three GICS sectors with 2-3 peers each (so the >= 2 peer threshold is
# cleared and the peer mean is a real sum), one sector ETF benchmark, plus SPY.
UNIVERSE = [
    "AAPL", "MSFT", "NVDA",   # Information Technology  -> XLK
    "JPM", "BAC",             # Financials              -> XLF (absent below)
    "XOM", "CVX",             # Energy                  -> XLE (absent below)
    "JNJ", "PFE",             # Health Care             -> XLV (absent below)
    "SPY",                    # "ETF"        — no peer group, no ETF benchmark
    "XLK",                    # "Sector ETF" — benchmark for the IT names
]


def _prices(tickers: list[str], n_days: int, start: date = date(2024, 1, 1)):
    """Deterministic multi-ticker prices. Each ticker gets its own drift and a
    sinusoidal volume so no feature degenerates to None and the tickers do not
    all share a value (which would hide an ordering bug)."""
    rows = []
    for t_idx, t in enumerate(tickers):
        base = 100.0 + t_idx * 7.5
        for i in range(n_days):
            c = base + i * (0.25 + 0.03 * t_idx) + ((i * (t_idx + 3)) % 11) * 0.1
            rows.append({
                "ticker": t, "date": start + timedelta(days=i),
                "open": c - 0.13, "high": c + 0.4, "low": c - 0.5,
                "close": c, "adj_close": c,
                "volume": 1_000_000 + (i * 1_000 * (t_idx + 1)) % 250_000,
            })
    return pd.DataFrame(rows)


def test_build_features_bit_identical_serial_vs_parallel():
    prices = _prices(UNIVERSE, n_days=320)
    asof = prices["date"].max()

    serial = build_features(prices, UNIVERSE, asof, workers=1)
    parallel = build_features(prices, UNIVERSE, asof, workers=2)

    assert len(serial) == len(UNIVERSE), "fixture must yield a full cross-section"
    assert serial["rel_strength_sector_30d"].abs().sum() > 0, (
        "sector peer feature must actually be exercised, not all-zero"
    )
    assert_frame_equal(serial, parallel, check_exact=True)
    assert list(serial.index) == list(parallel.index), "row ORDER, not just values"


def test_build_features_bit_identical_with_every_optional_input():
    """The optional per-asof dicts ride along in the shared context; a worker
    must see the same ones the parent was handed."""
    prices = _prices(UNIVERSE, n_days=320)
    asof = prices["date"].max()
    kwargs = dict(
        politician_flows={"AAPL": 125_000.0, "JPM": -40_000.0},
        earnings_calendar={"MSFT": asof + timedelta(days=9), "PFE": asof + timedelta(days=88)},
        earnings_surprises={"NVDA": 0.42, "XOM": -0.17},
        news_counts_7d={"AAPL": 512, "BAC": 3},
    )
    serial = build_features(prices, UNIVERSE, asof, workers=1, **kwargs)
    parallel = build_features(prices, UNIVERSE, asof, workers=3, **kwargs)
    assert_frame_equal(serial, parallel, check_exact=True)


def test_build_features_parallel_is_deterministic_across_runs():
    """Two independent parallel runs must agree bit-for-bit — the same
    guarantee as serial-vs-parallel, but catching anything that depends on
    which worker happened to finish first."""
    prices = _prices(UNIVERSE, n_days=320)
    asof = prices["date"].max()
    a = build_features(prices, UNIVERSE, asof, workers=2)
    b = build_features(prices, UNIVERSE, asof, workers=2)
    assert_frame_equal(a, b, check_exact=True)


def test_build_features_parallel_drops_the_same_tickers():
    """A ticker with too little history, and one with no rows at all, must fall
    out of the parallel path exactly as they do serially — the row list is not
    parallel to the chunk it came from."""
    prices = pd.concat(
        [_prices(UNIVERSE, n_days=320), _prices(["ORCL"], n_days=40)],
        ignore_index=True,
    )
    universe = [*UNIVERSE, "ORCL", "ZZZZ"]  # ORCL too short, ZZZZ absent
    asof = _prices(UNIVERSE, n_days=320)["date"].max()

    serial = build_features(prices, universe, asof, workers=1)
    parallel = build_features(prices, universe, asof, workers=4)

    assert "ORCL" not in serial.index and "ZZZZ" not in serial.index
    assert_frame_equal(serial, parallel, check_exact=True)


def test_build_features_empty_universe_is_identical_at_any_worker_count():
    prices = _prices(UNIVERSE, n_days=320)
    asof = prices["date"].max()
    assert_frame_equal(
        build_features(prices, [], asof, workers=1),
        build_features(prices, [], asof, workers=4),
        check_exact=True,
    )
