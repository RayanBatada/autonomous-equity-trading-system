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


# ---------------------------------------------------------------------------
# Complexity regression (2026-08-17). build_features used to be superlinear in
# universe size: the sector-leadership feature recomputed EVERY peer's 30d
# return for EVERY target, so a sector of S names did S*(S-1) computations for
# S distinct values. The breadth study measured 1.82 s/asof at 264 names vs
# 5.08 s/asof at 494 — 2.8x the cost for 1.87x the names. This asserts the
# shape of the work, not the wall clock, so it can't flake on a busy CI box.
# ---------------------------------------------------------------------------

def _count_ret_n_days_calls(n_names: int) -> int:
    from unittest.mock import patch

    from sma.features import technical

    names = [f"T{i:03d}" for i in range(n_names)]
    prices = _multi_ticker_prices(["SPY", *names], n_days=310)
    asof = prices[prices["ticker"] == "SPY"]["date"].max()
    # One shared sector: the worst case for the peer loop.
    sectors = {t: "Information Technology" for t in names}
    sectors["SPY"] = "ETF"

    calls = 0
    real = technical.ret_n_days

    def counting(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real(*args, **kwargs)

    with (
        patch.dict("sma.sectors.SECTORS", sectors, clear=False),
        patch.object(technical, "ret_n_days", counting),
    ):
        result = build_features(prices, ["SPY", *names], asof)
    assert len(result) == n_names + 1, "fixture must produce a full cross-section"
    return calls


def test_build_features_work_is_linear_in_universe_size():
    """Doubling the universe must roughly double the return computations.

    Under the old peer loop this ratio was ~3.5-4x (quadratic); a regression
    back to per-target peer recomputation trips this immediately.
    """
    c10 = _count_ret_n_days_calls(10)
    c20 = _count_ret_n_days_calls(20)
    ratio = c20 / c10
    assert ratio < 2.3, (
        f"build_features went superlinear again: {c10} calls at 10 names, "
        f"{c20} at 20 ({ratio:.2f}x for 2x the names)"
    )


def test_build_features_sector_feature_unchanged_by_peer_precompute():
    """The precomputed-returns path must produce the same sector-relative
    strength as computing each peer's return inside the loop."""
    from sma.features import technical

    tickers = ["SPY", "AAA", "BBB", "CCC", "DDD"]
    prices = _multi_ticker_prices(tickers, n_days=310)
    asof = prices[prices["ticker"] == "SPY"]["date"].max()
    peers = ["AAA", "BBB", "CCC", "DDD"]
    sectors = {t: "Information Technology" for t in peers}
    sectors["SPY"] = "ETF"

    from unittest.mock import patch
    with patch.dict("sma.sectors.SECTORS", sectors, clear=False):
        result = build_features(prices, tickers, asof)

    for target in peers:
        others = [p for p in peers if p != target]
        expected = technical.rel_strength_sector_30d(
            prices[prices["ticker"] == target],
            [prices[prices["ticker"] == p] for p in others],
            asof,
        )
        assert result.loc[target, "rel_strength_sector_30d"] == expected


# ---------------------------------------------------------------------------
# Truncation regression (2026-08-27). Each of the ~22 per-ticker feature
# functions used to open by re-masking and re-sorting the ticker's WHOLE
# history for itself (technical._rows_up_to): 226,352 calls for ~11,000
# distinct answers, 67.0s of a 101.4s feature build on a 266-name x 2y slice.
# build_features now truncates ONCE per (ticker, asof) and hands the same
# PriceWindow to every feature. Both halves of that are asserted on call SHAPE,
# not wall clock, so neither can flake on a busy CI box — and a regression is
# INVISIBLE in the output (identical values, just slower), which is exactly why
# it needs a test rather than a benchmark.
# ---------------------------------------------------------------------------

def test_build_features_never_lets_a_feature_truncate_for_itself():
    """Every _rows_up_to call inside build_features must receive a PriceWindow."""
    from unittest.mock import patch

    from sma.features import technical
    from sma.features.window import PriceWindow

    tickers = ["SPY", "AAA", "BBB", "CCC"]
    prices = _multi_ticker_prices(tickers, n_days=310)
    asof = prices["date"].max()

    raw_frame_calls: list = []
    window_calls: list = []
    real = technical._rows_up_to

    def watching(prices_arg, asof_date):
        target = window_calls if isinstance(prices_arg, PriceWindow) else raw_frame_calls
        target.append(prices_arg)
        return real(prices_arg, asof_date)

    with patch.object(technical, "_rows_up_to", watching):
        result = build_features(prices, tickers, asof)

    assert len(result) == len(tickers), "fixture must produce a full cross-section"
    assert window_calls, "the feature functions must actually have run"
    assert not raw_frame_calls, (
        f"{len(raw_frame_calls)} feature calls re-truncated a raw price frame; "
        "build_features must hand every feature a window it built once"
    )


def _windows_built_and_rows_requested(n_names: int) -> tuple[int, int]:
    """(PriceWindows built, times a feature asked for its visible rows)."""
    from unittest.mock import patch

    from sma.features import technical, window

    names = [f"T{i:03d}" for i in range(n_names)]
    tickers = ["SPY", *names]
    prices = _multi_ticker_prices(tickers, n_days=310)
    asof = prices["date"].max()

    built = 0
    real_init = window.PriceWindow.__init__

    def counting_init(self, rows, asof_date):
        nonlocal built
        built += 1
        real_init(self, rows, asof_date)

    requested = 0
    real_rows_up_to = technical._rows_up_to

    def counting_rows_up_to(prices_arg, asof_date):
        nonlocal requested
        requested += 1
        return real_rows_up_to(prices_arg, asof_date)

    with (
        patch.object(window.PriceWindow, "__init__", counting_init),
        patch.object(technical, "_rows_up_to", counting_rows_up_to),
    ):
        result = build_features(prices, tickers, asof)
    assert len(result) == len(tickers), "fixture must produce a full cross-section"
    return built, requested


def test_build_features_truncates_once_per_ticker_not_once_per_feature():
    """~22 features per ticker, ONE truncation per ticker.

    The window count is bounded by the universe (plus the SPY benchmark, which
    is windowed both as a universe member and as the benchmark), while the
    number of times a feature asks for its visible rows stays ~22x that. The
    old code's counts were equal.
    """
    n_names = 12
    built, requested = _windows_built_and_rows_requested(n_names)
    n_tickers = n_names + 1  # + SPY
    assert built <= n_tickers + 2, (
        f"{built} windows built for {n_tickers} tickers — build_features must "
        "truncate once per (ticker, asof), not once per feature"
    )
    assert requested > 10 * built, (
        f"only {requested} feature reads against {built} truncations; the "
        "fixture is no longer exercising the shared window"
    )


def test_build_features_scales_truncations_with_tickers_not_features():
    """Doubling the universe doubles the truncations. Under the old code it
    doubled them too — but from a base ~22x higher, which is what the absolute
    bound above pins. This pins the SHAPE so a partial regression (some
    features windowed, some not) also trips."""
    built_10, _ = _windows_built_and_rows_requested(10)
    built_20, _ = _windows_built_and_rows_requested(20)
    assert built_20 - built_10 == 10, (
        f"{built_10} windows at 10 names, {built_20} at 20 — each extra ticker "
        "must cost exactly one more truncation"
    )
