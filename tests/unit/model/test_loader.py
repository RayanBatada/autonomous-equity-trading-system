"""Tests for build_training_set in sma.model.loader."""

from datetime import date, timedelta

import pandas as pd
import pytest

from sma.features.builder import FEATURE_NAMES
from sma.model.loader import _compute_politician_flows, build_training_set

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _multi_ticker_prices(
    tickers: list[str],
    n_days: int,
    start: date = date(2024, 1, 1),
) -> pd.DataFrame:
    """Deterministic multi-ticker price DataFrame with gentle uptrend.

    Uses consecutive calendar days (no weekend gaps) to keep indexing simple
    in tests.
    """
    rows = []
    for t_idx, t in enumerate(tickers):
        base = 100.0 + t_idx * 10
        d = start
        for i in range(n_days):
            c = base + i * 0.5
            rows.append({
                "ticker": t, "date": d,
                "open": c - 0.1, "high": c + 0.1, "low": c - 0.1,
                "close": c, "adj_close": c, "volume": 1_000_000 + i * 100,
            })
            d = d + timedelta(days=1)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Politician-flow sign parity with the predictor (train/serve skew guard)
# ---------------------------------------------------------------------------

def test_compute_politician_flows_non_ps_is_neutral():
    """Buy='P', sell='S%', everything else (e.g. 'E' exchange) contributes 0 —
    matching predictor._fetch_politician_flows' serve-time SQL. Previously the
    training path counted any non-'S' type (including 'E') as a BUY, skewing
    the politician_flow_30d feature between train and serve."""
    fd = date(2025, 1, 10)  # filing (disclosure) date — what the window uses now
    trades = pd.DataFrame([
        {"ticker": "AAA", "transaction_date": fd, "filing_date": fd,
         "transaction_type": "P", "amount_min": 1000, "amount_max": 3000},
        {"ticker": "BBB", "transaction_date": fd, "filing_date": fd,
         "transaction_type": "S", "amount_min": 1000, "amount_max": 3000},
        {"ticker": "CCC", "transaction_date": fd, "filing_date": fd,
         "transaction_type": "S (partial)", "amount_min": 1000, "amount_max": 3000},
        {"ticker": "DDD", "transaction_date": fd, "filing_date": fd,
         "transaction_type": "E", "amount_min": 1000, "amount_max": 3000},
    ])
    flows = _compute_politician_flows(trades, date(2025, 1, 15), lookback_days=30)
    assert flows["AAA"] == 2000.0    # P  -> +midpoint
    assert flows["BBB"] == -2000.0   # S  -> -midpoint
    assert flows["CCC"] == -2000.0   # S (partial) -> -midpoint (LIKE 'S%')
    assert flows.get("DDD", 0.0) == 0.0  # E -> neutral (was +2000 pre-fix)


# ---------------------------------------------------------------------------
# Test 1: basic shape
# ---------------------------------------------------------------------------

def test_build_training_set_basic_shape():
    """X, y, asof_dates all have the same positive length; columns match FEATURE_NAMES."""
    # Need 252 rows for dist_from_52w_high to resolve, plus 30 for forward labels,
    # plus some wiggle room for the training window.
    tickers = ["SPY", "AAA", "BBB"]
    prices = _multi_ticker_prices(tickers, n_days=320)

    all_dates = sorted(prices["date"].unique())
    # train_start just before features can first resolve (day 252).
    train_start = all_dates[250]
    # Leave at least 30 trading days at the end for forward labels.
    train_end = all_dates[-(30 + 5)]

    x, y, asof_dates = build_training_set(
        prices, ["AAA", "BBB"],
        train_start=train_start,
        train_end=train_end,
        forward_horizon_days=30,
    )

    assert list(x.columns) == FEATURE_NAMES
    assert len(x) == len(y) == len(asof_dates)
    assert len(x) > 0, "Expected non-empty training set"
    # All feature values are float, no NaN.
    assert x.notna().all().all()
    for col in FEATURE_NAMES:
        assert pd.api.types.is_float_dtype(x[col])
    assert pd.api.types.is_float_dtype(y)


# ---------------------------------------------------------------------------
# Test 2: last 30 trading days of data excluded
# ---------------------------------------------------------------------------

def test_build_training_set_excludes_last_30_trading_days():
    """No asof_date in the result falls within the last 30 trading days of prices."""
    tickers = ["SPY", "AAA", "BBB"]
    prices = _multi_ticker_prices(tickers, n_days=320)

    all_dates = sorted(prices["date"].unique())
    train_start = all_dates[250]
    train_end = all_dates[-1]  # deliberately include last dates

    _, _, asof_dates = build_training_set(
        prices, ["AAA", "BBB"],
        train_start=train_start,
        train_end=train_end,
        forward_horizon_days=30,
    )

    if len(asof_dates) == 0:
        pytest.skip("No rows produced; cannot assert date boundary")

    # The last asof_date with a valid 30-day forward label must be
    # all_dates[-(30 + 1)] at most (index such that future_idx < len).
    max_valid_asof = all_dates[-(30 + 1)]
    assert asof_dates.max() <= max_valid_asof


# ---------------------------------------------------------------------------
# Test 3: hand-computable label
# ---------------------------------------------------------------------------

def test_build_training_set_label_is_next_open_to_forward_close():
    """Label entry is the NEXT session's split-adjusted OPEN — where a DAY_OPG
    buy actually fills — not close[asof]. Exit is adj_close N sessions out. This
    drops the untradable close[asof]->open[asof+1] overnight leg the model could
    otherwise 'capture' but never trade."""
    # 320 days: features resolve at 252, leaving a window to pick a known asof.
    tickers = ["SPY", "AAA"]
    prices = _multi_ticker_prices(tickers, n_days=320)

    all_dates = sorted(prices["date"].unique())
    asof_idx = 260  # 260 rows behind, 59 rows ahead
    asof = all_dates[asof_idx]
    entry = all_dates[asof_idx + 1]          # buy fills at this day's open
    future = all_dates[asof_idx + 30]        # exit close, 30 sessions from asof

    aaa = prices[prices["ticker"] == "AAA"].sort_values("date")
    e = aaa[aaa["date"] == entry].iloc[0]
    # Adjusted open = open * adj_close/close (same-day factor).
    entry_px = float(e["open"]) * float(e["adj_close"]) / float(e["close"])
    future_px = float(aaa[aaa["date"] == future].iloc[0]["adj_close"])
    expected_label = (future_px / entry_px) - 1.0

    train_start = all_dates[252]
    train_end = all_dates[-(30 + 5)]

    _x, y, asof_dates = build_training_set(
        prices, ["AAA"],
        train_start=train_start,
        train_end=train_end,
        forward_horizon_days=30,
    )

    # Find the row for AAA at our chosen asof_date.
    mask = asof_dates == asof
    assert mask.any(), f"asof_date {asof} not found in result"
    label_val = float(y[mask].iloc[0])
    assert label_val == pytest.approx(expected_label, rel=1e-6)


def test_build_training_set_drops_rows_with_nan_entry_price():
    """A NaN open/adj_close at the entry session must DROP the row, not emit a
    NaN label (NaN <= 0 is False, so the old `<= 0` guards let it through and
    poisoned training)."""
    import numpy as np

    tickers = ["SPY", "AAA"]
    prices = _multi_ticker_prices(tickers, n_days=320)
    all_dates = sorted(prices["date"].unique())
    asof_idx = 260
    asof_corrupt = all_dates[asof_idx]
    entry = all_dates[asof_idx + 1]
    # Corrupt AAA's entry-session open to NaN.
    mask = (prices["ticker"] == "AAA") & (prices["date"] == entry)
    prices.loc[mask, "open"] = np.nan

    train_start = all_dates[252]
    train_end = all_dates[-(30 + 5)]
    _x, y, asof_dates = build_training_set(
        prices, ["AAA"], train_start=train_start, train_end=train_end,
        forward_horizon_days=30,
    )
    # The row whose entry is the corrupted session is dropped; no NaN labels.
    assert not (asof_dates == asof_corrupt).any()
    assert not y.isna().any()


def test_build_training_set_label_stride_subsamples_dates():
    """label_stride keeps every Nth asof_date, cutting the 29/30 label overlap
    that inflates effective sample size. stride=1 is unchanged behavior."""
    tickers = ["SPY", "AAA"]
    prices = _multi_ticker_prices(tickers, n_days=400)
    all_dates = sorted(prices["date"].unique())
    train_start = all_dates[252]
    train_end = all_dates[-(30 + 5)]

    _x1, _y1, asof1 = build_training_set(
        prices, ["AAA"], train_start=train_start, train_end=train_end,
        forward_horizon_days=30, label_stride=1,
    )
    _x10, _y10, asof10 = build_training_set(
        prices, ["AAA"], train_start=train_start, train_end=train_end,
        forward_horizon_days=30, label_stride=10,
    )

    d1 = sorted(set(asof1))
    d10 = sorted(set(asof10))
    assert 0 < len(d10) < len(d1), "stride=10 must keep fewer (but some) dates"
    # Helper uses consecutive calendar days, so 10 sessions == 10 days apart.
    gaps = [(d10[i + 1] - d10[i]).days for i in range(len(d10) - 1)]
    assert all(g >= 10 for g in gaps), f"kept dates must be >= stride apart: {gaps}"


# ---------------------------------------------------------------------------
# Test 4: universe with no data in prices -> empty DataFrames, no crash
# ---------------------------------------------------------------------------

def test_build_training_set_empty_when_universe_has_no_data():
    """Universe of unknown tickers returns empty frames, not an exception."""
    tickers = ["SPY", "AAA"]
    prices = _multi_ticker_prices(tickers, n_days=320)

    all_dates = sorted(prices["date"].unique())
    train_start = all_dates[252]
    train_end = all_dates[-(30 + 5)]

    x, y, asof_dates = build_training_set(
        prices, ["UNKNOWN1", "UNKNOWN2"],
        train_start=train_start,
        train_end=train_end,
        forward_horizon_days=30,
    )

    assert list(x.columns) == FEATURE_NAMES
    assert len(x) == 0
    assert len(y) == 0
    assert len(asof_dates) == 0


def test_build_training_set_demean_labels_removes_market_component():
    """Strategy review 2026-06-11: a RAW forward-return label makes the model
    spend capacity on the market/beta component (common to every name,
    unpredictable from per-ticker features) — in a 2023+ up-tape it learns
    'rank high-beta high', producing the chronic IT/semi concentration.
    demean_labels=True subtracts each asof date's cross-sectional mean so the
    target is pure relative (alpha) return; within-date ordering unchanged."""
    tickers = ["SPY", "AAA", "BBB", "CCC"]
    prices = _multi_ticker_prices(tickers, n_days=320)
    all_dates = sorted(prices["date"].unique())
    train_start = all_dates[252]
    train_end = all_dates[-(30 + 5)]

    _x, y_raw, asof_raw = build_training_set(
        prices, ["AAA", "BBB", "CCC"],
        train_start=train_start, train_end=train_end,
        forward_horizon_days=30,
    )
    _x, y_dm, asof_dm = build_training_set(
        prices, ["AAA", "BBB", "CCC"],
        train_start=train_start, train_end=train_end,
        forward_horizon_days=30, demean_labels=True,
    )
    assert len(y_dm) == len(y_raw)
    import numpy as np
    for d in set(asof_dm):
        m = (asof_dm == d).to_numpy()
        if m.sum() < 2:
            continue
        # per-date mean is ~0 after demeaning
        assert abs(float(y_dm[m].mean())) < 1e-9
        # within-date ORDER is preserved (pure shift)
        assert (np.argsort(y_dm[m].to_numpy()) == np.argsort(y_raw[m].to_numpy())).all()


def test_compute_latest_surprises_resolves_most_recent_clamped():
    """2026-06-12: first true fundamentals feature, unblocked by the
    yfinance_hist earnings backfill (4.8k rows, 2022+). Latest report at or
    before asof wins; surprise = (actual-est)/|est| clamped to ±1; rows
    missing either EPS leg are ignored; future reports never leak."""
    import pandas as pd

    from sma.model.loader import _compute_latest_surprises

    earnings = pd.DataFrame([
        # ticker, report_date, eps_estimate, eps_actual
        {"ticker": "AAA", "report_date": date(2025, 1, 10), "eps_estimate": 1.0, "eps_actual": 1.2},
        {"ticker": "AAA", "report_date": date(2025, 4, 10), "eps_estimate": 1.0, "eps_actual": 0.5},
        # future report (must not leak):
        {"ticker": "AAA", "report_date": date(2025, 7, 10), "eps_estimate": 1.0, "eps_actual": 9.0},
        # never reported an actual:
        {"ticker": "BBB", "report_date": date(2025, 3, 1), "eps_estimate": 2.0, "eps_actual": None},
        # +400% surprise clamps to +1:
        {"ticker": "CCC", "report_date": date(2025, 2, 1), "eps_estimate": 0.10,
         "eps_actual": 0.50},
    ])
    s = _compute_latest_surprises(earnings, date(2025, 6, 1))
    assert s["AAA"] == pytest.approx(-0.5)   # April report (latest <= asof), not January
    assert "BBB" not in s                     # never reported an actual
    assert s["CCC"] == 1.0                    # clamped at +100%
