"""PriceWindow / SortedPrices: the truncation must be IDENTICAL, not similar.

Every feature function's visible history is `truncate_and_sort` — one boolean
mask plus one sort. This module caches that per (ticker, asof) and, where the
ticker's dates are distinct and non-null, replaces it with a searchsorted
prefix slice. These tests pin the equivalence at the frame level
(assert_frame_equal(check_exact=True), including the index and the row order),
because a feature value computed off a subtly different slice would be wrong in
a way nothing downstream could notice.
"""

from datetime import date, timedelta

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from sma.features import technical
from sma.features.window import (
    PriceWindow,
    SortedPrices,
    truncate_and_sort,
    window_for,
)

START = date(2024, 1, 1)


def _prices(tickers, n_days, start=START):
    """Multi-ticker prices with a distinct level per ticker, ascending dates."""
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


def _naive(prices, ticker, asof):
    """What a feature function did for itself before any of this existed."""
    return truncate_and_sort(prices[prices["ticker"] == ticker], asof)


# ---------------------------------------------------------------------------
# The equivalence
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("offset", [-5, 0, 1, 29, 59, 60, 61, 200])
def test_window_is_the_naive_truncation_bit_for_bit(offset):
    """Every asof of interest: before the first row, on a row, past the last."""
    prices = _prices(["AAA", "BBB", "CCC"], n_days=60)
    asof = START + timedelta(days=offset)
    index = SortedPrices(prices)
    for ticker in ("AAA", "BBB", "CCC"):
        got = index.window(ticker, asof)
        assert_frame_equal(got.rows, _naive(prices, ticker, asof), check_exact=True)
        assert list(got.rows.index) == list(_naive(prices, ticker, asof).index)


def test_window_matches_the_naive_truncation_on_an_unsorted_frame():
    """The index must sort, not assume sorted input — the raw path did."""
    prices = _prices(["AAA", "BBB"], n_days=40).sample(frac=1.0, random_state=7)
    index = SortedPrices(prices)
    asof = START + timedelta(days=25)
    for ticker in ("AAA", "BBB"):
        assert_frame_equal(
            index.window(ticker, asof).rows,
            _naive(prices, ticker, asof),
            check_exact=True,
        )


def test_window_carries_its_asof_date():
    index = SortedPrices(_prices(["AAA"], n_days=10))
    asof = START + timedelta(days=4)
    assert index.window("AAA", asof).asof_date == asof


def test_absent_ticker_has_no_window():
    """None, not an empty window: build_features drops such a ticker outright."""
    index = SortedPrices(_prices(["AAA"], n_days=10))
    assert index.window("ZZZZ", START) is None
    assert "ZZZZ" not in index
    assert "AAA" in index


def test_empty_window_keeps_the_frame_s_columns_and_dtypes():
    """Stands in for an absent benchmark ticker (no SPY rows in the frame)."""
    prices = _prices(["AAA"], n_days=10)
    w = SortedPrices(prices).empty_window(START)
    assert len(w.rows) == 0
    assert list(w.rows.columns) == list(prices.columns)
    assert w.rows.dtypes.to_dict() == prices.dtypes.to_dict()


def test_index_over_an_empty_frame_yields_no_windows():
    prices = _prices(["AAA"], n_days=3).iloc[:0]
    index = SortedPrices(prices)
    assert index.window("AAA", START) is None
    assert len(index.empty_window(START).rows) == 0


def test_frames_are_date_sorted_per_ticker():
    """The label lookup in loader._rows_for_asof reads these."""
    prices = _prices(["AAA", "BBB"], n_days=20).sample(frac=1.0, random_state=3)
    frames = SortedPrices(prices).frames()
    assert set(frames) == {"AAA", "BBB"}
    for ticker, frame in frames.items():
        assert_frame_equal(
            frame,
            prices[prices["ticker"] == ticker].sort_values("date"),
            check_exact=True,
        )


# ---------------------------------------------------------------------------
# The slow path: the searchsorted prefix is only the sorted mask when the
# ticker's dates are distinct and non-null.
# ---------------------------------------------------------------------------

def test_duplicate_dates_fall_back_to_mask_and_sort():
    """sort_values is quicksort — not stable — so tied dates may order one way
    in the whole frame and another in the masked subset. Such a ticker keeps
    the slow path rather than risking a different slice."""
    prices = _prices(["AAA", "BBB"], n_days=20)
    dup = prices[prices["ticker"] == "AAA"].iloc[[7]].assign(close=999.0)
    prices = pd.concat([prices, dup], ignore_index=True)

    index = SortedPrices(prices)
    assert index.slow_path_tickers == ["AAA"]
    asof = START + timedelta(days=15)
    assert_frame_equal(
        index.window("AAA", asof).rows, _naive(prices, "AAA", asof), check_exact=True,
    )
    # BBB is unaffected and keeps the fast path.
    assert_frame_equal(
        index.window("BBB", asof).rows, _naive(prices, "BBB", asof), check_exact=True,
    )


def test_null_dates_fall_back_to_mask_and_sort():
    """A null date sorts to the END (na_position="last") but the mask DROPS it,
    so a prefix length would be wrong."""
    prices = _prices(["AAA"], n_days=20)
    prices.loc[3, "date"] = None
    index = SortedPrices(prices)
    assert index.slow_path_tickers == ["AAA"]
    asof = START + timedelta(days=15)
    assert_frame_equal(
        index.window("AAA", asof).rows, _naive(prices, "AAA", asof), check_exact=True,
    )


# ---------------------------------------------------------------------------
# _rows_up_to: the one place a window and a frame are told apart
# ---------------------------------------------------------------------------

def test_rows_up_to_returns_a_window_s_rows_untouched():
    prices = _prices(["AAA"], n_days=30)
    asof = START + timedelta(days=17)
    w = window_for(prices, asof)
    assert technical._rows_up_to(w, asof) is w.rows


def test_rows_up_to_rejects_a_window_built_for_another_date():
    """A stale window returns a plausible number, not an error, so it has to be
    caught here or not at all."""
    prices = _prices(["AAA"], n_days=30)
    w = window_for(prices, START + timedelta(days=10))
    with pytest.raises(ValueError, match="built for 2024-01-11 but used at"):
        technical._rows_up_to(w, START + timedelta(days=20))


def test_rows_up_to_still_takes_a_raw_frame():
    prices = _prices(["AAA"], n_days=30)
    asof = START + timedelta(days=12)
    assert_frame_equal(
        technical._rows_up_to(prices, asof),
        truncate_and_sort(prices, asof),
        check_exact=True,
    )


# ---------------------------------------------------------------------------
# Every feature function must agree frame-form vs window-form
# ---------------------------------------------------------------------------

_ONE_FRAME_FEATURES = [
    technical.ret_1d, technical.ret_5d, technical.ret_20d, technical.ret_60d,
    technical.vol_20d, technical.vol_60d, technical.rsi_14,
    technical.volume_z_20d, technical.dollar_volume_20d, technical.gap_open,
    technical.dist_from_52w_high, technical.downside_vol_ratio_60d,
    technical.reversal_5d_z,
]


@pytest.mark.parametrize("fn", _ONE_FRAME_FEATURES, ids=lambda f: f.__name__)
def test_single_frame_feature_agrees_with_its_window(fn):
    prices = _prices(["AAA"], n_days=300)
    asof = START + timedelta(days=280)
    frame_value = fn(prices, asof)
    assert frame_value is not None, "fixture must give the feature enough history"
    assert fn(window_for(prices, asof), asof) == frame_value


def test_two_frame_features_agree_with_their_windows():
    prices = _prices(["AAA", "SPY", "XLK"], n_days=300)
    asof = START + timedelta(days=280)
    tgt = prices[prices["ticker"] == "AAA"]
    spy = prices[prices["ticker"] == "SPY"]
    etf = prices[prices["ticker"] == "XLK"]
    wt, ws, we = (window_for(f, asof) for f in (tgt, spy, etf))

    assert technical.rel_strength_spy_60d(wt, ws, asof) == \
        technical.rel_strength_spy_60d(tgt, spy, asof)
    assert technical.vol_adj_mom_60d(wt, ws, asof) == \
        technical.vol_adj_mom_60d(tgt, spy, asof)
    assert technical.rel_strength_sector_etf_30d(wt, we, asof) == \
        technical.rel_strength_sector_etf_30d(tgt, etf, asof)
    assert technical.rel_strength_sector_30d(wt, [ws, we], asof) == \
        technical.rel_strength_sector_30d(tgt, [spy, etf], asof)


def test_window_preserves_the_lookahead_contract():
    """Rows after asof_date must not reach a feature — the contract the raw
    mask enforced, now enforced once at window construction."""
    full = _prices(["AAA"], n_days=300)
    asof = START + timedelta(days=200)
    truncated = full[full["date"] <= asof]
    assert technical.rsi_14(window_for(full, asof), asof) == \
        technical.rsi_14(window_for(truncated, asof), asof)


def test_price_window_survives_pickling():
    """The parallel feature build ships a few hundred of these per worker."""
    import pickle
    w = window_for(_prices(["AAA"], n_days=30), START + timedelta(days=12))
    back = pickle.loads(pickle.dumps(w, protocol=5))
    assert isinstance(back, PriceWindow)
    assert back.asof_date == w.asof_date
    assert_frame_equal(back.rows, w.rows, check_exact=True)
