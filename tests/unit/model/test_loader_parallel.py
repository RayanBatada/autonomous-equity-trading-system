"""build_training_set must be BIT-IDENTICAL at any feature_workers count.

The asof axis is what production fans out over — it is the coarse,
embarrassingly-parallel one (no asof reads another asof's result) and it is
where 65-80 min of every retrain goes. The guarantees pinned here:

  * X, y AND asof_dates are exactly equal to the serial path, including the
    ROW ORDER (asof-major, feats_df index order within an asof);
  * that holds with the PIT membership map applied and with demeaned labels —
    the demeaning is a cross-asof reduction and runs in the parent, so it must
    see the same assembled series either way;
  * two parallel runs agree with each other, so nothing depends on which
    worker finished first.
"""

from datetime import date, timedelta

import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal

from sma.model.loader import build_training_set

UNIVERSE = [
    "AAPL", "MSFT", "NVDA",   # Information Technology -> XLK
    "JPM", "BAC",             # Financials
    "XOM", "CVX",             # Energy
    "JNJ", "PFE",             # Health Care
    "SPY", "XLK",
]
START = date(2024, 1, 1)
N_DAYS = 400
# 30 asofs, all far enough from the end of the fixture to carry a 30-session
# forward label.
TRAIN_START = START + timedelta(days=320)
TRAIN_END = START + timedelta(days=349)


def _prices(tickers: list[str] = UNIVERSE, n_days: int = N_DAYS) -> pd.DataFrame:
    rows = []
    for t_idx, t in enumerate(tickers):
        base = 100.0 + t_idx * 7.5
        for i in range(n_days):
            c = base + i * (0.25 + 0.03 * t_idx) + ((i * (t_idx + 3)) % 11) * 0.1
            rows.append({
                "ticker": t, "date": START + timedelta(days=i),
                "open": c - 0.13, "high": c + 0.4, "low": c - 0.5,
                "close": c, "adj_close": c,
                "volume": 1_000_000 + (i * 1_000 * (t_idx + 1)) % 250_000,
            })
    return pd.DataFrame(rows)


def _politician_trades() -> pd.DataFrame:
    return pd.DataFrame([
        {"ticker": "AAPL", "filing_date": TRAIN_START + timedelta(days=3),
         "transaction_date": TRAIN_START, "transaction_type": "P",
         "amount_min": 15_000, "amount_max": 50_000},
        {"ticker": "JPM", "filing_date": TRAIN_START + timedelta(days=11),
         "transaction_date": TRAIN_START, "transaction_type": "S",
         "amount_min": 1_000, "amount_max": 15_000},
    ])


def _earnings() -> pd.DataFrame:
    return pd.DataFrame([
        {"ticker": "MSFT", "report_date": TRAIN_START + timedelta(days=12),
         "eps_estimate": 2.5, "eps_actual": 2.9},
        {"ticker": "NVDA", "report_date": START + timedelta(days=200),
         "eps_estimate": 0.5, "eps_actual": 0.4},
    ])


def _news() -> pd.DataFrame:
    rows = []
    for i in range(40):
        rows.append({"ticker": "AAPL",
                     "published_at_date": TRAIN_START + timedelta(days=i % 30)})
    rows.append({"ticker": "PFE", "published_at_date": TRAIN_START})
    return pd.DataFrame(rows)


# PFE joins late and BAC is removed mid-window, so the cross-section genuinely
# CHANGES from asof to asof — a parallel path that leaked one asof's universe
# into another would show up here.
MEMBERSHIP = {
    "PFE": (TRAIN_START + timedelta(days=10), None),
    "BAC": (None, TRAIN_START + timedelta(days=20)),
}


def _kwargs(**overrides):
    base = dict(
        prices=_prices(), universe=UNIVERSE,
        train_start=TRAIN_START, train_end=TRAIN_END,
        forward_horizon_days=30,
        politician_trades=_politician_trades(),
        earnings=_earnings(),
        news=_news(),
    )
    base.update(overrides)
    return base


def _assert_identical(a, b):
    x1, y1, d1 = a
    x2, y2, d2 = b
    assert_frame_equal(x1, x2, check_exact=True)
    assert_series_equal(y1, y2, check_exact=True)
    assert_series_equal(d1, d2, check_exact=True)


def test_training_set_bit_identical_with_membership_and_demeaned_labels():
    """The production configuration: PIT membership map + demeaned labels."""
    kw = _kwargs(membership=MEMBERSHIP, demean_labels=True)
    serial = build_training_set(**kw, feature_workers=1)
    parallel = build_training_set(**kw, feature_workers=2)

    x, _y, asofs = serial
    assert len(x) > 100, "fixture must produce a real training set"
    assert asofs.nunique() > 5, "must span several asof dates"
    # The membership map must actually bite, or this proves nothing about it.
    assert set(asofs) and len(set(x.index)) == len(x)
    _assert_identical(serial, parallel)


def test_training_set_bit_identical_with_plain_labels_and_no_membership():
    kw = _kwargs(membership=None, demean_labels=False)
    _assert_identical(
        build_training_set(**kw, feature_workers=1),
        build_training_set(**kw, feature_workers=3),
    )


def test_training_set_parallel_is_deterministic_across_runs():
    """Two independent parallel runs, bit-identical to each other."""
    kw = _kwargs(membership=MEMBERSHIP, demean_labels=True)
    _assert_identical(
        build_training_set(**kw, feature_workers=2),
        build_training_set(**kw, feature_workers=2),
    )


def test_training_set_row_order_is_asof_major_in_both_paths():
    """Row ORDER, not just row content: asof_dates must come out sorted
    ascending, which is what the walk-forward CV's positional slicing assumes."""
    kw = _kwargs(membership=MEMBERSHIP, demean_labels=True)
    _x, _y, asofs = build_training_set(**kw, feature_workers=2)
    assert list(asofs) == sorted(asofs)


def test_membership_filter_still_applies_per_asof_in_parallel():
    """BAC is removed 20 days into the window and PFE joins 10 days in, so both
    names must be present for SOME asofs and absent for others — proof that
    each worker recomputed the cross-section for ITS asof rather than
    inheriting a neighbour's."""
    kw = _kwargs(membership=MEMBERSHIP, demean_labels=False)
    with_map = build_training_set(**kw, feature_workers=2)
    kw_open = _kwargs(membership=None, demean_labels=False)
    without_map = build_training_set(**kw_open, feature_workers=2)
    assert len(with_map[0]) < len(without_map[0]), (
        "the PIT membership map must remove rows; if it does not, this file "
        "is not testing the membership path at all"
    )


def test_empty_training_set_is_identical_at_any_worker_count():
    """A window with no labelable dates returns the same empty triple either
    way (dtypes included — the trainer keys off them)."""
    kw = _kwargs(train_start=START + timedelta(days=395),
                 train_end=START + timedelta(days=399))
    _assert_identical(
        build_training_set(**kw, feature_workers=1),
        build_training_set(**kw, feature_workers=4),
    )


# ---------------------------------------------------------------------------
# The shared sort (2026-08-27). build_features groups and sorts the price frame
# for itself when it is handed one asof. A retrain hands it ~2,170, and paying
# that per date was pure repetition: the frame never changes. _TrainingInputs
# now builds ONE SortedPrices per process and every asof windows out of it.
# ---------------------------------------------------------------------------

def test_build_training_set_sorts_the_price_frame_once_for_all_asofs():
    from unittest.mock import patch

    from sma.features import builder as builder_mod
    from sma.features.window import SortedPrices
    from sma.model import loader as loader_mod

    loader_built = 0
    builder_built = 0

    def counting_loader(prices):
        nonlocal loader_built
        loader_built += 1
        return SortedPrices(prices)

    def counting_builder(prices):
        nonlocal builder_built
        builder_built += 1
        return SortedPrices(prices)

    with (
        patch.object(loader_mod, "SortedPrices", counting_loader),
        patch.object(builder_mod, "SortedPrices", counting_builder),
    ):
        x, y, asof_dates = build_training_set(
            _prices(), UNIVERSE, TRAIN_START, TRAIN_END,
            forward_horizon_days=30, feature_workers=1,
        )

    assert len(x) > 0, "fixture must produce rows"
    assert asof_dates.nunique() >= 20, "fixture must span many asof dates"
    assert loader_built == 1, (
        f"the price frame was grouped and sorted {loader_built} times for "
        f"{asof_dates.nunique()} asof dates; it must be built once per process"
    )
    assert builder_built == 0, (
        "build_features built its own index instead of using the loader's — "
        "the per-asof groupby and sort is back"
    )


def test_training_inputs_drop_the_sorted_index_from_the_pickle():
    """A worker rebuilds it locally in well under a second; shipping it would
    send a second, sorted copy of the entire price frame per task."""
    import pickle

    from sma.model.loader import _TrainingInputs

    inputs = _TrainingInputs(
        prices=_prices(), universe=UNIVERSE, membership=None,
        politician_trades=None, earnings=None, news=None, forward_horizon_days=30,
    )
    assert inputs.sorted_prices() is inputs.sorted_prices(), "must be cached"
    revived = pickle.loads(pickle.dumps(inputs, protocol=5))
    assert revived._sorted_prices is None
    assert set(revived.sorted_prices().frames()) == set(UNIVERSE)


def test_prices_by_ticker_still_serves_date_sorted_frames_for_the_labels():
    """The label lookup reads these; they must stay sorted by date whatever the
    row order of the frame handed in."""
    from pandas.testing import assert_frame_equal

    from sma.model.loader import _TrainingInputs

    shuffled = _prices().sample(frac=1.0, random_state=11)
    inputs = _TrainingInputs(
        prices=shuffled, universe=UNIVERSE, membership=None,
        politician_trades=None, earnings=None, news=None, forward_horizon_days=30,
    )
    frames = inputs.prices_by_ticker()
    assert set(frames) == set(UNIVERSE)
    for ticker, frame in frames.items():
        assert_frame_equal(
            frame,
            shuffled[shuffled["ticker"] == ticker].sort_values("date"),
            check_exact=True,
        )
