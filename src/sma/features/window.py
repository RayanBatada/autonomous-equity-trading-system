"""Truncated, date-sorted price views, computed ONCE per (ticker, asof_date).

Every one of the ~22 per-ticker feature functions in `technical` opens with the
same line — `prices[prices["date"] <= asof].sort_values("date")` — over the
ticker's whole price history. So the same boolean mask and the same sort ran
once PER FEATURE, per ticker, per asof date. cProfile on a 266-name x 2y slice
(2026-08-27, 40 asofs): 226,352 calls, 67.0s of a 101.4s feature build, for
about 11,000 distinct answers.

Two objects here, one per axis of that redundancy:

  PriceWindow  — the truncated, sorted rows for ONE (ticker, asof). Every
                 feature function accepts it wherever it accepts a raw price
                 frame; `technical._rows_up_to` is the single place that tells
                 the two apart. Kills the PER-FEATURE repeat (~21x).

  SortedPrices — every ticker's frame date-sorted ONCE, so a window is a
                 `searchsorted` prefix slice rather than a mask plus a sort.
                 Kills the PER-ASOF repeat, which is what a retrain pays: the
                 same ~2,100-row history re-masked and re-sorted for each of
                 ~2,170 asof dates. `build_features` builds one per call by
                 default; `build_training_set` builds one per worker and hands
                 it to every asof.

Neither changes what a feature SEES. `truncate_and_sort` below is the single
definition of the truncation, `_rows_up_to` still calls it for a raw frame, and
the fast path is only taken where it is provably the same frame — see the
distinct-dates condition on SortedPrices.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd


def truncate_and_sort(prices: pd.DataFrame, asof_date: date) -> pd.DataFrame:
    """Rows with date <= asof_date, sorted ascending by date.

    THE definition of a feature function's visible history. Both
    `technical._rows_up_to` (raw frames) and `SortedPrices`' slow path call it,
    so there is exactly one copy of this expression in the codebase.
    """
    return prices[prices["date"] <= asof_date].sort_values("date")


class PriceWindow:
    """One ticker's visible rows at one asof_date: `truncate_and_sort`, cached.

    Passed to the feature functions in place of the raw price frame. It is
    deliberately dumb — no lazy computation, no dict — because it is built once
    per (ticker, asof) and then read ~22 times, and because the parallel feature
    build pickles a few hundred of them to each worker.

    `asof_date` rides along so `_rows_up_to` can refuse a window built for a
    different date rather than silently returning the wrong history: a window is
    the one input whose staleness would look like a plausible feature value.
    """

    __slots__ = ("asof_date", "rows")

    def __init__(self, rows: pd.DataFrame, asof_date: date) -> None:
        self.rows = rows
        self.asof_date = asof_date

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"PriceWindow(asof_date={self.asof_date!r}, rows={len(self.rows)})"


def window_for(prices: pd.DataFrame, asof_date: date) -> PriceWindow:
    """Build a window from a raw single-ticker frame, the slow way.

    The unconditional path: one mask plus one sort, exactly what a feature
    function would have done for itself. `SortedPrices.window` is the fast
    equivalent when the same frame is windowed at many dates.
    """
    return PriceWindow(truncate_and_sort(prices, asof_date), asof_date)


class SortedPrices:
    """Per-ticker price frames, date-sorted once; windows by binary search.

    Built from the same multi-ticker frame `build_features` takes (columns
    ticker, date, open, high, low, close, adj_close, volume). One `groupby` and
    one sort per ticker up front, and thereafter `window(ticker, asof)` is a
    `searchsorted` plus an `iloc` prefix slice — O(log rows) instead of an
    O(rows) boolean mask plus an O(rows log rows) sort.

    WHY THE PREFIX IS THE SAME FRAME, not merely an equal one: in a frame
    sorted ascending by date, the rows with date <= asof are exactly the first
    k, and their order is the sorted order — SO LONG AS the ticker's dates are
    distinct and non-null. With a duplicate date the two paths can genuinely
    disagree: `sort_values` defaults to quicksort, which is not stable, so tied
    rows may land in one order when the whole frame is sorted and another when
    the masked subset is. Nulls are worse — `na_position="last"` parks them
    after the prefix, but the mask drops them, so k would be wrong.

    Both conditions are checked ONCE per ticker at construction. Any ticker that
    fails keeps the slow path (`truncate_and_sort` on the original group) and is
    listed in `slow_path_tickers`. Production prices cannot fail it — the
    training/predict SELECT dedupes on (ticker, date) with ROW_NUMBER and
    requires a non-null date — but a hand-built frame can, and it must stay
    correct rather than fast.
    """

    __slots__ = ("_dates", "_empty", "_slow", "_sorted")

    def __init__(self, prices: pd.DataFrame) -> None:
        self._sorted: dict[str, pd.DataFrame] = {}
        self._dates: dict[str, np.ndarray] = {}
        self._slow: dict[str, pd.DataFrame] = {}
        # Reproduces the old `prices.iloc[:0]` stand-in for an absent benchmark
        # ticker: same columns, same dtypes, no rows.
        self._empty = prices.iloc[:0]
        if prices.empty:
            return
        for ticker, group in prices.groupby("ticker", sort=False):
            # groupby preserves each group's original row order, so this sorts
            # exactly the frame a `prices["ticker"] == t` mask would produce.
            srt = group.sort_values("date")
            self._sorted[ticker] = srt
            dates = srt["date"]
            if dates.isna().any() or dates.duplicated().any():
                self._slow[ticker] = group
            else:
                self._dates[ticker] = dates.to_numpy()

    def __contains__(self, ticker: str) -> bool:
        return ticker in self._sorted

    @property
    def slow_path_tickers(self) -> list[str]:
        """Tickers whose dates are duplicated or null, kept on mask-and-sort."""
        return list(self._slow)

    def window(self, ticker: str, asof_date: date) -> PriceWindow | None:
        """The (ticker, asof_date) window, or None if the ticker has no rows.

        None — rather than an empty window — because that is the distinction
        `build_features` already draws: a ticker absent from the price frame is
        dropped before any feature runs.
        """
        srt = self._sorted.get(ticker)
        if srt is None:
            return None
        group = self._slow.get(ticker)
        if group is not None:
            return window_for(group, asof_date)
        k = int(np.searchsorted(self._dates[ticker], asof_date, side="right"))
        return PriceWindow(srt.iloc[:k], asof_date)

    def empty_window(self, asof_date: date) -> PriceWindow:
        """A window over no rows — what an absent benchmark ticker resolves to."""
        return window_for(self._empty, asof_date)

    def frames(self) -> dict[str, pd.DataFrame]:
        """Each ticker's rows, date-sorted, keyed by ticker.

        The same mapping `loader._TrainingInputs.prices_by_ticker()` used to
        build for itself, so the label lookup shares this one sort instead of
        paying a second one.
        """
        return self._sorted
