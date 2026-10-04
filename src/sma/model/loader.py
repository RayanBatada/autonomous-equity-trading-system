"""Build the (X, y, asof_dates) training dataset for the XGBoost model."""

from datetime import date, timedelta

import pandas as pd

from sma.features.builder import FEATURE_NAMES, build_features
from sma.features.parallel import map_ordered
from sma.features.window import SortedPrices


def _compute_politician_flows(
    politician_trades: pd.DataFrame, asof_date: date, lookback_days: int = 30,
) -> dict[str, float]:
    """Net dollar flow per ticker over trades DISCLOSED in [asof - lookback, asof].

    Windows on filing_date (the public DISCLOSURE date), NOT transaction_date:
    congressional trades are disclosed ~58 days (avg) after the trade, so a
    transaction_date window counted trades that weren't public yet at asof — a
    look-ahead leak AND not the live-tradable information event. Buys add the
    midpoint of the disclosed amount range; sells subtract it."""
    if politician_trades is None or politician_trades.empty:
        return {}
    start = asof_date - timedelta(days=lookback_days)
    # Normalize filing_date (DuckDB returns datetime64; a manual DataFrame may use
    # python date objects) so the window comparison is type-safe for both.
    fdates = pd.to_datetime(politician_trades["filing_date"])
    sub = politician_trades[
        (fdates >= pd.Timestamp(start))
        & (fdates <= pd.Timestamp(asof_date))
        & politician_trades["ticker"].notna()
    ]
    if sub.empty:
        return {}
    midpoint = (sub["amount_min"] + sub["amount_max"]) / 2.0
    # Sign must match predictor._fetch_politician_flows exactly: buy='P',
    # sell=any 'S%' (S, S (partial)), everything else (e.g. 'E' exchange)
    # neutral. Previously "not S" counted 'E' as a buy → train/serve skew.
    tt = sub["transaction_type"].fillna("")
    sign = pd.Series(0.0, index=sub.index)
    sign = sign.mask(tt == "P", 1.0)
    sign = sign.mask(tt.str.startswith("S"), -1.0)
    sub = sub.assign(_flow=midpoint * sign)
    return sub.groupby("ticker")["_flow"].sum().to_dict()


def _compute_news_counts_7d(
    news: pd.DataFrame, asof_date: date,
) -> dict[str, int]:
    """Per-ticker count of news rows published in [asof-7d, asof].
    Empty dict when `news` is None/empty or no rows in window."""
    if news is None or news.empty:
        return {}
    start = asof_date - timedelta(days=7)
    sub = news[
        (news["published_at_date"] >= start)
        & (news["published_at_date"] <= asof_date)
        & news["ticker"].notna()
    ]
    if sub.empty:
        return {}
    return sub.groupby("ticker").size().to_dict()


def _compute_next_earnings(
    earnings: pd.DataFrame, asof_date: date,
) -> dict[str, date]:
    """For each ticker, return the earliest report_date > asof_date. Empty
    dict when `earnings` is None/empty or no rows beat the cutoff. Used by
    the days_to_next_earnings feature; an absent ticker defaults to the
    60-day cap (≈ "no earnings soon")."""
    if earnings is None or earnings.empty:
        return {}
    sub = earnings[earnings["report_date"] > asof_date]
    if sub.empty:
        return {}
    return sub.groupby("ticker")["report_date"].min().to_dict()


def _compute_latest_surprises(
    earnings: pd.DataFrame, asof_date: date,
) -> dict[str, float]:
    """ticker → latest EPS surprise ((actual−est)/|est|, clamped ±1) for the
    most recent report at or before asof_date with BOTH legs present. Empty
    when no usable rows. Future reports never leak (report_date <= asof).
    Unblocked 2026-06-12 by the yfinance_hist earnings backfill."""
    if earnings is None or earnings.empty:
        return {}
    cols = {"eps_estimate", "eps_actual"}
    if not cols.issubset(earnings.columns):
        return {}
    sub = earnings[
        (earnings["report_date"] <= asof_date)
        & earnings["eps_actual"].notna()
        & earnings["eps_estimate"].notna()
        & (earnings["eps_estimate"].abs() > 1e-6)
    ]
    if sub.empty:
        return {}
    latest = sub.sort_values("report_date").groupby("ticker").tail(1)
    out: dict[str, float] = {}
    for _, r in latest.iterrows():
        surprise = (float(r["eps_actual"]) - float(r["eps_estimate"])) / abs(
            float(r["eps_estimate"])
        )
        out[str(r["ticker"])] = max(-1.0, min(1.0, surprise))
    return out


class _TrainingInputs:
    """The read-only inputs every asof's rows are computed from.

    Shipped to each worker process ONCE, through the pool initializer, rather
    than attached to each of the ~2,170 per-asof tasks a production retrain
    submits. Everything on it is treated as immutable for the duration of the
    build.

    The SortedPrices index is DERIVED from `prices` and is deliberately dropped
    from the pickle (see __getstate__): shipping it would send a second, sorted
    copy of the entire price frame — double the IPC and double the worker RSS —
    when a worker can rebuild it locally once, in well under a second, and then
    reuse it for every asof it handles.
    """

    __slots__ = (
        "prices", "universe", "membership", "politician_trades", "earnings",
        "news", "forward_horizon_days", "_sorted_prices",
    )

    def __init__(
        self,
        prices: pd.DataFrame,
        universe: list[str],
        membership: dict[str, tuple[date | None, date | None]] | None,
        politician_trades: pd.DataFrame | None,
        earnings: pd.DataFrame | None,
        news: pd.DataFrame | None,
        forward_horizon_days: int,
    ) -> None:
        self.prices = prices
        self.universe = universe
        self.membership = membership
        self.politician_trades = politician_trades
        self.earnings = earnings
        self.news = news
        self.forward_horizon_days = forward_horizon_days
        self._sorted_prices: SortedPrices | None = None

    def sorted_prices(self) -> SortedPrices:
        """The per-ticker groupby-and-sort of `prices`, done ONCE per process.

        Serves both halves of an asof's work. The LABELS need each ticker's
        rows sorted by date; the FEATURES need them truncated at asof_date, and
        with the frame pre-sorted that truncation is a searchsorted prefix
        slice rather than a mask plus a sort — repeated for every one of the
        ~2,170 asof dates a retrain builds.

        Cached on the instance and rebuilt per worker (see __getstate__)."""
        if self._sorted_prices is None:
            self._sorted_prices = SortedPrices(self.prices)
        return self._sorted_prices

    def prices_by_ticker(self) -> dict[str, pd.DataFrame]:
        """Per-ticker price frames, sorted once. This sort used to sit INSIDE
        the (asof x ticker) double loop as
            prices[prices["ticker"] == ticker].sort_values("date")
        so a full-frame boolean scan and a sort ran once per label — the same
        quadratic the feature builder carried, and it grows with both the date
        range and the universe (1ee39b1). groupby preserves each group's
        original row order, so this sorts exactly the frame the mask used to
        produce and every label is bit-identical."""
        return self.sorted_prices().frames()

    def __getstate__(self) -> dict:
        return {
            name: (None if name == "_sorted_prices" else getattr(self, name))
            for name in self.__slots__
        }

    def __setstate__(self, state: dict) -> None:
        for name, value in state.items():
            setattr(self, name, value)


def _rows_for_asof(
    unit: tuple[date, date, date], inputs: _TrainingInputs,
) -> tuple[list[dict], list[float]]:
    """Every (feature row, label) this one asof_date contributes.

    The whole per-asof body of build_training_set, lifted out verbatim so the
    serial path and each parallel worker run the SAME code on different dates.
    Depends on nothing but `unit` and the read-only `inputs`, which is what
    makes the asof axis embarrassingly parallel: no asof reads another asof's
    result, and the only cross-asof step (label demeaning) happens once, in the
    parent, after everything is assembled.

    `unit` is (asof, entry_date, future_date) — the two label dates are
    resolved by the parent from the full trading calendar, which the worker
    therefore never needs a copy of.

    Must stay a module-level function: `spawn` pickles the task by qualified
    name, so a closure or a local def would not survive the trip to a worker.
    """
    asof, entry_date, future_date = unit

    # PIT membership: former members only contribute while a member —
    # [added, removed), added inclusive, removed exclusive. NOTE: for a
    # true delisting, asofs within forward_horizon_days of the ticker's
    # LAST price row drop out (no forward label), so the terminal collapse
    # is only partially captured — the last labeled row sits ~a horizon
    # before the end of its data. Acceptable for research; know it when
    # interpreting decliner results (review 2026-07-20).
    membership = inputs.membership
    if membership:
        universe_at_asof = [
            t for t in inputs.universe
            if t not in membership
            or ((membership[t][0] is None or membership[t][0] <= asof)
                and (membership[t][1] is None or asof < membership[t][1]))
        ]
    else:
        universe_at_asof = inputs.universe

    # Compute features for the universe at asof, including the
    # politician_flow_30d feature for asof_dates where we have PTR data,
    # days_to_next_earnings where we have earnings calendar coverage,
    # and news_count_7d_log where we have news data.
    flows = _compute_politician_flows(inputs.politician_trades, asof) \
        if inputs.politician_trades is not None else None
    earnings_cal = _compute_next_earnings(inputs.earnings, asof) \
        if inputs.earnings is not None else None
    earnings_surprises = _compute_latest_surprises(inputs.earnings, asof)
    news_counts = _compute_news_counts_7d(inputs.news, asof) \
        if inputs.news is not None else None
    # workers=1: this call may already BE running inside a worker, and the asof
    # axis is the coarser, cheaper-to-dispatch one. Never nest the two pools.
    feats_df = build_features(
        inputs.prices, universe_at_asof, asof,
        politician_flows=flows,
        earnings_calendar=earnings_cal,
        earnings_surprises=earnings_surprises,
        news_counts_7d=news_counts,
        workers=1,
        # Built from inputs.prices ONCE per process and reused by every asof
        # this worker handles — otherwise build_features re-groups and re-sorts
        # the whole price frame on each of the ~2,170 dates a retrain covers.
        sorted_prices=inputs.sorted_prices(),
    )
    if feats_df.empty:
        return [], []

    prices_by_ticker = inputs.prices_by_ticker()
    feature_rows: list[dict] = []
    labels: list[float] = []
    for ticker in feats_df.index:
        # Compute label for this ticker: forward return from the next
        # session's OPEN (executable entry) to the close N sessions out.
        t_prices = prices_by_ticker.get(ticker)
        if t_prices is None:
            continue  # no rows at all — the old empty mask fell through
        entry_row = t_prices[t_prices["date"] == entry_date]
        future_row = t_prices[t_prices["date"] == future_date]
        if entry_row.empty or future_row.empty:
            continue
        er = entry_row.iloc[0]
        o, c, ac = float(er["open"]), float(er["close"]), float(er["adj_close"])
        # Positive checks (not `<= 0`) so NaN is rejected too: `NaN <= 0` is
        # False, which would otherwise pass a NaN open/adj_close through to a
        # NaN label and poison training.
        if not (o > 0 and c > 0):
            continue
        # Adjusted open = raw open scaled by the same-day adj_close/close
        # factor, so entry and exit share one split/dividend basis.
        entry_px = o * ac / c
        future_px = float(future_row.iloc[0]["adj_close"])
        if not (entry_px > 0 and future_px > 0):
            continue
        label = (future_px / entry_px) - 1.0

        feature_rows.append({col: feats_df.loc[ticker, col] for col in FEATURE_NAMES})
        labels.append(label)
    return feature_rows, labels


def build_training_set(
    prices: pd.DataFrame,
    universe: list[str],
    train_start: date,
    train_end: date,
    forward_horizon_days: int = 30,
    label_stride: int = 1,
    demean_labels: bool = False,
    politician_trades: pd.DataFrame | None = None,
    earnings: pd.DataFrame | None = None,
    news: pd.DataFrame | None = None,
    membership: dict[str, tuple[date | None, date | None]] | None = None,
    feature_workers: int | None = 1,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Build (X, y, asof_dates) for training.

    `membership` (optional): {ticker: (added, removed)} point-in-time intervals
    for FORMER universe members. A listed ticker contributes rows only while a
    member — asof in [added, removed) with a None bound unbounded; unlisted
    tickers are unrestricted (active names). This is the survivorship fix
    (2026-07-20): delisted/removed names join the training cross-section for
    exactly the period they were members, so the losers today's universe
    survived are learned from too. The demeaned label's cross-sectional mean is
    computed on the membership-filtered set, as it should be. None/{} =
    original behavior.

    `feature_workers` (optional): processes to spread the per-asof work over.
    This is THE cost centre — 65-80 min of every retrain and the dominant cost
    of every walk-forward study — and the asof axis is embarrassingly parallel,
    so this is where the cores go. 1 (the default, so every existing caller is
    untouched) is the exact serial path and builds no pool; None means auto,
    `min(4, cpu_count - 1)`. Production supplies it from `model.feature_workers`
    in config.yaml. Output is BIT-IDENTICAL at any worker count: each asof is
    computed from read-only inputs, results are collected in submission order
    (not completion order), and the one cross-asof step — label demeaning —
    runs here in the parent after everything is assembled.

    For each (ticker, asof_date) in [train_start, train_end]:
      - Compute the 12 features at asof_date.
      - Compute the label as the EXECUTABLE forward return: from the next
        session's split-adjusted open (where a DAY_OPG buy fills) to the
        adj_close forward_horizon_days sessions after asof —
        (adj_close[asof + N tr] / adj_open[asof + 1 tr] - 1). Using close[asof]
        as the entry would credit the untradable close[asof]->open[asof+1] gap.
      - Skip the row if any feature is None (insufficient history) OR if the
        label cannot be computed (insufficient forward data).

    The forward-label requirement means rows with asof_date in the last
    forward_horizon_days trading days of the price data are excluded
    automatically (no label exists yet).

    Returns:
        X: DataFrame with feature columns, no index (just rows).
        y: Series of float labels, parallel to X.
        asof_dates: Series of date values, parallel to X. Used by walk-forward
            CV to slice train/val correctly.
    """
    # Build the set of all unique trading dates in the prices data.
    all_dates = sorted(prices["date"].unique())

    # Filter to dates inside [train_start, train_end].
    in_window = [d for d in all_dates if train_start <= d <= train_end]

    # Overlapping-label mitigation: consecutive asof_dates produce labels that
    # overlap by (forward_horizon_days - 1) sessions, so daily rows are far from
    # independent (effective N ~ N / horizon) and inflate CV optimism. Keeping
    # every Nth date thins that overlap. stride=1 is the original behavior.
    # NOTE: at stride>1 the walk-forward CV purge (trainer.py) counts purge_days
    # in *positions* of this now-strided date list, so the gap grows to
    # ~purge_days*stride sessions — safe (over-purges, no leakage) but wasteful;
    # scale purge_days down by ~stride before engaging striding in production.
    if label_stride > 1:
        in_window = in_window[::label_stride]

    # Resolve each asof's two label dates HERE, against the full trading
    # calendar, so a worker never needs a copy of it. The label runs from the
    # EXECUTABLE entry — the next session's open, where a DAY_OPG buy actually
    # fills — to the close N sessions out. An asof whose exit close is past the
    # end of the data is dropped, which is what excludes the last
    # forward_horizon_days of the range.
    date_index = {d: i for i, d in enumerate(all_dates)}
    units: list[tuple[date, date, date]] = []
    for asof in in_window:
        asof_idx = date_index.get(asof)
        if asof_idx is None:
            continue
        entry_idx = asof_idx + 1
        future_idx = asof_idx + forward_horizon_days
        if future_idx >= len(all_dates):
            continue  # need the exit close; entry_idx < future_idx is implied
        units.append((asof, all_dates[entry_idx], all_dates[future_idx]))

    inputs = _TrainingInputs(
        prices=prices,
        universe=universe,
        membership=membership,
        politician_trades=politician_trades,
        earnings=earnings,
        news=news,
        forward_horizon_days=forward_horizon_days,
    )

    feature_rows: list[dict] = []
    labels: list[float] = []
    asofs: list[date] = []
    # map_ordered returns results in UNIT order regardless of which worker
    # finished first, so extending in this loop reproduces the serial path's
    # row order exactly — asof-major, then feats_df index order within an asof.
    per_asof = map_ordered(_rows_for_asof, units, inputs, workers=feature_workers)
    for (asof, _entry, _future), (rows_for_asof, labels_for_asof) in zip(
        units, per_asof, strict=True,
    ):
        feature_rows.extend(rows_for_asof)
        labels.extend(labels_for_asof)
        asofs.extend([asof] * len(rows_for_asof))

    if not feature_rows:
        empty_x = pd.DataFrame(columns=FEATURE_NAMES)
        return (
            empty_x,
            pd.Series(dtype=float, name="ret_30d_forward"),
            pd.Series(dtype="object", name="asof_date"),
        )

    x = pd.DataFrame(feature_rows)[FEATURE_NAMES]
    y = pd.Series(labels, name="ret_30d_forward")
    asof_dates = pd.Series(asofs, name="asof_date")
    if demean_labels:
        # Cross-sectional demeaning (strategy review 2026-06-11): subtract each
        # asof date's mean label so the target is RELATIVE return. A raw label
        # makes the model fit the market/beta component — common to every name
        # and unpredictable from per-ticker features — so in a one-direction
        # training regime it learns 'rank high-beta high' (the chronic IT/semi
        # concentration). Within-date ordering is unchanged; the pooled
        # regression target changes materially.
        y = y - y.groupby(asof_dates.to_numpy()).transform("mean")
        y.name = "ret_30d_forward"
    return x, y, asof_dates
