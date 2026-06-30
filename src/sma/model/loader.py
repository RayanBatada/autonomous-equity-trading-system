"""Build the (X, y, asof_dates) training dataset for the XGBoost model."""

from datetime import date, timedelta

import pandas as pd

from sma.features.builder import FEATURE_NAMES, build_features


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
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Build (X, y, asof_dates) for training.

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

    feature_rows: list[dict] = []
    labels: list[float] = []
    asofs: list[date] = []

    for asof in in_window:
        # The label runs from the EXECUTABLE entry — the next session's open,
        # where a DAY_OPG buy actually fills — to the close N sessions out.
        try:
            asof_idx = all_dates.index(asof)
        except ValueError:
            continue
        entry_idx = asof_idx + 1
        future_idx = asof_idx + forward_horizon_days
        if future_idx >= len(all_dates):
            continue  # need the exit close; entry_idx < future_idx is implied
        entry_date = all_dates[entry_idx]
        future_date = all_dates[future_idx]

        # Compute features for the universe at asof, including the
        # politician_flow_30d feature for asof_dates where we have PTR data,
        # days_to_next_earnings where we have earnings calendar coverage,
        # and news_count_7d_log where we have news data.
        flows = _compute_politician_flows(politician_trades, asof) \
            if politician_trades is not None else None
        earnings_cal = _compute_next_earnings(earnings, asof) \
            if earnings is not None else None
        earnings_surprises = _compute_latest_surprises(earnings, asof)
        news_counts = _compute_news_counts_7d(news, asof) \
            if news is not None else None
        feats_df = build_features(
            prices, universe, asof,
            politician_flows=flows,
            earnings_calendar=earnings_cal,
            earnings_surprises=earnings_surprises,
            news_counts_7d=news_counts,
        )
        if feats_df.empty:
            continue

        for ticker in feats_df.index:
            # Compute label for this ticker: forward return from the next
            # session's OPEN (executable entry) to the close N sessions out.
            t_prices = prices[prices["ticker"] == ticker].sort_values("date")
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

            row = {col: feats_df.loc[ticker, col] for col in FEATURE_NAMES}
            feature_rows.append(row)
            labels.append(label)
            asofs.append(asof)

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
