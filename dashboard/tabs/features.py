"""Features tab — diagnostic view of the 17-feature model surface.

Shows what the model actually sees for any asof_date:

1. Per-feature distribution (histogram) across the current universe
2. Per-feature summary stats (mean, std, range, % default)
3. Correlation matrix between all 17 features (heatmap)
4. Top/bottom 10 by any feature
5. Single-ticker feature row inspector (what does the model see for NVDA today?)

This is the dashboard answer to "is this feature actually carrying signal,
or just emitting its default value?" — a feature whose distribution is
99% one value (e.g. days_to_next_earnings stuck at the 60d cap because
the earnings calendar is sparse) is silently degrading the model.

Data source: features are RECOMPUTED on-the-fly using
sma.features.builder.build_features against the live prices + earnings +
news + politician tables. No precomputed feature_store table yet.
"""

from __future__ import annotations

from datetime import date as date_cls
from datetime import timedelta

import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.data import DB_PATH
from sma.db_connect import read_only_connect
from sma.features.builder import FEATURE_NAMES, build_features
from sma.ingest.universe import load_universe
from sma.model.loader import (
    _compute_news_counts_7d,
    _compute_next_earnings,
    _compute_politician_flows,
)

# Columns here MUST cover every field `_compute_politician_flows` reads (it
# windows on filing_date as of 077e1e9). Keep this query and the model loader's
# in lockstep — a missing column silently crashes the Features tab.
_POLITICIAN_TRADES_SQL = (
    "SELECT ticker, transaction_date, filing_date, transaction_type, "
    "amount_min, amount_max FROM politician_trades "
    "WHERE ticker IS NOT NULL"
)


def render() -> None:
    st.header("Features")
    st.caption(
        f"Live diagnostic for the **{len(FEATURE_NAMES)}-feature** model "
        "surface. Features are recomputed on-the-fly against current data; "
        "this is what the next predict.daily would see for the selected asof."
    )

    if not DB_PATH.exists():
        st.error(f"DuckDB not found at {DB_PATH}.")
        return

    asof = st.date_input(
        "asof_date",
        value=date_cls.today(),
        help="Compute features as of this date. The model retrain target "
        "is 30-day forward return, so pick something at least 30 trading "
        "days back for label-aware analysis.",
    )

    universe = load_universe(__import__("pathlib").Path("src/sma/universe.yaml"))
    feats_df = _build_features_cached(asof, tuple(universe))

    if feats_df.empty:
        st.warning(
            "No tickers resolved features for this asof. Common cause: asof is "
            "too early (need ~252 trading days of history for "
            "dist_from_52w_high)."
        )
        return

    st.success(
        f"Features computed for **{len(feats_df)} / {len(universe)} tickers** "
        f"at asof={asof.isoformat()}."
    )

    _render_summary_stats(feats_df)
    st.divider()
    _render_distributions(feats_df)
    st.divider()
    _render_top_bottom_by_feature(feats_df)
    st.divider()
    _render_correlation_matrix(feats_df)
    st.divider()
    _render_ticker_inspector(feats_df)


@st.cache_data(ttl=300, show_spinner="Computing features...")
def _build_features_cached(asof: date_cls, universe: tuple[str, ...]) -> pd.DataFrame:
    """Cache feature computation for 5 minutes. Tuple-ify universe so it's
    hashable for the cache key."""
    universe_list = list(universe)

    # Load 400-day price window (need ~252 trading days for the longest feature).
    start_date = asof - timedelta(days=400)
    con = read_only_connect(DB_PATH)
    try:
        prices = con.execute(
            """
            SELECT ticker, date, open, high, low, close, adj_close, volume
            FROM (
                SELECT *,
                       ROW_NUMBER() OVER (
                           PARTITION BY ticker, date
                           ORDER BY CASE source WHEN 'yfinance' THEN 0
                                                 WHEN 'alpaca' THEN 1
                                                 ELSE 2 END
                       ) AS rn
                FROM prices
                WHERE ticker = ANY($tickers)
                  AND date BETWEEN $start AND $end
                  AND adj_close IS NOT NULL
            ) t
            WHERE rn = 1
            ORDER BY ticker, date
            """,
            {"tickers": universe_list, "start": start_date, "end": asof},
        ).df()
        prices["date"] = pd.to_datetime(prices["date"]).dt.date

        # Side tables for the side-data features.
        try:
            pol_df = con.execute(_POLITICIAN_TRADES_SQL).df()
            if not pol_df.empty:
                pol_df["transaction_date"] = pd.to_datetime(
                    pol_df["transaction_date"],
                ).dt.date
        except Exception:
            pol_df = pd.DataFrame()

        try:
            earn_df = con.execute(
                "SELECT DISTINCT ticker, report_date FROM earnings "
                "WHERE ticker IS NOT NULL AND report_date IS NOT NULL"
            ).df()
            if not earn_df.empty:
                earn_df["report_date"] = pd.to_datetime(earn_df["report_date"]).dt.date
        except Exception:
            earn_df = pd.DataFrame()

        try:
            news_df = con.execute(
                "SELECT ticker, CAST(published_at AS DATE) AS published_at_date "
                "FROM news WHERE ticker IS NOT NULL AND published_at IS NOT NULL"
            ).df()
            if not news_df.empty:
                news_df["published_at_date"] = pd.to_datetime(
                    news_df["published_at_date"],
                ).dt.date
        except Exception:
            news_df = pd.DataFrame()
    finally:
        con.close()

    flows = _compute_politician_flows(pol_df, asof) if not pol_df.empty else None
    earnings_cal = _compute_next_earnings(earn_df, asof) if not earn_df.empty else None
    news_counts = _compute_news_counts_7d(news_df, asof) if not news_df.empty else None

    return build_features(
        prices, universe_list, asof,
        politician_flows=flows,
        earnings_calendar=earnings_cal,
        news_counts_7d=news_counts,
    )


def _render_summary_stats(feats_df: pd.DataFrame) -> None:
    st.subheader("Per-feature summary")
    st.caption(
        "% default = fraction of tickers where the feature equals 0 (or the "
        "cap for days_to_next_earnings). High % default means the feature "
        "isn't carrying much signal — check the upstream data source."
    )
    rows = []
    for col in FEATURE_NAMES:
        s = feats_df[col]
        default_val = 60.0 if col == "days_to_next_earnings" else 0.0
        n_default = (s == default_val).sum()
        rows.append({
            "feature": col,
            "mean": float(s.mean()),
            "std": float(s.std()),
            "min": float(s.min()),
            "max": float(s.max()),
            "%_default": f"{100 * n_default / len(s):.1f}%",
        })
    summary = pd.DataFrame(rows)
    st.dataframe(summary, hide_index=True, width="stretch")


def _render_distributions(feats_df: pd.DataFrame) -> None:
    st.subheader("Feature distributions")
    feature = st.selectbox(
        "Pick a feature",
        FEATURE_NAMES,
        index=FEATURE_NAMES.index("rel_strength_sector_etf_30d"),
        key="features_dist_picker",
    )
    s = feats_df[feature]
    col1, col2 = st.columns([2, 1])
    with col1:
        st.bar_chart(s.value_counts(bins=30).sort_index(), height=250)
    with col2:
        st.metric("count", len(s))
        st.metric("mean", f"{s.mean():.4f}")
        st.metric("std", f"{s.std():.4f}")
        st.caption(f"min: {s.min():.4f} | max: {s.max():.4f}")


def _render_top_bottom_by_feature(feats_df: pd.DataFrame) -> None:
    st.subheader("Top / bottom 10 by feature")
    feature = st.selectbox(
        "Rank by",
        FEATURE_NAMES,
        index=FEATURE_NAMES.index("politician_flow_30d"),
        key="features_topk_picker",
    )
    s = feats_df[feature].sort_values(ascending=False)
    col_t, col_b = st.columns(2)
    with col_t:
        st.markdown(f"**Top 10 — {feature}**")
        st.dataframe(
            s.head(10).rename("value").reset_index().rename(columns={"index": "ticker"}),
            hide_index=True, width="stretch",
        )
    with col_b:
        st.markdown(f"**Bottom 10 — {feature}**")
        st.dataframe(
            s.tail(10).iloc[::-1].rename("value").reset_index().rename(columns={"index": "ticker"}),
            hide_index=True, width="stretch",
        )


def _render_correlation_matrix(feats_df: pd.DataFrame) -> None:
    st.subheader("Feature correlations")
    st.caption(
        "Pairs with |corr| > 0.7 suggest redundancy — one might be removable. "
        "Pairs with |corr| ≈ 0 carry independent signal."
    )
    corr = feats_df[FEATURE_NAMES].corr().round(2)
    # plotly heatmap (already a dashboard dep) instead of a pandas Styler
    # background_gradient, which requires matplotlib (not installed) and crashed
    # the tab.
    fig = px.imshow(
        corr,
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1,
        aspect="auto",
        text_auto=True,
    )
    st.plotly_chart(fig, width="stretch")


def _render_ticker_inspector(feats_df: pd.DataFrame) -> None:
    st.subheader("Single-ticker feature row")
    ticker = st.text_input(
        "Ticker",
        value="NVDA",
        key="features_ticker_input",
    ).strip().upper()
    if not ticker:
        st.info("Enter a ticker to inspect its feature row.")
        return
    if ticker not in feats_df.index:
        st.warning(
            f"{ticker} not in feature output for this asof. Either it's not "
            "in the universe, or it lacks enough price history (need ~252 "
            "trading days for dist_from_52w_high)."
        )
        return
    row = feats_df.loc[ticker]
    inspector = pd.DataFrame({
        "feature": row.index,
        "value": row.values,
    })
    st.dataframe(inspector, hide_index=True, width="stretch")
