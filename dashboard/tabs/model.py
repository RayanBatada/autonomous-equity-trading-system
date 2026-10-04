"""Model tab: inspect the trained XGBoost model, feature importances, and predictions."""

from datetime import date
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from sma.db_connect import read_only_connect
from sma.eval import live_ic

_MODELS_DIR = Path("models_artifacts")
_DB_PATH = Path("data/sma.duckdb")
_UNIVERSE_PATH = Path("src/sma/universe.yaml")


@st.cache_resource
def _load_model_cached(pkl_path: Path):
    from sma.model.persistence import load_model
    return load_model(pkl_path)


@st.cache_data(ttl=60)
def _load_metadata_cached(json_path: Path) -> dict:
    from sma.model.persistence import load_metadata
    return load_metadata(json_path)


@st.cache_data(ttl=60)
def _list_pkl_files() -> list[Path]:
    if not _MODELS_DIR.exists():
        return []
    return sorted(_MODELS_DIR.glob("*.pkl"), reverse=True)


@st.cache_data(ttl=60)
def _predictions_count_for_model(model_id: str) -> int:
    if not _DB_PATH.exists():
        return 0
    try:
        con = read_only_connect(_DB_PATH)
        try:
            result = con.execute(
                "SELECT COUNT(*) FROM predictions WHERE model_id = $model_id",
                {"model_id": model_id},
            ).fetchone()
            return int(result[0]) if result else 0
        finally:
            con.close()
    except Exception:
        return 0


@st.cache_data(ttl=60)
def _recent_returns(universe: list[str], asof_date: date) -> dict[str, float]:
    """30-day trailing return ending on asof_date for each ticker."""
    if not _DB_PATH.exists():
        return {}
    from datetime import timedelta
    start = asof_date - timedelta(days=90)
    con = read_only_connect(_DB_PATH)
    try:
        df = con.execute(
            """
            SELECT ticker, date, adj_close
            FROM (
                SELECT *,
                       ROW_NUMBER() OVER (
                           PARTITION BY ticker, date
                           ORDER BY CASE source WHEN 'yfinance' THEN 0 ELSE 1 END
                       ) AS rn
                FROM prices
                WHERE ticker = ANY($tickers)
                  AND date BETWEEN $start_date AND $end_date
                  AND adj_close IS NOT NULL
            ) t
            WHERE rn = 1
            ORDER BY ticker, date
            """,
            {"tickers": list(universe), "start_date": start, "end_date": asof_date},
        ).fetchdf()
    finally:
        con.close()

    if df.empty:
        return {}

    results = {}
    for ticker, grp in df.groupby("ticker"):
        grp = grp.sort_values("date")
        if len(grp) < 5:
            continue
        # Use last 30 trading days (roughly 30 rows if daily data)
        cutoff_idx = max(0, len(grp) - 30)
        start_price = grp["adj_close"].iloc[cutoff_idx]
        end_price = grp["adj_close"].iloc[-1]
        if start_price and start_price > 0:
            results[ticker] = (end_price - start_price) / start_price
    return results


# ── Model Edge (live IC) ────────────────────────────────────────────────
#
# Rolling live cross-sectional rank-IC: computed from STORED predictions
# joined to realized forward returns from STORED prices -- never re-invoking
# the model. This is the honest "is there edge, right now" read (see
# CLAUDE.md: judge on IC, not P&L).

_MIN_CROSS_SECTION = live_ic.MIN_CROSS_SECTION
_IC_HORIZONS_DAYS = live_ic.IC_HORIZONS_DAYS
_IC_REGIME_WINDOW = live_ic.IC_REGIME_WINDOW

# Pure computation now lives in sma.eval.live_ic (extracted 2026-08-25 so the
# regime-turn monitoring check, sma.monitoring.check_regime_turn, can reuse
# it without importing streamlit). These are direct aliases, not
# reimplementations -- see live_ic.py for the full docstrings.
_forward_returns = live_ic.forward_returns
_rank_ic_series = live_ic.rank_ic_series
_smoothed_ic = live_ic.smoothed_ic
_trailing_ic_regime = live_ic.trailing_ic_regime


@st.cache_data(ttl=300)
def _decide_dates() -> list[date]:
    """Real live decide dates -- see sma.eval.live_ic.decide_dates for the
    full docstring; this just supplies the dashboard's DB path and adds
    Streamlit caching."""
    return live_ic.decide_dates(db_path=_DB_PATH)


@st.cache_data(ttl=300)
def _live_predictions_df() -> pd.DataFrame:
    """See sma.eval.live_ic.live_predictions_df for the full docstring.

    Cached with a longer TTL than most dashboard reads (predictions only
    change once/day at the evening predict job, and this joins the full
    live-history predictions table -- too heavy to redo on every rerun).
    """
    return live_ic.live_predictions_df(db_path=_DB_PATH, universe_path=_UNIVERSE_PATH)


@st.cache_data(ttl=300)
def _ic_prices_df() -> pd.DataFrame:
    """See sma.eval.live_ic.ic_prices_df for the full docstring."""
    return live_ic.ic_prices_df(db_path=_DB_PATH, universe_path=_UNIVERSE_PATH)


@st.cache_data(ttl=300)
def _model_edge_ic_df(horizon_days: int) -> pd.DataFrame:
    """Cached per-horizon rolling live IC series -- see
    sma.eval.live_ic.model_edge_ic_df for the actual computation. Cached
    because both raw DB pulls (predictions, full price history) and the
    per-date Spearman loop are too heavy to redo on every Streamlit rerun."""
    return live_ic.model_edge_ic_df(
        horizon_days, db_path=_DB_PATH, universe_path=_UNIVERSE_PATH
    )


def _render_model_edge_ic() -> None:
    st.subheader("Model Edge (live IC)")
    st.caption(
        "Rolling live cross-sectional rank-IC: Spearman correlation between "
        "each decide date's prediction ranks and REALIZED forward returns, "
        "computed from stored predictions joined to stored prices (never "
        "re-run through the model). Independent of turnover/costs/strategy "
        "-- the bedrock 'is there edge' read."
    )

    ic_by_horizon = {h: _model_edge_ic_df(h) for h in _IC_HORIZONS_DAYS}
    if all(df.empty for df in ic_by_horizon.values()):
        st.info(
            "Not enough live decide-date history yet to compute rolling IC "
            "-- need decide dates old enough to have realized 10/20-session "
            "forward returns (or predictions are too sparse per date). "
            "Check back as more live days accumulate."
        )
        return

    st.caption(
        "Known regime history (repeated live verification): mid-May 2026 "
        "positive (~+0.09 IC), June-July 2026 negative (-0.15 to -0.21 IC) "
        "-- a semis/momentum-factor reversal. This strategy's edge is "
        "regime-dependent, not a stable constant; read the chart below with "
        "that in mind."
    )

    # --- (1) chart: raw + smoothed IC for both horizons, with a 0 line and
    # each horizon's full-period mean marked.
    chart_frames = []
    for h in _IC_HORIZONS_DAYS:
        df = ic_by_horizon[h]
        if df.empty:
            continue
        d = df.sort_values("asof_date").copy()
        d["smoothed"] = _smoothed_ic(d["ic"])
        raw = d[["asof_date", "ic"]].rename(columns={"ic": "value"})
        raw["series"] = f"{h}d fwd IC"
        smooth = d[["asof_date", "smoothed"]].rename(columns={"smoothed": "value"})
        smooth["series"] = f"{h}d fwd IC (21-decide-date smoothed)"
        chart_frames.extend([raw, smooth])
    chart_df = pd.concat(chart_frames, ignore_index=True)

    fig = px.line(
        chart_df, x="asof_date", y="value", color="series",
        title="Rolling live rank-IC (prediction rank vs realized forward return)",
        labels={"asof_date": "Decide date", "value": "Spearman IC"},
    )
    fig.add_hline(y=0, line_dash="dot", line_color="gray")
    for h in _IC_HORIZONS_DAYS:
        df = ic_by_horizon[h]
        if not df.empty:
            fig.add_hline(
                y=float(df["ic"].mean()), line_dash="dash", line_color="gray",
                annotation_text=f"{h}d mean {df['ic'].mean():+.3f}",
                annotation_position="top left" if h == _IC_HORIZONS_DAYS[0] else "bottom left",
            )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Lag is honest: a decide date only gets a point once its realized "
        "forward return exists, so the most recent ~10 (or ~20) trading "
        "sessions have no point yet on that horizon's line -- that's "
        "elapsed-time lag, not zero edge."
    )

    # --- (2) compact regime read: trailing-21-session mean + t-stat, colored.
    st.markdown("**Trailing 21-decide-date regime read**")
    cols = st.columns(len(_IC_HORIZONS_DAYS))
    render_fn = {
        "positive": st.success, "negative": st.error,
        "neutral": st.warning, "insufficient": st.info,
    }
    for col, h in zip(cols, _IC_HORIZONS_DAYS, strict=True):
        df = ic_by_horizon[h]
        regime = (
            _trailing_ic_regime(df.sort_values("asof_date")["ic"])
            if not df.empty
            else {"mean": None, "t_stat": None, "n": 0, "level": "insufficient"}
        )
        with col:
            mean_str = f"{regime['mean']:+.3f}" if regime["mean"] is not None else "—"
            t_str = f"t={regime['t_stat']:+.2f}" if regime["t_stat"] is not None else "t=n/a"
            render_fn[regime["level"]](
                f"**{h}d horizon** — mean IC {mean_str} ({t_str}, n={regime['n']}) "
                f"— **{regime['level'].upper()}**"
            )
    st.caption(
        "t-stat caveat: consecutive decide dates' forward returns overlap "
        "heavily (a 10 or 20-session horizon vs ~1 trading day between "
        "decides), so these points are far from independent draws -- the "
        "textbook sqrt(n) t-stat overstates significance. Read this as a "
        "rough signpost, not a rigorous hypothesis test."
    )


def render() -> None:
    st.header("Model inspector")

    _render_model_edge_ic()
    st.divider()

    # --- A. Model selector ---
    pkl_files = _list_pkl_files()

    if not pkl_files:
        st.info(
            "No trained models found. Run "
            "`uv run python -m sma.model train --asof YYYY-MM-DD` to train one."
        )
        return

    selected_name = st.selectbox(
        "Model file",
        [p.name for p in pkl_files],
        index=0,
    )
    pkl_path = _MODELS_DIR / selected_name
    json_path = pkl_path.with_suffix(".json")

    # --- B. Metadata card ---
    st.subheader("Metadata")
    try:
        meta = _load_metadata_cached(json_path)
    except FileNotFoundError:
        st.warning(f"No JSON sidecar found for {selected_name}. Metadata unavailable.")
        meta = {}

    if meta:
        left_col, right_col = st.columns(2)
        with left_col:
            st.markdown(f"**Model ID:** `{meta.get('model_id', 'N/A')}`")
            st.markdown(f"**Target:** `{meta.get('target', 'N/A')}`")
            st.markdown(f"**Train end date:** `{meta.get('train_end_date', 'N/A')}`")
            st.markdown(f"**Commit:** `{str(meta.get('code_commit', ''))[:12]}`")
            st.markdown(
                f"**Train rows:** {meta.get('train_rows', 0):,} · "
                f"**Train time:** {meta.get('training_duration_seconds', 0):.2f}s"
            )

            def _fmt(v):
                return f"{v:.4f}" if isinstance(v, (int, float)) and v == v else "—"

            cv = meta.get("cv_rmse")
            tr = meta.get("train_rmse")
            gap = (
                cv - tr
                if all(isinstance(x, (int, float)) and x == x for x in (cv, tr))
                else None
            )
            m1, m2, m3 = st.columns(3)
            m1.metric("CV RMSE (out-of-sample)", _fmt(cv),
                      help="Walk-forward CV error — how the model generalizes. "
                           "The deploy gate blocks a retrain >10% worse than the incumbent.")
            m2.metric("Train RMSE (in-sample)", _fmt(tr))
            m3.metric("Overfit gap (CV − train)", _fmt(gap),
                      help="Larger gap = more overfitting.")

        with right_col:
            st.markdown(f"**Created at:** `{meta.get('created_at', 'N/A')}`")
            st.markdown("**Hyperparameters:**")
            st.json(meta.get("hyperparams", {}))

    st.divider()

    # --- C. Feature importance ---
    st.subheader("Feature importance (gain)")
    try:
        model = _load_model_cached(pkl_path)
        from sma.features.builder import FEATURE_NAMES
        importances = model.feature_importances_
        if len(importances) == len(FEATURE_NAMES):
            imp_df = pd.DataFrame(
                {"feature": FEATURE_NAMES, "importance": importances}
            ).sort_values("importance", ascending=True)

            fig = px.bar(
                imp_df,
                x="importance",
                y="feature",
                orientation="h",
                color="importance",
                color_continuous_scale="Viridis",
                labels={"importance": "Importance (gain)", "feature": "Feature"},
                title="Feature importance (gain)",
            )
            fig.update_layout(
                height=420,
                coloraxis_showscale=False,
                yaxis_title=None,
            )
            st.plotly_chart(fig, width="stretch")
        else:
            st.warning(
                f"Feature count mismatch: model has {len(importances)} importances "
                f"but FEATURE_NAMES has {len(FEATURE_NAMES)}."
            )
    except Exception as exc:
        st.error(f"Could not load model for feature importance: {exc}")

    st.divider()

    # --- D. Today's predictions ---
    st.subheader("Predictions")

    train_end_str = meta.get("train_end_date", "2020-01-01") if meta else "2020-01-01"
    try:
        min_date = date.fromisoformat(train_end_str)
    except ValueError:
        min_date = date(2020, 1, 1)

    pred_date = st.date_input(
        "As-of date",
        value=date.today(),
        min_value=min_date,
        max_value=date.today(),
        key="model_pred_date",
    )

    if not _UNIVERSE_PATH.exists():
        st.error(f"Universe file not found at {_UNIVERSE_PATH}.")
    else:
        if st.button("Compute predictions", key="model_pred_btn"):
            try:
                from sma.features.builder import FEATURE_NAMES, build_features
                from sma.ingest.universe import load_universe
                from sma.model.predictor import Predictor

                universe = load_universe(_UNIVERSE_PATH)

                with st.spinner("Building features and predicting..."):
                    model = _load_model_cached(pkl_path)

                    # Use Predictor internals directly so we control which pkl is used.
                    predictor = Predictor(models_dir=_MODELS_DIR, db_path=_DB_PATH)
                    prices = predictor._load_prices(pred_date, universe, lookback_days=400)

                    if prices.empty:
                        st.warning(
                            "No price data found for this date. "
                            "The database may not have data through this date."
                        )
                    else:
                        feats_df = build_features(prices, universe, pred_date)

                        if feats_df.empty:
                            st.warning(
                                "Could not build features for any ticker "
                                "(insufficient price history for this date)."
                            )
                        else:
                            # Use the artifact's OWN expected columns, not the
                            # current FEATURE_NAMES constant — an older model trained
                            # on a different column set would otherwise KeyError/skew.
                            expected = list(getattr(model, "feature_names_in_", FEATURE_NAMES))
                            x = feats_df[expected]
                            raw_preds = model.predict(x)
                            pred_map = {
                                ticker: float(p)
                                for ticker, p in zip(feats_df.index, raw_preds, strict=True)
                            }

                            # Preview ONLY — do not write to the live `predictions`
                            # table. The predict.daily job is its sole writer;
                            # persisting ad-hoc dashboard previews here corrupted the
                            # source of truth the pipeline reads (2026-06-05 audit).
                            st.caption(
                                "Preview scores — not persisted. The live predict "
                                "job is the only writer of the predictions table."
                            )

                            recent = _recent_returns(universe, pred_date)

                            rows = []
                            for ticker, predicted in pred_map.items():
                                recent_ret = recent.get(ticker)
                                nan = float("nan")
                                rows.append(
                                    {
                                        "Ticker": ticker,
                                        "Predicted 30d return": predicted,
                                        "Recent 30d return": (
                                            recent_ret if recent_ret is not None else nan
                                        ),
                                        "Predicted - Recent": (
                                            predicted - recent_ret
                                            if recent_ret is not None
                                            else nan
                                        ),
                                    }
                                )

                            pred_df = (
                                pd.DataFrame(rows)
                                .sort_values("Predicted 30d return", ascending=False)
                                .reset_index(drop=True)
                            )

                            # Top-5 caption
                            top5 = pred_df.head(5)
                            top5_str = ", ".join(
                                f"{row['Ticker']} ({row['Predicted 30d return']:+.1%})"
                                for _, row in top5.iterrows()
                            )
                            st.caption(f"Top picks: {top5_str}")

                            # Color the predicted return column
                            def _color_pred(val):
                                if pd.isna(val):
                                    return ""
                                if val > 0.05:
                                    return "color: #2ca02c"
                                if val < -0.05:
                                    return "color: #d62728"
                                return ""

                            styled = pred_df.style.format(
                                {
                                    "Predicted 30d return": "{:+.2%}",
                                    "Recent 30d return": "{:+.2%}",
                                    "Predicted - Recent": "{:+.2%}",
                                }
                            ).map(_color_pred, subset=["Predicted 30d return"])

                            st.dataframe(styled, width="stretch", hide_index=True)

            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

    st.divider()

    # --- E. Trade history for this model ---
    st.subheader("Prediction history")
    if meta:
        model_id = meta.get("model_id", "")
        count = _predictions_count_for_model(model_id)
        if count > 0:
            st.metric("Stored predictions for this model", f"{count:,}")
        else:
            st.info(
                "No stored predictions for this model yet. "
                "Run `uv run python -m sma.model predict` to generate live predictions."
            )
