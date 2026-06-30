"""Model tab: inspect the trained XGBoost model, feature importances, and predictions."""

from datetime import date
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from sma.db_connect import read_only_connect

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


def render() -> None:
    st.header("Model inspector")

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
