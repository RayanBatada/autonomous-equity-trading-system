"""System tab: how the pipeline works + live freshness indicators.

Static explainers describe data sources, the daily flow, and model training.
Live queries against DuckDB show per-source last-fetch recency and let the
operator drill into recent activity at each pipeline step.
"""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import duckdb
import pandas as pd
import streamlit as st

from dashboard.data import DB_PATH
from sma.db_connect import read_only_connect
from sma.schedule import get as get_schedule

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models_artifacts"
PLISTS_DIR = PROJECT_ROOT / "ops" / "launchd"

ET = ZoneInfo("America/New_York")


# Static descriptions: (source key, endpoint, what it gives, free-tier note,
# active flag). Keep this list aligned with src/sma/ingest/sources/*.py.
DATA_SOURCES: list[dict[str, str | bool]] = [
    {"source": "alpaca", "endpoint": "Alpaca Market Data (IEX)",
     "gives": "OHLCV daily bars", "tier": "Free with paper account",
     "active": True},
    {"source": "alpaca_news", "endpoint": "data.alpaca.markets/v1beta1/news",
     "gives": "Tagged-symbol news (Benzinga)", "tier": "Free, 200 req/min",
     "active": True},
    {"source": "finnhub_news", "endpoint": "Finnhub /news",
     "gives": "Per-ticker company news (deduped)", "tier": "Free tier",
     "active": True},
    {"source": "finnhub_sentiment", "endpoint": "Finnhub /news-sentiment",
     "gives": "Sentiment score per (ticker, asof)",
     "tier": "Paid tier (free returns 403)", "active": False},
    {"source": "finnhub_fundamentals",
     "endpoint": "Finnhub fundamentals + earnings calendar",
     "gives": "Earnings dates + key ratios", "tier": "Free tier",
     "active": True},
    {"source": "newsapi", "endpoint": "newsapi.org",
     "gives": "Per-ticker news (1 req/ticker/day)",
     "tier": "Free, 100 req/day", "active": True},
    {"source": "edgar", "endpoint": "data.sec.gov submissions",
     "gives": "10-K / 10-Q / 8-K filings", "tier": "Free (User-Agent required)",
     "active": True},
    {"source": "yfinance", "endpoint": "Yahoo Finance bulk OHLCV",
     "gives": "Backup price feed", "tier": "Free (unofficial)",
     "active": True},
]


def _read_plist_time(filename: str) -> str:
    """Best-effort parse of Hour/Minute from the first StartCalendarInterval
    entry of an ops/launchd/<filename> plist. Returns 'unknown' if anything fails."""
    path = PLISTS_DIR / filename
    if not path.exists():
        return "unknown"
    try:
        text = path.read_text()
        # Tiny manual extraction; avoids pulling in plistlib for this tab.
        import re
        hours = re.findall(r"<key>Hour</key>\s*<integer>(\d+)</integer>", text)
        mins = re.findall(r"<key>Minute</key>\s*<integer>(\d+)</integer>", text)
        if not hours or not mins:
            return "unknown"
        return f"{int(hours[0]):02d}:{int(mins[0]):02d}"
    except Exception:
        return "unknown"


def _schedule_time(label: str) -> str:
    return get_schedule(label).fire_time_et.strftime("%H:%M")


def _schedule_days(label: str) -> str:
    names = [day.name.title() for day in get_schedule(label).days]
    weekdays = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    if names == weekdays:
        return "Mon-Fri"
    if names == weekdays + ["Sat", "Sun"]:
        return "Mon-Sun"
    return ", ".join(names)


# Schedule mirrors production launchd plists in ops/launchd and the canonical
# schedule module used to render them.
SCHEDULE: list[dict[str, str]] = [
    {"job": "ingest.daily",
     "current_et": _read_plist_time("com.sma.ingest.daily.plist"),
     "phase_5_5_et": _schedule_time("com.sma.ingest.daily"),
     "days": _schedule_days("com.sma.ingest.daily"),
     "purpose": "All 7 sources fire, write to DuckDB"},
    {"job": "model.predict.daily",
     "current_et": _read_plist_time("com.sma.model.predict.daily.plist"),
     "phase_5_5_et": _schedule_time("com.sma.model.predict.daily"),
     "days": _schedule_days("com.sma.model.predict.daily"),
     "purpose": "Build features, run XGBoost inference"},
    {"job": "agents.daily",
     "current_et": _read_plist_time("com.sma.agents.daily.plist"),
     "phase_5_5_et": _schedule_time("com.sma.agents.daily"),
     "days": _schedule_days("com.sma.agents.daily"),
     "purpose": "LLM thesis pipeline (researcher -> analyst -> strategist)"},
    {"job": "live.decide.daily",
     "current_et": _read_plist_time("com.sma.live.decide.daily.plist"),
     "phase_5_5_et": _schedule_time("com.sma.live.decide.daily"),
     "days": _schedule_days("com.sma.live.decide.daily"),
     "purpose": "Submit OPG buys (decisions + theses + risk rails)"},
    {"job": "live.stop-loss.weekday",
     "current_et": _read_plist_time("com.sma.live.stop-loss.weekday.plist"),
     "phase_5_5_et": _schedule_time("com.sma.live.stop-loss.weekday"),
     "days": _schedule_days("com.sma.live.stop-loss.weekday"),
     "purpose": "Pre-open stop-loss sweep (currently disabled)"},
    {"job": "live.reconcile.daily",
     "current_et": _read_plist_time("com.sma.live.reconcile.daily.plist"),
     "phase_5_5_et": _schedule_time("com.sma.live.reconcile.daily"),
     "days": _schedule_days("com.sma.live.reconcile.daily"),
     "purpose": "Pull fills, snapshot account, drift alerts"},
    {"job": "model.retrain.weekly",
     "current_et": _read_plist_time("com.sma.model.retrain.weekly.plist"),
     "phase_5_5_et": _schedule_time("com.sma.model.retrain.weekly"),
     "days": _schedule_days("com.sma.model.retrain.weekly"),
     "purpose": "Walk-forward XGBoost retrain"},
    {"job": "backup.daily",
     "current_et": _read_plist_time("com.sma.backup.daily.plist"),
     "phase_5_5_et": _schedule_time("com.sma.backup.daily"),
     "days": _schedule_days("com.sma.backup.daily"),
     "purpose": "DuckDB snapshot to iCloud, 30-daily + 12-monthly retention"},
]


# Graphviz DOT for the LOGICAL pipeline flow. Times intentionally omitted
# here — they belong in the Schedule table where they're accurate. This
# diagram shows the dependency ordering (which is invariant across phases).
PIPELINE_DOT = """
digraph pipeline {
  rankdir=TB;
  bgcolor="transparent";
  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=12,
        margin="0.18,0.10"];
  edge [fontname="Helvetica", fontsize=10, color="#888888"];

  ingest    [label="INGEST\\n7 sources → DuckDB",          fillcolor="#dbeafe"];
  predict   [label="PREDICT\\n12 features → XGBoost",      fillcolor="#e0e7ff"];
  agents    [label="AGENTS\\nresearcher → analyst → strategist",
                                                            fillcolor="#ede9fe"];
  decide    [label="DECIDE\\nrails + theses → submit OPG",  fillcolor="#fae8ff"];
  queue     [label="queued at Alpaca\\n(overnight)",
             shape=ellipse, style="dashed,filled",          fillcolor="#fef3c7"];
  auction   [label="OPEN AUCTION\\nOPG fills (or expires)", fillcolor="#fef3c7"];
  reconcile [label="RECONCILE\\nfills → paper_fills + snapshots",
                                                            fillcolor="#dcfce7"];
  stoploss  [label="STOP-LOSS sweep\\n(disabled today)",
             shape=box, style="rounded,dashed,filled",      fillcolor="#fee2e2"];
  retrain   [label="RETRAIN (weekly)\\nwalk-forward CV → new XGB",
             shape=box, style="rounded,filled",             fillcolor="#e0e7ff"];

  ingest -> predict;
  predict -> agents;
  agents -> decide;
  decide -> queue [label="DAY_OPG"];
  queue -> auction [label="next session"];
  auction -> reconcile [label="end of day"];
  stoploss -> auction [style=dashed, label="pre-open"];

  // Retrain feeds new model artifacts into PREDICT
  retrain -> predict [label="weekly", style=dashed, color="#9333ea"];
}
"""


def _ro_conn() -> duckdb.DuckDBPyConnection:
    if not DB_PATH.exists():
        st.error(f"DuckDB not found at {DB_PATH}.")
        st.stop()
    return read_only_connect(DB_PATH)


def _freshness_badge(last_run: datetime | None) -> str:
    """Return a readable freshness bucket. last_run from DuckDB is naive UTC."""
    if last_run is None:
        return "[no data]"
    if last_run.tzinfo is None:
        last_run = last_run.replace(tzinfo=UTC)
    age = datetime.now(UTC) - last_run
    if age < timedelta(hours=24):
        return "fresh (<24h)"
    if age < timedelta(hours=72):
        return "stale (24-72h)"
    return "old (>72h)"


@st.cache_data(ttl=60)
def _ingest_freshness() -> pd.DataFrame:
    con = _ro_conn()
    try:
        rows = con.execute("""
            SELECT source, MAX(started_at) AS last_run, COUNT(DISTINCT run_id) AS run_count
            FROM ingest_log GROUP BY source ORDER BY source
        """).fetchall()
    finally:
        con.close()
    if not rows:
        return pd.DataFrame(columns=["source", "last_run", "run_count", "freshness"])
    df = pd.DataFrame(rows, columns=["source", "last_run", "run_count"])
    df["freshness"] = df["last_run"].apply(_freshness_badge)
    return df


@st.cache_data(ttl=60)
def _recent_ingest_runs(limit: int = 5) -> pd.DataFrame:
    con = _ro_conn()
    try:
        rows = con.execute(
            "SELECT source, started_at, finished_at, status, rows_inserted, error "
            "FROM ingest_log ORDER BY started_at DESC LIMIT ?",
            [int(limit)],
        ).fetchall()
    finally:
        con.close()
    return pd.DataFrame(rows, columns=[
        "source", "started_at", "finished_at", "status", "rows_inserted", "error",
    ])


@st.cache_data(ttl=60)
def _latest_predictions(limit: int = 20) -> tuple[str | None, pd.DataFrame]:
    con = _ro_conn()
    try:
        latest_row = con.execute(
            "SELECT MAX(asof_date) FROM predictions"
        ).fetchone()
        latest = latest_row[0] if latest_row else None
        if latest is None:
            return None, pd.DataFrame()
        rows = con.execute(
            "SELECT ticker, target, predicted_value, model_id, computed_at "
            "FROM predictions WHERE asof_date = ? "
            "ORDER BY predicted_value DESC LIMIT ?",
            [latest, int(limit)],
        ).fetchall()
    finally:
        con.close()
    df = pd.DataFrame(rows, columns=[
        "ticker", "target", "predicted_value", "model_id", "computed_at",
    ])
    return str(latest), df


@st.cache_data(ttl=60)
def _latest_theses(limit: int = 10) -> tuple[str | None, pd.DataFrame]:
    con = _ro_conn()
    try:
        latest_row = con.execute("SELECT MAX(asof_date) FROM theses").fetchone()
        latest = latest_row[0] if latest_row else None
        if latest is None:
            return None, pd.DataFrame()
        rows = con.execute(
            "SELECT ticker, action_hint, conviction, score, catalyst_window "
            "FROM theses WHERE asof_date = ? "
            "ORDER BY score DESC LIMIT ?",
            [latest, int(limit)],
        ).fetchall()
    finally:
        con.close()
    df = pd.DataFrame(rows, columns=[
        "ticker", "action_hint", "conviction", "score", "catalyst_window",
    ])
    return str(latest), df


@st.cache_data(ttl=60)
def _recent_decide(limit: int = 10) -> pd.DataFrame:
    con = _ro_conn()
    try:
        rows = con.execute(
            "SELECT asof_date, ticker, side, target_shares, status, "
            "alpaca_order_id, error "
            "FROM intended_orders ORDER BY created_at DESC LIMIT ?",
            [int(limit)],
        ).fetchall()
    finally:
        con.close()
    return pd.DataFrame(rows, columns=[
        "asof_date", "ticker", "side", "target_shares", "status",
        "alpaca_order_id", "error",
    ])


@st.cache_data(ttl=60)
def _recent_fills_and_snapshots() -> tuple[pd.DataFrame, pd.DataFrame]:
    con = _ro_conn()
    try:
        fills = con.execute(
            "SELECT asof_date, ticker, side, filled_shares, fill_price, status, "
            "filled_at FROM paper_fills ORDER BY filled_at DESC LIMIT 10"
        ).fetchall()
        snaps = con.execute(
            "SELECT asof_date, equity, cash, long_market_value, position_count, "
            "created_at FROM account_snapshots ORDER BY asof_date DESC LIMIT 5"
        ).fetchall()
    finally:
        con.close()
    fills_df = pd.DataFrame(fills, columns=[
        "asof_date", "ticker", "side", "filled_shares", "fill_price", "status",
        "filled_at",
    ])
    snaps_df = pd.DataFrame(snaps, columns=[
        "asof_date", "equity", "cash", "long_market_value", "position_count",
        "created_at",
    ])
    return fills_df, snaps_df


def _latest_model_metadata() -> dict | None:
    if not MODELS_DIR.exists():
        return None
    metas = sorted(MODELS_DIR.glob("*.json"))
    if not metas:
        return None
    try:
        return json.loads(metas[-1].read_text())
    except Exception:
        return None


def render() -> None:
    st.title("System overview")
    st.markdown(
        "How this pipeline works: data ingestion every weekday evening, "
        "an XGBoost model retrained weekly, an LLM thesis layer, then "
        "decide / submit / reconcile. Live sections below show freshness "
        "and recent activity."
    )

    st.subheader("Logical pipeline flow")
    st.graphviz_chart(PIPELINE_DOT, width="stretch")
    st.caption(
        "Step ordering only — actual times are in the schedule table below. "
        "The dashed STOP-LOSS / RETRAIN edges show how those run alongside "
        "the main daily loop."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Data sources")
        sources_df = pd.DataFrame(DATA_SOURCES)
        st.dataframe(sources_df, width="stretch", hide_index=True)
        st.caption(
            "`active=False` means the source is disabled in `config.yaml` "
            "(typically because it requires a paid plan). 7 sources run on "
            "every ingest cycle today."
        )

        st.markdown("**Last fetch (live from `ingest_log`):**")
        fresh = _ingest_freshness()
        if fresh.empty:
            st.info("No ingest history yet.")
        else:
            st.dataframe(fresh, width="stretch", hide_index=True)

    with col2:
        st.subheader("Schedule")
        st.dataframe(pd.DataFrame(SCHEDULE), width="stretch", hide_index=True)
        st.caption(
            "**current_et** is read from the live launchd plists in "
            "`ops/launchd/`. **phase_5_5_et** is rendered from "
            "`sma.schedule.SCHEDULE`, the source used to generate those plists."
        )

    st.divider()
    st.subheader("Latest activity (drill-downs)")

    with st.expander("INGEST -- last 5 source runs"):
        runs = _recent_ingest_runs(limit=5)
        if runs.empty:
            st.info("No ingest runs yet.")
        else:
            st.dataframe(runs, width="stretch", hide_index=True)

    with st.expander("PREDICT -- top 20 from latest model run"):
        asof, preds = _latest_predictions(limit=20)
        if asof is None:
            st.info("No predictions yet.")
        else:
            st.caption(f"asof_date = {asof}")
            st.dataframe(preds, width="stretch", hide_index=True)

    with st.expander("AGENTS -- top 10 by score from latest theses run"):
        asof, theses_df = _latest_theses(limit=10)
        if asof is None:
            st.info("No theses yet.")
        else:
            st.caption(f"asof_date = {asof}")
            st.dataframe(theses_df, width="stretch", hide_index=True)

    with st.expander("DECIDE -- 10 most recent intended orders"):
        decide_df = _recent_decide(limit=10)
        if decide_df.empty:
            st.info("No intended orders yet.")
        else:
            st.dataframe(decide_df, width="stretch", hide_index=True)

    with st.expander("RECONCILE -- recent paper fills + account snapshots"):
        fills_df, snaps_df = _recent_fills_and_snapshots()
        st.markdown("**paper_fills** (last 10):")
        if fills_df.empty:
            st.info("No fills yet.")
        else:
            st.dataframe(fills_df, width="stretch", hide_index=True)
        st.markdown("**account_snapshots** (last 5):")
        if snaps_df.empty:
            st.info("No snapshots yet.")
        else:
            st.dataframe(snaps_df, width="stretch", hide_index=True)

    st.divider()
    st.subheader("Model training")
    st.markdown(
        "**XGBoost regressor**, retrained weekly on accumulated prices + "
        "news + fundamentals. Features: 12 per ticker per day (technical "
        "indicators + sentiment aggregates) defined in `src/sma/features/`. "
        "Target: forward returns over a 3-30 day swing horizon. "
        "Hyperparameter search uses **walk-forward cross-validation** "
        "(train 2015-2018 → validate 2019, train 2015-2019 → validate 2020, "
        "etc.) so no validation fold sees the future. Models are saved with "
        "their training-cutoff date; the predictor loads the latest model "
        "whose cutoff is on-or-before today, so live predictions never use "
        "a model trained on data they would not have had."
    )
    meta = _latest_model_metadata()
    if meta is not None:
        st.markdown("**Latest model artifact:**")
        st.json(meta)
    else:
        st.info("No model artifact found in `models_artifacts/`.")
