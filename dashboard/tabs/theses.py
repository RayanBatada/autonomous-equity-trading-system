"""Theses tab: Phase 4 LLM agent outputs + cost telemetry."""

import json

import duckdb
import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.data import DB_PATH
from sma.db_connect import read_only_connect

# OVERRIDE_CATALYST_FLAGS mirrors the set in
# src/sma/backtest/strategies/xgb_top_k.py — these are the flags the strategy
# treats as "real catalyst" for the outside-top-30 override gate.
_OVERRIDE_CATALYST_FLAGS = {
    "earnings_beat", "guidance_raise", "m&a_announcement",
    "regulatory_win", "fda_approval", "secular_inflection",
    "macro_tailwind",
}


def _ro_conn() -> duckdb.DuckDBPyConnection:
    if not DB_PATH.exists():
        st.error(
            f"DuckDB not found at {DB_PATH}. "
            "Run `uv run python -m sma.ingest run` to populate."
        )
        st.stop()
    return read_only_connect(DB_PATH)


@st.cache_data(ttl=30)
def _has_phase4_tables() -> bool:
    con = _ro_conn()
    try:
        names = {t for (t,) in con.execute("SHOW TABLES").fetchall()}
        return "theses" in names and "agent_calls" in names
    finally:
        con.close()


@st.cache_data(ttl=30)
def _theses_summary() -> dict:
    con = _ro_conn()
    try:
        n_total = con.execute("SELECT COUNT(*) FROM theses").fetchone()[0]
        n_tickers = con.execute("SELECT COUNT(DISTINCT ticker) FROM theses").fetchone()[0]
        date_range = con.execute(
            "SELECT MIN(asof_date), MAX(asof_date) FROM theses"
        ).fetchone()
        latest_run = con.execute("SELECT MAX(run_id) FROM theses").fetchone()[0]
        return {
            "total": int(n_total or 0),
            "tickers": int(n_tickers or 0),
            "date_min": date_range[0],
            "date_max": date_range[1],
            "latest_run_id": latest_run,
        }
    finally:
        con.close()


@st.cache_data(ttl=30)
def _cost_summary() -> dict:
    """Spend snapshots: today / 7d / MTD / all-time."""
    con = _ro_conn()
    try:
        today = con.execute(
            "SELECT COALESCE(SUM(est_cost_usd), 0.0), COUNT(*) "
            "FROM agent_calls WHERE created_at::DATE = CURRENT_DATE"
        ).fetchone()
        last7 = con.execute(
            "SELECT COALESCE(SUM(est_cost_usd), 0.0), COUNT(*) "
            "FROM agent_calls WHERE created_at >= CURRENT_DATE - INTERVAL '7 days'"
        ).fetchone()
        mtd = con.execute(
            "SELECT COALESCE(SUM(est_cost_usd), 0.0), COUNT(*) "
            "FROM agent_calls "
            "WHERE date_trunc('month', created_at) = date_trunc('month', CURRENT_DATE)"
        ).fetchone()
        total = con.execute(
            "SELECT COALESCE(SUM(est_cost_usd), 0.0), COUNT(*) FROM agent_calls"
        ).fetchone()
        return {
            "today_usd": float(today[0] or 0.0),
            "today_calls": int(today[1] or 0),
            "last7_usd": float(last7[0] or 0.0),
            "last7_calls": int(last7[1] or 0),
            "mtd_usd": float(mtd[0] or 0.0),
            "mtd_calls": int(mtd[1] or 0),
            "total_usd": float(total[0] or 0.0),
            "total_calls": int(total[1] or 0),
        }
    finally:
        con.close()


@st.cache_data(ttl=30)
def _per_agent_breakdown() -> pd.DataFrame:
    con = _ro_conn()
    try:
        return con.execute(
            """
            SELECT
                agent_role AS role,
                COUNT(*) AS calls,
                ROUND(SUM(est_cost_usd), 4) AS total_cost_usd,
                ROUND(AVG(latency_ms), 0) AS avg_latency_ms,
                SUM(CASE WHEN status = 'ok' THEN 1 ELSE 0 END) AS ok_count,
                SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) AS error_count,
                SUM(CASE WHEN status = 'budget_exhausted_skipped' THEN 1 ELSE 0 END)
                    AS skipped_count
            FROM agent_calls
            GROUP BY agent_role
            ORDER BY calls DESC
            """
        ).df()
    finally:
        con.close()


@st.cache_data(ttl=30)
def _theses_index(ticker_filter: str | None) -> pd.DataFrame:
    """Latest thesis per (ticker, asof) — one row per thesis."""
    con = _ro_conn()
    try:
        params = []
        where = ""
        if ticker_filter and ticker_filter.strip() and ticker_filter != "All":
            where = "WHERE ticker = ?"
            params = [ticker_filter.strip().upper()]
        return con.execute(
            f"""
            SELECT
                ticker,
                asof_date,
                conviction,
                ROUND(score, 3) AS score,
                action_hint,
                flags,
                reasoning
            FROM theses
            {where}
            ORDER BY asof_date DESC, ticker ASC
            LIMIT 200
            """,
            params,
        ).df()
    finally:
        con.close()


@st.cache_data(ttl=30)
def _full_thesis(ticker: str, asof_str: str) -> dict | None:
    con = _ro_conn()
    try:
        row = con.execute(
            """
            SELECT news_summary, key_developments, notable_filings,
                   bull_case, bear_case, asymmetric_risks, catalyst_window,
                   conviction, score, flags, action_hint, reasoning,
                   run_id, created_at
            FROM theses
            WHERE ticker = ? AND asof_date = ?
            ORDER BY run_id DESC
            LIMIT 1
            """,
            [ticker, asof_str],
        ).fetchone()
        if row is None:
            return None
        return {
            "news_summary": row[0],
            "key_developments": _json_or_passthrough(row[1]),
            "notable_filings": _json_or_passthrough(row[2]),
            "bull_case": row[3],
            "bear_case": row[4],
            "asymmetric_risks": _json_or_passthrough(row[5]),
            "catalyst_window": row[6],
            "conviction": row[7],
            "score": row[8],
            "flags": _json_or_passthrough(row[9]),
            "action_hint": row[10],
            "reasoning": row[11],
            "run_id": row[12],
            "created_at": row[13],
        }
    finally:
        con.close()


def _json_or_passthrough(v):
    if v is None:
        return []
    if isinstance(v, list):
        return v
    if isinstance(v, str):
        try:
            return json.loads(v)
        except (json.JSONDecodeError, TypeError):
            return [v]
    return v


@st.cache_data(ttl=30)
def _cost_timeline() -> pd.DataFrame:
    """Per-day cost + cumulative cost since first call."""
    con = _ro_conn()
    try:
        return con.execute(
            """
            SELECT
                created_at::DATE AS day,
                ROUND(SUM(est_cost_usd), 5) AS daily_usd,
                COUNT(*) AS calls
            FROM agent_calls
            GROUP BY day
            ORDER BY day
            """
        ).df()
    finally:
        con.close()


@st.cache_data(ttl=30)
def _conviction_distribution() -> pd.DataFrame:
    """Count of theses by conviction (latest run per ticker/date)."""
    con = _ro_conn()
    try:
        return con.execute(
            """
            WITH latest AS (
                SELECT ticker, asof_date,
                       FIRST_VALUE(conviction) OVER (
                           PARTITION BY ticker, asof_date
                           ORDER BY run_id DESC
                       ) AS conviction
                FROM theses
            )
            SELECT conviction, COUNT(*) AS n
            FROM (SELECT DISTINCT ticker, asof_date, conviction FROM latest)
            GROUP BY conviction
            ORDER BY
                CASE conviction
                    WHEN 'strong_bullish' THEN 1
                    WHEN 'bullish' THEN 2
                    WHEN 'neutral' THEN 3
                    WHEN 'bearish' THEN 4
                    WHEN 'strong_bearish' THEN 5
                    ELSE 6
                END
            """
        ).df()
    finally:
        con.close()


@st.cache_data(ttl=30)
def _flag_distribution() -> pd.DataFrame:
    """Count of catalyst flags across all theses (UNNEST the JSON array)."""
    con = _ro_conn()
    try:
        df = con.execute(
            """
            SELECT flags FROM theses WHERE flags IS NOT NULL
            """
        ).df()
        if df.empty:
            return pd.DataFrame(columns=["flag", "count"])
        all_flags: list[str] = []
        for raw in df["flags"]:
            parsed = _json_or_passthrough(raw)
            if isinstance(parsed, list):
                all_flags.extend([str(f) for f in parsed if f])
        if not all_flags:
            return pd.DataFrame(columns=["flag", "count"])
        return (
            pd.Series(all_flags, name="flag")
            .value_counts()
            .reset_index()
            .rename(columns={"flag": "flag", "count": "count"})
        )
    finally:
        con.close()


@st.cache_data(ttl=30)
def _top_picks() -> pd.DataFrame:
    """strong_bullish theses with at least one override-eligible catalyst flag.

    These are the names the LLM is making its strongest case for that the
    quant model might not have surfaced. Most actionable signal in the table.
    """
    con = _ro_conn()
    try:
        return con.execute(
            """
            SELECT ticker, asof_date, ROUND(score, 3) AS score,
                   flags, action_hint, reasoning
            FROM theses
            WHERE conviction = 'strong_bullish'
            ORDER BY asof_date DESC, ticker ASC
            LIMIT 20
            """
        ).df()
    finally:
        con.close()


@st.cache_data(ttl=30)
def _top_avoids() -> pd.DataFrame:
    """strong_bearish theses — names the LLM thinks are sells/avoids."""
    con = _ro_conn()
    try:
        return con.execute(
            """
            SELECT ticker, asof_date, ROUND(score, 3) AS score,
                   flags, action_hint, reasoning
            FROM theses
            WHERE conviction IN ('bearish', 'strong_bearish')
            ORDER BY asof_date DESC, ticker ASC
            LIMIT 20
            """
        ).df()
    finally:
        con.close()


@st.cache_data(ttl=30)
def _recent_calls(limit: int = 50) -> pd.DataFrame:
    con = _ro_conn()
    try:
        return con.execute(
            """
            SELECT
                created_at,
                ticker,
                agent_role AS role,
                status,
                input_tokens,
                output_tokens,
                cache_read_tokens AS cache_read,
                ROUND(est_cost_usd, 5) AS cost_usd,
                latency_ms,
                CASE WHEN error IS NOT NULL THEN substr(error, 1, 60) ELSE NULL END AS error_excerpt
            FROM agent_calls
            ORDER BY created_at DESC
            LIMIT ?
            """,
            [limit],
        ).df()
    finally:
        con.close()


def _conviction_color(c: str | None) -> str:
    return {
        "strong_bullish": "#0a7d2e",
        "bullish": "#2ca02c",
        "neutral": "#888888",
        "bearish": "#d62728",
        "strong_bearish": "#7d0a0a",
    }.get(c or "", "#888888")


def render() -> None:
    st.header("Phase 4: LLM Theses")
    st.caption(
        "Researcher → Analyst → Strategist outputs persisted to `theses`. "
        "Cost telemetry from `agent_calls`."
    )

    if not _has_phase4_tables():
        st.info(
            "Phase 4 tables (`theses`, `agent_calls`) don't exist yet. "
            "Reconnect after running Phase 4 migration v3 — typically by running "
            "`uv run python -m sma.agents thesis --ticker AAPL --asof 2026-04-25` once "
            "(it auto-applies pending migrations)."
        )
        return

    summary = _theses_summary()
    cost = _cost_summary()

    # --- Top tiles ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Theses persisted", f"{summary['total']:,}")
    c2.metric("Distinct tickers", str(summary['tickers']))
    c3.metric("Today's spend", f"${cost['today_usd']:.4f}",
              delta=f"{cost['today_calls']} calls" if cost['today_calls'] else None)
    c4.metric("Total Phase 4 spend", f"${cost['total_usd']:.4f}",
              delta=f"{cost['total_calls']} calls" if cost['total_calls'] else None)

    if summary["date_min"] and summary["date_max"]:
        st.caption(
            f"Theses span {summary['date_min']} → {summary['date_max']} "
            f"(latest run_id: {summary['latest_run_id']})"
        )

    if summary["total"] == 0:
        st.warning(
            "No theses yet. Run `uv run python -m sma.agents thesis "
            "--ticker AAPL --asof 2026-04-25` for a single-ticker smoke "
            "(~$0.005), or `uv run python -m sma.agents prefill --start "
            "2025-07-01 --end 2026-04-26 --dry-run` to plan a backtest pre-fill."
        )
        return

    st.divider()

    # --- Cost detail ---
    st.subheader("Cost telemetry")
    cc1, cc2, cc3 = st.columns(3)
    cc1.metric("7-day spend", f"${cost['last7_usd']:.4f}",
                delta=f"{cost['last7_calls']} calls" if cost['last7_calls'] else None)
    cc2.metric("Month-to-date", f"${cost['mtd_usd']:.4f}",
                delta=f"{cost['mtd_calls']} calls" if cost['mtd_calls'] else None)
    daily_avg = cost["last7_usd"] / 7 if cost["last7_usd"] else 0.0
    cc3.metric("Daily avg (7d)", f"${daily_avg:.4f}")

    breakdown = _per_agent_breakdown()
    if not breakdown.empty:
        st.markdown("**Per-agent breakdown**")
        st.dataframe(breakdown, width="stretch", hide_index=True)

    # --- Cost timeline chart ---
    timeline = _cost_timeline()
    if not timeline.empty and len(timeline) >= 1:
        timeline = timeline.copy()
        timeline["cumulative_usd"] = timeline["daily_usd"].cumsum()
        st.markdown("**Spend over time**")
        col_a, col_b = st.columns(2)
        with col_a:
            fig_daily = px.bar(
                timeline, x="day", y="daily_usd",
                title="Daily spend ($USD)",
                labels={"day": "", "daily_usd": "USD"},
            )
            fig_daily.update_layout(height=240, showlegend=False, margin={"t": 36})
            st.plotly_chart(fig_daily, width="stretch")
        with col_b:
            fig_cum = px.line(
                timeline, x="day", y="cumulative_usd",
                title="Cumulative spend ($USD)",
                labels={"day": "", "cumulative_usd": "USD"},
            )
            fig_cum.update_traces(mode="lines+markers")
            fig_cum.update_layout(height=240, showlegend=False, margin={"t": 36})
            st.plotly_chart(fig_cum, width="stretch")

    st.divider()

    # --- Conviction + flag distributions ---
    st.subheader("What the LLM is saying")
    convictions = _conviction_distribution()
    flags = _flag_distribution()

    dcol1, dcol2 = st.columns(2)
    with dcol1:
        if not convictions.empty:
            color_map = {
                "strong_bullish": "#0a7d2e",
                "bullish": "#2ca02c",
                "neutral": "#888888",
                "bearish": "#d62728",
                "strong_bearish": "#7d0a0a",
            }
            fig = px.bar(
                convictions, x="conviction", y="n",
                color="conviction", color_discrete_map=color_map,
                title="Theses by conviction",
                labels={"conviction": "", "n": "theses"},
            )
            fig.update_layout(height=300, showlegend=False, margin={"t": 36})
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("No theses to chart yet.")

    with dcol2:
        if not flags.empty:
            fig = px.bar(
                flags.head(15), x="count", y="flag",
                orientation="h",
                title="Most-cited catalyst flags",
                labels={"flag": "", "count": "occurrences"},
            )
            fig.update_layout(
                height=300, margin={"t": 36},
                yaxis={"categoryorder": "total ascending"},
            )
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("No catalyst flags persisted yet.")

    st.divider()

    # --- Top picks + Top avoids ---
    st.subheader("Top picks / avoids")
    pcol, acol = st.columns(2)
    with pcol:
        st.markdown("🟢 **Strong-bullish picks**")
        picks = _top_picks()
        if picks.empty:
            st.info("No strong_bullish theses yet.")
        else:
            picks_disp = picks.copy()
            picks_disp["flags"] = picks_disp["flags"].apply(
                lambda v: ", ".join(_json_or_passthrough(v)) if v else "—"
            )
            picks_disp["catalyst?"] = picks["flags"].apply(
                lambda v: "✓" if (
                    set(_json_or_passthrough(v) or []) & _OVERRIDE_CATALYST_FLAGS
                ) else ""
            )
            st.dataframe(
                picks_disp[["ticker", "asof_date", "score", "catalyst?", "flags", "reasoning"]],
                width="stretch", hide_index=True,
            )
            st.caption(
                "✓ = has at least one override-eligible catalyst flag "
                "(the strategy can promote these into the buy list even if "
                "ranked outside the quant top-30)."
            )

    with acol:
        st.markdown("🔴 **Bearish / strong-bearish (potential exits)**")
        avoids = _top_avoids()
        if avoids.empty:
            st.info("No bearish theses yet.")
        else:
            avoids_disp = avoids.copy()
            avoids_disp["flags"] = avoids_disp["flags"].apply(
                lambda v: ", ".join(_json_or_passthrough(v)) if v else "—"
            )
            st.dataframe(
                avoids_disp[["ticker", "asof_date", "score", "flags", "reasoning"]],
                width="stretch", hide_index=True,
            )

    st.divider()

    # --- Theses browser ---
    st.subheader("Browse theses")
    fcol1, fcol2 = st.columns([2, 3])
    tickers = ["All"] + sorted(
        _theses_index(None)["ticker"].unique().tolist()
        if not _theses_index(None).empty else []
    )
    with fcol1:
        pick_ticker = st.selectbox(
            "Filter by ticker", tickers, index=0, key="thesis_ticker_picker"
        )
    with fcol2:
        conviction_filter = st.multiselect(
            "Filter by conviction (default: all)",
            ["strong_bullish", "bullish", "neutral", "bearish", "strong_bearish"],
            default=[],
            key="thesis_conviction_filter",
        )
    idx_df = _theses_index(pick_ticker if pick_ticker != "All" else None)
    if conviction_filter:
        idx_df = idx_df[idx_df["conviction"].isin(conviction_filter)].reset_index(drop=True)

    if idx_df.empty:
        st.info("No theses match the filter.")
        return

    # color-code conviction in the table by emoji
    def _emoji(c):
        return {
            "strong_bullish": "🟢🟢", "bullish": "🟢",
            "neutral": "⚪",
            "bearish": "🔴", "strong_bearish": "🔴🔴",
        }.get(c, "")
    idx_df_disp = idx_df.copy()
    idx_df_disp["conviction"] = idx_df_disp["conviction"].apply(
        lambda c: f"{_emoji(c)} {c or ''}"
    )
    st.dataframe(idx_df_disp, width="stretch", hide_index=True)

    st.divider()

    # --- Drill-down on a single thesis ---
    st.subheader("Drill-down: full agent trace")
    options = [
        f"{r.ticker} @ {r.asof_date}" for r in idx_df.itertuples()
    ]
    pick = st.selectbox("Pick a (ticker, asof_date)", options, index=0,
                          key="thesis_detail_picker")
    if pick:
        ticker, asof_str = pick.split(" @ ")
        thesis = _full_thesis(ticker, asof_str)
        if thesis is None:
            st.warning("No matching thesis row.")
            return

        # Header band w/ conviction color
        color = _conviction_color(thesis["conviction"])
        st.markdown(
            f"<div style='padding:8px 12px;background:{color};color:white;"
            f"border-radius:4px;font-weight:600'>"
            f"{ticker} @ {asof_str} — {thesis['conviction']} (score {thesis['score']:.2f}) — "
            f"action: {thesis['action_hint']}"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.caption(f"run_id={thesis['run_id']} · created_at={thesis['created_at']}")

        st.markdown("### Researcher")
        st.markdown(f"**News summary**: {thesis['news_summary']}")
        if thesis["key_developments"]:
            st.markdown("**Key developments**")
            for kd in thesis["key_developments"]:
                st.markdown(f"- {kd}")
        if thesis["notable_filings"]:
            st.markdown("**Notable filings**")
            for nf in thesis["notable_filings"]:
                st.markdown(f"- {nf}")

        st.markdown("### Analyst")
        st.markdown(f"**Bull case**: {thesis['bull_case']}")
        st.markdown(f"**Bear case**: {thesis['bear_case']}")
        st.markdown(f"**Catalyst window**: `{thesis['catalyst_window']}`")
        if thesis["asymmetric_risks"]:
            st.markdown("**Asymmetric risks**")
            for ar in thesis["asymmetric_risks"]:
                st.markdown(f"- {ar}")

        st.markdown("### Strategist")
        st.markdown(f"**Reasoning**: {thesis['reasoning']}")
        flag_str = ", ".join(f"`{f}`" for f in thesis["flags"]) if thesis["flags"] else "(none)"
        st.markdown(f"**Flags**: {flag_str}")

    st.divider()

    # --- Recent agent calls ---
    st.subheader("Recent calls (latest 50)")
    calls = _recent_calls(50)
    if calls.empty:
        st.info("No agent calls logged.")
    else:
        st.dataframe(calls, width="stretch", hide_index=True)
