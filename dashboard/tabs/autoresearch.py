"""Autoresearch tab: surface the experiment_log for promotion review.

Shows top-k iterations by (monotonicity_score, sharpe_overall) and a
recent-runs feed so the operator can spot trends, bad iterations, and
candidates worth promoting.
"""

import duckdb
import pandas as pd
import streamlit as st

from dashboard.data import DB_PATH
from sma.db_connect import read_only_connect


def _ro_conn() -> duckdb.DuckDBPyConnection:
    if not DB_PATH.exists():
        st.error(f"DuckDB not found at {DB_PATH}.")
        st.stop()
    return read_only_connect(DB_PATH)


@st.cache_data(ttl=30)
def _top_experiments(k: int = 20) -> pd.DataFrame:
    con = _ro_conn()
    try:
        rows = con.execute("""
            SELECT experiment_id, run_id, iter_index, proposal_summary,
                   monotonicity_score, sharpe_overall, baseline_overall,
                   sharpe_w1, sharpe_w2, sharpe_w3, sharpe_w4, sharpe_w5,
                   agent_cost_usd, duration_seconds, status, created_at
            FROM autoresearch_experiments
            WHERE status = 'ok'
            ORDER BY monotonicity_score DESC NULLS LAST,
                     sharpe_overall DESC NULLS LAST
            LIMIT ?
        """, [int(k)]).fetchall()
    finally:
        con.close()
    cols = ["experiment_id", "run_id", "iter_index", "proposal_summary",
            "monotonicity_score", "sharpe_overall", "baseline_overall",
            "sharpe_w1", "sharpe_w2", "sharpe_w3", "sharpe_w4", "sharpe_w5",
            "agent_cost_usd", "duration_seconds", "status", "created_at"]
    return pd.DataFrame(rows, columns=cols)


@st.cache_data(ttl=30)
def _recent_experiments(limit: int = 30) -> pd.DataFrame:
    con = _ro_conn()
    try:
        rows = con.execute("""
            SELECT iter_index, status, monotonicity_score, sharpe_overall,
                   agent_cost_usd, duration_seconds, proposal_summary,
                   error, created_at
            FROM autoresearch_experiments
            ORDER BY created_at DESC LIMIT ?
        """, [int(limit)]).fetchall()
    finally:
        con.close()
    cols = ["iter_index", "status", "monotonicity_score", "sharpe_overall",
            "agent_cost_usd", "duration_seconds", "proposal_summary",
            "error", "created_at"]
    return pd.DataFrame(rows, columns=cols)


@st.cache_data(ttl=30)
def _run_summary() -> pd.DataFrame:
    con = _ro_conn()
    try:
        rows = con.execute("""
            SELECT run_id,
                   COUNT(*) AS total,
                   SUM(CASE WHEN status = 'ok' THEN 1 ELSE 0 END) AS ok,
                   SUM(CASE WHEN status = 'agent_error' THEN 1 ELSE 0 END) AS agent_err,
                   SUM(CASE WHEN status = 'eval_error' THEN 1 ELSE 0 END) AS eval_err,
                   SUM(CASE WHEN status = 'parse_error' THEN 1 ELSE 0 END) AS parse_err,
                   SUM(agent_cost_usd) AS total_cost,
                   MAX(monotonicity_score) AS best_mono,
                   MAX(sharpe_overall) AS best_sharpe,
                   MIN(created_at) AS started_at
            FROM autoresearch_experiments
            GROUP BY run_id ORDER BY started_at DESC LIMIT 20
        """).fetchall()
    finally:
        con.close()
    cols = ["run_id", "total", "ok", "agent_err", "eval_err", "parse_err",
            "total_cost", "best_mono", "best_sharpe", "started_at"]
    return pd.DataFrame(rows, columns=cols)


def _promotion_mask(df: pd.DataFrame) -> pd.Series:
    return (
        (df["monotonicity_score"] >= 3)
        & (df["sharpe_overall"] > df["baseline_overall"] + 0.1)
    )


def render() -> None:
    st.title("Autoresearch")
    st.markdown(
        "Phase 6 loop output. The LLM agent proposes new `tilt()` bodies "
        "for `src/sma/strategy/active.py`; each proposal is evaluated "
        "against the walk-forward CV harness across 5 sub-windows. "
        "**Promotion criterion**: `monotonicity_score >= 3 AND "
        "sharpe_overall > baseline + 0.1`. No auto-merge — winners shown "
        "here are candidates for the human to review + copy into "
        "`active.py` manually."
    )

    st.subheader("Run-level summary (last 20 runs)")
    runs = _run_summary()
    if runs.empty:
        st.info(
            "No autoresearch experiments yet. Run "
            "`python -m sma.autoresearch run --iterations 5` to start."
        )
        return
    st.dataframe(runs, width="stretch", hide_index=True)

    st.divider()
    st.subheader("Top candidates (by monotonicity, then overall Sharpe)")
    top = _top_experiments(k=20)
    if top.empty:
        st.info("No successful (status='ok') experiments yet.")
    else:
        # Flag promotion-eligible rows.
        promo_mask = _promotion_mask(top)
        if promo_mask.any():
            st.success(
                f"**{int(promo_mask.sum())} candidate(s) meet the promotion "
                f"criterion** (mono≥3 AND overall > baseline+0.1). Review "
                f"the diff before copying any tilt() body into active.py."
            )
        st.dataframe(top, width="stretch", hide_index=True)

    st.divider()
    st.subheader("Recent iterations (any status)")
    recent = _recent_experiments(limit=30)
    if recent.empty:
        st.info("No recent iterations.")
    else:
        st.dataframe(recent, width="stretch", hide_index=True)

    st.divider()
    st.caption(
        "Cost is Sonnet 4.6 API spend per iteration. Eval-error rows "
        "indicate the proposal failed during the real backtest run; "
        "agent-error rows indicate the LLM call itself failed. "
        "Parse-error rows indicate the proposal didn't preserve the "
        "frozen `tilt()` signature."
    )
