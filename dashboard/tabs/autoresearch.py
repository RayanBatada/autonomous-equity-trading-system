"""Autoresearch tab: the live weekly CV-IC config-search record, plus an
archived view of the retired LLM tilt()-proposal loop it replaced.

Primary section reads data/sentinels/com.sma.autoresearch.nightly-*.json via
sma.autoresearch.history.load_search_runs / format_history -- the
deterministic search that has run every Monday since the 2026-06-19 migration
(commits 391c146/5a68736). Before this rewrite (2026-09-02) this tab queried
ONLY the old autoresearch_experiments DuckDB table -- the LLM tilt()-proposal
loop that has been dead since that migration (last row 2026-06-15), making
the whole tab a permanently-stale gauge. That table is kept below as a
collapsed archive; the live record is now primary.
"""

from __future__ import annotations

import math
from datetime import date, datetime
from zoneinfo import ZoneInfo

import duckdb
import pandas as pd
import streamlit as st

from dashboard.data import DB_PATH
from sma.autoresearch.history import format_history, load_search_runs
from sma.db_connect import read_only_connect
from sma.schedule import get as get_schedule
from sma.sentinels import sentinel_dir

ET = ZoneInfo("America/New_York")

SEARCH_LABEL = "com.sma.autoresearch.nightly"

# Must match sma.autoresearch.promotion.IC_PROMOTE_MARGIN. Kept as a literal
# (not imported) so the dashboard doesn't drag sma.model.persistence's heavy
# import chain into module load -- the same lazy-import convention model.py
# uses for _load_model_cached.
IC_PROMOTE_MARGIN = 0.005

ARCHIVE_HEADER = (
    "Archive: retired LLM proposal loop (May-Jun 2026, 0/52 promoted, "
    "retired 6/19)"
)

_HISTORY_COLUMNS = [
    "Asof", "Configs evaluated", "Best CV-IC", "Incumbent CV-IC",
    "Decision", "Reason", "Duration (fire→complete)",
]


def _ro_conn() -> duckdb.DuckDBPyConnection:
    if not DB_PATH.exists():
        st.error(f"DuckDB not found at {DB_PATH}.")
        st.stop()
    return read_only_connect(DB_PATH)


# ---------------------------------------------------------------------------
# Live weekly search record (primary section)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=60)
def _search_runs(limit: int = 20) -> list[dict]:
    """Recent config-search runs, newest first. sentinel_dir() is re-resolved
    on every call (cheap: it just reads an env var + globs a handful of small
    JSON files), so it honors SMA_SENTINEL_DIR and stays test-isolated. A
    missing or empty sentinel directory returns [] -- never raises (see
    load_search_runs)."""
    return load_search_runs(sentinel_dir(), limit=limit)


def _is_num(x) -> bool:
    return isinstance(x, (int, float)) and not (isinstance(x, float) and math.isnan(x))


def _fmt_ic(x) -> str:
    return f"{x:+.4f}" if _is_num(x) else "n/a"


def _decision_label(run: dict) -> str:
    if run.get("dry_run"):
        return "DRY-RUN"
    return "PROMOTE" if run.get("promoted") else "HOLD"


def _search_duration(run: dict) -> str:
    """Wall-clock from this run's scheduled Mon 07:00 ET fire to its
    completed_at sentinel timestamp. This is fire-to-complete, not pure
    compute time: a run the watchdog deferred (waiting on the retrain
    sentinel, see schedule.py's depends_on) and re-kicked later shows a
    longer "duration" here -- that's real information (the run was late),
    not a bug. Returns "n/a" on any missing or unparseable field; never
    raises."""
    asof_s, completed_s = run.get("asof"), run.get("completed_at")
    if not asof_s or not completed_s:
        return "n/a"
    try:
        asof_d = date.fromisoformat(asof_s)
        fire = datetime.combine(
            asof_d, get_schedule(SEARCH_LABEL).fire_time_et, tzinfo=ET
        )
        completed = datetime.fromisoformat(
            completed_s.replace("Z", "+00:00")
        ).astimezone(ET)
    except (ValueError, TypeError, KeyError):
        return "n/a"
    total_minutes = int((completed - fire).total_seconds() // 60)
    if total_minutes < 0:
        return "n/a"
    hours, minutes = divmod(total_minutes, 60)
    return f"{hours}h{minutes:02d}m" if hours else f"{minutes}m"


def _search_history_df(runs: list[dict]) -> pd.DataFrame:
    """Build the primary display table from load_search_runs' output
    (already newest-first -- not re-sorted here). All columns are rendered
    as strings: the underlying fields are a mix of int/float/None across
    older and newer sentinels (e.g. ensemble_seeds/gate_cv_ic only appear
    from 2026-08-31 on), and a DataFrame column with mixed int/str rows
    can't serialize through Arrow for st.dataframe (see dashboard/data.py's
    coverage_summary for the same fix)."""
    if not runs:
        return pd.DataFrame(columns=_HISTORY_COLUMNS)
    rows = [
        {
            "Asof": str(r.get("asof", "?")),
            "Configs evaluated": str(r.get("n_configs", "n/a")),
            "Best CV-IC": _fmt_ic(r.get("best_cv_ic")),
            "Incumbent CV-IC": _fmt_ic(r.get("incumbent_cv_ic")),
            "Decision": _decision_label(r),
            "Reason": str(r.get("reason") or "n/a"),
            "Duration (fire→complete)": _search_duration(r),
        }
        for r in runs
    ]
    return pd.DataFrame(rows, columns=_HISTORY_COLUMNS)


# ---------------------------------------------------------------------------
# Archived LLM tilt()-proposal loop (retired 2026-06-19; collapsed by default)
# ---------------------------------------------------------------------------

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


def _render_archive() -> None:
    st.caption(
        "Retired 2026-06-19 (commit 5a68736). The LLM agent proposed new "
        "`tilt()` bodies for `src/sma/strategy/active.py`; each proposal "
        "was evaluated against the walk-forward CV harness across 5 "
        "sub-windows, with no auto-merge -- a human had to review and copy "
        "a winner into `active.py` manually. 52 iterations ran "
        "2026-05-25..2026-06-15; 0 were ever promoted. Replaced above by "
        "the deterministic CV-IC config search. The data below is frozen "
        "at 2026-06-15 -- kept for the historical record only."
    )

    st.subheader("Run-level summary (last 20 runs)")
    runs_df = _run_summary()
    if runs_df.empty:
        st.info("No autoresearch_experiments rows found.")
        return
    st.dataframe(runs_df, width="stretch", hide_index=True)

    st.divider()
    st.subheader("Top candidates (by monotonicity, then overall Sharpe)")
    top = _top_experiments(k=20)
    if top.empty:
        st.info("No successful (status='ok') experiments.")
    else:
        promo_mask = _promotion_mask(top)
        if promo_mask.any():
            st.success(
                f"{int(promo_mask.sum())} candidate(s) met the promotion "
                "criterion (mono≥3 AND overall > baseline+0.1), but "
                "the loop was retired before any human copied one into "
                "active.py."
            )
        st.dataframe(top, width="stretch", hide_index=True)

    st.divider()
    st.subheader("Recent iterations (any status)")
    recent = _recent_experiments(limit=30)
    if recent.empty:
        st.info("No iterations recorded.")
    else:
        st.dataframe(recent, width="stretch", hide_index=True)


def render() -> None:
    st.title("Autoresearch")

    st.subheader("Weekly search record")
    runs = _search_runs(limit=20)
    if not runs:
        st.info(
            f"No config-search runs recorded yet in `{sentinel_dir()}` "
            "(directory empty or missing) -- first run Mon 07:00 ET."
        )
    else:
        st.dataframe(_search_history_df(runs), width="stretch", hide_index=True)
        n_promoted = sum(
            1 for r in runs if r.get("promoted") and not r.get("dry_run")
        )
        st.caption(
            "Deterministic CV-IC config search (`python -m sma.autoresearch "
            "search`, Mon 07:00 ET, depends on the Mon 04:00 retrain) -- no "
            "LLM in this loop, nothing here needs human review before it "
            "deploys. Promotes only when the best evaluated config's CV-IC "
            f"clears the incumbent's by ≥ {IC_PROMOTE_MARGIN} "
            f"({IC_PROMOTE_MARGIN:.3f} ≈ 1 SE of the CV-IC estimate -- "
            "deliberately noise-proof, so a PROMOTE reflects real edge, not "
            f"a coin-flip). {len(runs)} run(s) shown, newest first, "
            f"{n_promoted} promoted. Duration is fire-time-to-completion, "
            "not pure compute time -- a run the watchdog deferred and "
            "re-kicked shows longer here."
        )
        with st.expander(
            "Full per-config detail (every config evaluated, incl. HELD runs)"
        ):
            st.text(format_history(runs, margin=IC_PROMOTE_MARGIN))

    st.divider()

    with st.expander(ARCHIVE_HEADER, expanded=False):
        _render_archive()

    st.divider()
    st.caption(
        "This view does not auto-refresh. Click the refresh button in the "
        "top-right Streamlit toolbar to update."
    )
