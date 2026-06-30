"""Quality tab: post-ingest data-quality check reports."""

import re
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

_QUALITY_DIR = Path("logs/quality")

_CHECK_RE = re.compile(r"^\s*\[(PASS|FAIL)\]\s+(\S+?):\s*(.*)$")
_HEADER_RE = re.compile(r"^Quality report for (\d{4}-\d{2}-\d{2})\s*(?:\(run_id=(\d+)\))?")
_OVERALL_RE = re.compile(r"^OVERALL:\s+(PASS|FAIL)")


def _parse_report(text: str, fallback_date: str | None = None) -> dict:
    """Parse a single quality-report text file.

    Returns: {report_date, run_id, checks: [{name, status, detail}], overall}
    """
    report_date = fallback_date
    run_id: str | None = None
    checks: list[dict] = []
    overall: str | None = None

    for raw in text.splitlines():
        h = _HEADER_RE.match(raw)
        if h:
            report_date = h.group(1)
            run_id = h.group(2)
            continue
        c = _CHECK_RE.match(raw)
        if c:
            checks.append({
                "name": c.group(2),
                "status": c.group(1),
                "detail": c.group(3).strip(),
            })
            continue
        o = _OVERALL_RE.match(raw)
        if o:
            overall = o.group(1)

    return {
        "report_date": report_date,
        "run_id": run_id,
        "checks": checks,
        "overall": overall,
    }


@st.cache_data(ttl=60)
def _load_all_reports() -> list[dict]:
    """Read every .txt under logs/quality/ and return parsed reports newest first."""
    if not _QUALITY_DIR.exists():
        return []
    reports = []
    for path in sorted(_QUALITY_DIR.glob("*.txt"), reverse=True):
        text = path.read_text()
        parsed = _parse_report(text, fallback_date=path.stem)
        parsed["_path"] = str(path)
        parsed["_text"] = text
        reports.append(parsed)
    return reports


def render() -> None:
    st.header("Data quality")
    st.caption(
        "Post-ingest SQL checks. Run after each daily ingest, "
        "logged to `logs/quality/<date>.txt`."
    )

    reports = _load_all_reports()
    if not reports:
        st.info(
            "No quality reports yet. The daily ingest writes one per run "
            "to `logs/quality/`. Wait for the launchd job to fire (weekdays 18:30 ET) "
            "or run `uv run python -m sma.ingest run` manually."
        )
        return

    # --- Summary tiles ---
    latest = reports[0]
    n_reports = len(reports)
    n_overall_pass = sum(1 for r in reports if r["overall"] == "PASS")
    n_overall_fail = sum(1 for r in reports if r["overall"] == "FAIL")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Reports", str(n_reports))
    c2.metric("Pass days", str(n_overall_pass))
    c3.metric("Fail days", str(n_overall_fail))
    c4.metric(
        "Latest",
        latest["report_date"] or "?",
        delta=latest["overall"] or "",
        delta_color="normal" if latest["overall"] == "PASS" else "inverse",
    )

    st.divider()

    # --- Per-check pass-rate over time (stacked bar) ---
    st.subheader("Check status by day")
    rows = []
    for r in reports:
        for c in r["checks"]:
            rows.append({
                "date": r["report_date"],
                "check": c["name"],
                "status": c["status"],
            })
    if rows:
        df = pd.DataFrame(rows)
        # one row per (date, check) with color = status
        fig = px.scatter(
            df,
            x="date",
            y="check",
            color="status",
            color_discrete_map={"PASS": "#2ca02c", "FAIL": "#d62728"},
            symbol="status",
            title=None,
        )
        fig.update_traces(marker={"size": 14})
        fig.update_layout(
            height=350,
            xaxis_title="Report date",
            yaxis_title=None,
            legend_title=None,
        )
        st.plotly_chart(fig, width="stretch")

    st.divider()

    # --- Latest report detail ---
    st.subheader(f"Latest report: {latest['report_date']}")
    if latest["overall"] == "PASS":
        st.success(f"OVERALL: PASS ({len(latest['checks'])} checks)")
    elif latest["overall"] == "FAIL":
        n_fail = sum(1 for c in latest["checks"] if c["status"] == "FAIL")
        st.error(f"OVERALL: FAIL ({n_fail} of {len(latest['checks'])} checks failed)")

    if latest["checks"]:
        check_df = pd.DataFrame(latest["checks"])[["status", "name", "detail"]]
        check_df.columns = ["Status", "Check", "Detail"]
        st.dataframe(check_df, width="stretch", hide_index=True)

    st.divider()

    # --- Browse older reports ---
    st.subheader("Browse all reports")
    options = [r["report_date"] or r["_path"] for r in reports]
    pick = st.selectbox("Pick a report", options, index=0, key="quality_picker")
    chosen = next((r for r in reports if (r["report_date"] or r["_path"]) == pick), None)
    if chosen is not None:
        st.code(chosen["_text"], language="text")
