"""Pipeline tab: live status of today's scheduled jobs.

Shows a status table for all 7 daily/weekly pipeline jobs, today's intended
orders (if any), and today's fills (if any). Read-only DuckDB queries only.

Sentinel files (data/sentinels/*.json) are not yet generated (Phase 5.5) --
this tab degrades gracefully when they are absent and derives status from DB.
"""

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import duckdb
import pandas as pd
import streamlit as st

from dashboard.data import DB_PATH
from sma.db_connect import read_only_connect
from sma.readiness import sentinel_lineage_stale
from sma.schedule import get as get_schedule
from sma.sentinels import read_sentinel

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models_artifacts"
SENTINELS_DIR = PROJECT_ROOT / "data" / "sentinels"

ET = ZoneInfo("America/New_York")

# NYSE holiday calendar (partial -- covers 2026 and near-term dates).
# Extend this list as needed. Format: YYYY-MM-DD strings in ET.
_NYSE_HOLIDAYS: set[str] = {
    "2026-01-01",  # New Year's Day
    "2026-01-19",  # MLK Day
    "2026-02-16",  # Presidents' Day
    "2026-04-03",  # Good Friday
    "2026-05-25",  # Memorial Day
    "2026-07-03",  # Independence Day (observed)
    "2026-09-07",  # Labor Day
    "2026-11-26",  # Thanksgiving
    "2026-11-27",  # Day after Thanksgiving (early close, treat as holiday)
    "2026-12-25",  # Christmas
}

INGEST_LABEL = "com.sma.ingest.daily"
PREDICT_LABEL = "com.sma.model.predict.daily"
AGENTS_LABEL = "com.sma.agents.daily"
DECIDE_LABEL = "com.sma.live.decide.daily"
RECONCILE_LABEL = "com.sma.live.reconcile.daily"
RECONCILE_RAN_LABEL = "com.sma.live.reconcile.daily.ran"
STOP_LOSS_LABEL = "com.sma.live.stop-loss.weekday"
RETRAIN_LABEL = "com.sma.model.retrain.weekly"


def _ro_conn() -> duckdb.DuckDBPyConnection:
    if not DB_PATH.exists():
        st.error(
            f"DuckDB not found at {DB_PATH}. "
            "Run `uv run python -m sma.ingest run` to populate."
        )
        st.stop()
    return read_only_connect(DB_PATH)


def _now_et() -> datetime:
    return datetime.now(tz=ET)


def _is_trading_day(dt: datetime) -> bool:
    """Return True if dt falls on a NYSE trading day (Mon-Fri, non-holiday)."""
    if dt.weekday() >= 5:  # Saturday=5, Sunday=6
        return False
    return dt.strftime("%Y-%m-%d") not in _NYSE_HOLIDAYS


def _prev_trading_day(d: date) -> date:
    """The NYSE trading day strictly before d (skips weekends/holidays)."""
    d -= timedelta(days=1)
    while not _is_trading_day(datetime(d.year, d.month, d.day, 12, tzinfo=ET)):
        d -= timedelta(days=1)
    return d


def _most_recent_completed_trading_day(now_et: datetime) -> date:
    """The latest NYSE session fully closed as of now_et.

    Today's session counts only at/after the 16:00 ET cash close; before the
    close (or on a weekend/holiday) the most recent completed session is the
    prior trading day. Used by the freshness banner to know what data we
    should already have — independent of today's job schedule."""
    close_et = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
    if _is_trading_day(now_et) and now_et >= close_et:
        return now_et.date()
    return _prev_trading_day(now_et.date())


def _data_staleness_trading_days(
    latest_data_date: date | None, now_et: datetime
) -> int:
    """Number of completed NYSE sessions newer than latest_data_date.

    0  → data is current through the most recent completed session.
    >=1 → the pipeline is that many trading days behind (a freeze): the
          per-job table can't surface this before the day's fire time, so the
          banner uses this instead. None (empty DB) returns a large sentinel
          so it always alarms rather than reading as fresh."""
    if latest_data_date is None:
        return 999
    target = _most_recent_completed_trading_day(now_et)
    n = 0
    d = target
    while d > latest_data_date:
        n += 1  # d is a trading day by construction of the walk-back
        d = _prev_trading_day(d)
    return n


def _latest_price_date() -> date | None:
    """MAX(date) in prices — the freshest market data we hold, or None."""
    con = _ro_conn()
    try:
        tables = {t for (t,) in con.execute("SHOW TABLES").fetchall()}
        if "prices" not in tables:
            return None
        row = con.execute("SELECT MAX(date) FROM prices").fetchone()
        return row[0] if row and row[0] is not None else None
    finally:
        con.close()


def _scheduled_time(label: str) -> str:
    return get_schedule(label).fire_time_et.strftime("%H:%M")


def _scheduled_display(label: str) -> str:
    job = get_schedule(label)
    days = ",".join(day.name.title() for day in job.days)
    return f"{days} {job.fire_time_et.strftime('%H:%M')}"


def _read_job_sentinel(label: str, asof_str: str) -> dict | None:
    try:
        return read_sentinel(label=label, asof=date.fromisoformat(asof_str))
    except Exception:
        return None


def _prediction_lineage_stale(asof_str: str) -> bool:
    return sentinel_lineage_stale(
        consumer=_read_job_sentinel(PREDICT_LABEL, asof_str),
        upstream=_read_job_sentinel(INGEST_LABEL, asof_str),
    )


def _latest_fill_asof() -> str | None:
    con = _ro_conn()
    try:
        tables = {t for (t,) in con.execute("SHOW TABLES").fetchall()}
        if "paper_fills" not in tables:
            return None
        row = con.execute("SELECT MAX(asof_date) FROM paper_fills").fetchone()
        return str(row[0]) if row and row[0] is not None else None
    finally:
        con.close()


def _reconciled_fill_asof(today_str: str) -> str | None:
    sentinel = _read_job_sentinel(RECONCILE_RAN_LABEL, today_str)
    if sentinel is not None and sentinel.get("reconciled_asof"):
        return str(sentinel["reconciled_asof"])
    return _latest_fill_asof()


# ---------------------------------------------------------------------------
# Per-job status helpers
# ---------------------------------------------------------------------------

def _ingest_status(today_str: str, now_et: datetime) -> tuple[str, str, str]:
    """Return (status, last_completed, output) for the ingest job."""
    con = _ro_conn()
    try:
        tables = {t for (t,) in con.execute("SHOW TABLES").fetchall()}
        if "ingest_log" not in tables:
            return _pending_or_missed(_scheduled_time(INGEST_LABEL), now_et), "", "table missing"

        rows = con.execute(
            """
            SELECT source, status, rows_inserted, MAX(started_at) AS ts
            FROM ingest_log
            WHERE DATE_TRUNC('day', started_at) = ?
            GROUP BY source, status, rows_inserted
            ORDER BY ts DESC
            """,
            [today_str],
        ).fetchall()
    finally:
        con.close()

    if not rows:
        return _pending_or_missed("18:30", now_et), "", "no run today"

    # Build per-source summary
    total_rows = sum(r[2] or 0 for r in rows)
    sources_ok = sum(1 for r in rows if r[1] == "ok")
    sources_total = len(rows)
    last_ts = max(r[3] for r in rows)
    last_ts_str = _fmt_ts(last_ts)

    # Critical sources: yfinance and alpaca must be ok
    source_map = {r[0]: r[1] for r in rows}
    critical = ("yfinance", "alpaca")
    all_critical_ok = all(source_map.get(s) == "ok" for s in critical)

    if all_critical_ok and sources_ok == sources_total:
        status = "DONE"
    elif sources_ok > 0:
        status = "IN PROGRESS"
    else:
        status = _pending_or_missed(_scheduled_time(INGEST_LABEL), now_et)

    output = f"{sources_ok}/{sources_total} sources OK, {total_rows:,} rows"
    return status, last_ts_str, output


def _predict_status(today_str: str, now_et: datetime) -> tuple[str, str, str]:
    con = _ro_conn()
    try:
        tables = {t for (t,) in con.execute("SHOW TABLES").fetchall()}
        if "predictions" not in tables:
            return _pending_or_missed(_scheduled_time(PREDICT_LABEL), now_et), "", "table missing"
        row = con.execute(
            "SELECT COUNT(*) AS cnt, MAX(computed_at) AS ts, MAX(model_id) AS mid "
            "FROM predictions WHERE asof_date = ?",
            [today_str],
        ).fetchone()
    finally:
        con.close()

    cnt = row[0] or 0
    if cnt > 0:
        model_date = (row[2] or "").split("_")[3][:10] if row[2] else ""
        output = f"{cnt} predictions, model {model_date}"
        if _prediction_lineage_stale(today_str):
            return "STALE (repredict pending)", _fmt_ts(row[1]), output
        return "DONE", _fmt_ts(row[1]), output
    return _pending_or_missed(_scheduled_time(PREDICT_LABEL), now_et), "", "0 predictions today"


def _agents_status(today_str: str, now_et: datetime) -> tuple[str, str, str]:
    con = _ro_conn()
    try:
        tables = {t for (t,) in con.execute("SHOW TABLES").fetchall()}
        if "theses" not in tables:
            return _pending_or_missed(_scheduled_time(AGENTS_LABEL), now_et), "", "table missing"
        row = con.execute(
            "SELECT COUNT(*) AS cnt, MAX(created_at) AS ts "
            "FROM theses WHERE asof_date = ?",
            [today_str],
        ).fetchone()
    finally:
        con.close()

    cnt = row[0] or 0
    if cnt > 0:
        return "DONE", _fmt_ts(row[1]), f"{cnt} theses today"
    return (
        _pending_or_missed(_scheduled_time(AGENTS_LABEL), now_et),
        "",
        "0 theses today (Phase 4 deferred)",
    )


def _decide_status(today_str: str, now_et: datetime) -> tuple[str, str, str]:
    con = _ro_conn()
    try:
        tables = {t for (t,) in con.execute("SHOW TABLES").fetchall()}
        if "intended_orders" not in tables:
            return _pending_or_missed(_scheduled_time(DECIDE_LABEL), now_et), "", "table missing"
        rows = con.execute(
            "SELECT status, COUNT(*) AS cnt, MAX(created_at) AS ts "
            "FROM intended_orders WHERE asof_date = ? "
            "GROUP BY status",
            [today_str],
        ).fetchall()
    finally:
        con.close()

    if not rows:
        return _pending_or_missed(_scheduled_time(DECIDE_LABEL), now_et), "", "0 orders today"

    last_ts = _fmt_ts(max(r[2] for r in rows))
    status_map = {r[0]: r[1] for r in rows}
    total = sum(status_map.values())

    if "submission_failed" in status_map:
        n_failed = status_map["submission_failed"]
        status_str = "IN PROGRESS" if "submitted" in status_map else "MISSED"
        output = f"{n_failed} submission_failed, {total} total orders"
    elif "submitted" in status_map:
        n_sub = status_map["submitted"]
        output = f"{n_sub} submitted, {total} total orders"
        status_str = "DONE"
    else:
        output = f"{total} orders today"
        status_str = "DONE"

    return status_str, last_ts, output


def _reconcile_status(today_str: str, now_et: datetime) -> tuple[str, str, str]:
    """Reconcile records fills under the DECIDE date of the batch it
    reconciled (yesterday), never under today — querying paper_fills by
    today's date showed a perpetual false '0 fills today' (audit
    pipeline.py:257). The run-date LIVENESS sentinel (2026-06-09) is the
    authority for 'did today's reconcile run', and its reconciled_asof keys
    the fills it recorded."""
    sentinel = None
    try:
        sentinel = read_sentinel(
            label=RECONCILE_RAN_LABEL,
            asof=date.fromisoformat(today_str),
        )
    except Exception:
        sentinel = None  # unreadable sentinel falls back to the legacy query

    if sentinel is not None:
        reconciled = sentinel.get("reconciled_asof")
        ts = _fmt_ts(sentinel.get("completed_at", ""))
        if not reconciled:
            return "DONE", ts, "ran; no unreconciled batch"
        con = _ro_conn()
        try:
            tables = {t for (t,) in con.execute("SHOW TABLES").fetchall()}
            cnt = 0
            if "paper_fills" in tables:
                cnt = con.execute(
                    "SELECT COUNT(*) FROM paper_fills WHERE asof_date = ?",
                    [reconciled],
                ).fetchone()[0] or 0
        finally:
            con.close()
        return "DONE", ts, f"{cnt} fills for batch {reconciled}"

    # Legacy fallback (pre-liveness-sentinel days): fills keyed by decide
    # date, so look at the most RECENT batch rather than today.
    con = _ro_conn()
    try:
        tables = {t for (t,) in con.execute("SHOW TABLES").fetchall()}
        if "paper_fills" not in tables:
            return (
                _pending_or_missed(_scheduled_time(RECONCILE_LABEL), now_et),
                "",
                "table missing",
            )
        row = con.execute(
            "SELECT COUNT(*), MAX(filled_at), MAX(asof_date) FROM paper_fills "
            "WHERE asof_date = (SELECT MAX(asof_date) FROM paper_fills)",
        ).fetchone()
    finally:
        con.close()

    cnt = row[0] or 0
    if cnt > 0:
        return "DONE", _fmt_ts(row[1]), f"{cnt} fills, latest batch {row[2]}"
    return _pending_or_missed(_scheduled_time(RECONCILE_LABEL), now_et), "", "no fills recorded"


def _stop_loss_status(today_str: str, now_et: datetime) -> tuple[str, str, str]:
    """Derive stop-loss status from net positions in paper_fills."""
    con = _ro_conn()
    try:
        tables = {t for (t,) in con.execute("SHOW TABLES").fetchall()}
        if "paper_fills" not in tables:
            return (
                _pending_or_missed(_scheduled_time(STOP_LOSS_LABEL), now_et),
                "",
                "table missing",
            )
        # Net position count: sum filled_shares signed by side
        row = con.execute("""
            SELECT SUM(
                CASE WHEN UPPER(side) = 'BUY' THEN filled_shares
                     WHEN UPPER(side) = 'SELL' THEN -filled_shares
                     ELSE 0 END
            ) AS net_shares
            FROM paper_fills
        """).fetchone()
    finally:
        con.close()

    net = row[0] or 0
    if net <= 0:
        output = "no positions held"
        # Stop-loss fires at 09:25 ET; if before that time show PENDING
        sched = get_schedule(STOP_LOSS_LABEL).fire_time_et
        fire_time = now_et.replace(
            hour=sched.hour, minute=sched.minute, second=0, microsecond=0
        )
        if now_et < fire_time:
            return "PENDING", "", output
        return "DONE", "", output  # fired and found nothing to stop
    return "TBD", "", f"{net} net shares held"


def _retrain_status() -> tuple[str, str, str]:
    """Return status based on latest model artifact file."""
    if not MODELS_DIR.exists():
        return "PENDING", "", "no artifacts dir"

    json_files = sorted(MODELS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime)
    if not json_files:
        return "PENDING", "", "no artifacts found"

    latest = json_files[-1]
    try:
        meta = json.loads(latest.read_text())
        train_end = meta.get("train_end_date", "unknown")
    except Exception:
        train_end = "unknown"

    mtime = datetime.fromtimestamp(latest.stat().st_mtime, tz=ET)
    return "DONE", _fmt_ts(mtime), f"latest artifact {train_end}"


def _pending_or_missed(scheduled_time_et: str, now_et: datetime) -> str:
    """Return PENDING if before fire time today, MISSED if past it with no data."""
    hour, minute = map(int, scheduled_time_et.split(":"))
    fire_dt = now_et.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if now_et < fire_dt:
        return "PENDING"
    # Give a 30-minute grace window before calling it MISSED
    grace_minutes = 30
    from datetime import timedelta
    if now_et < fire_dt + timedelta(minutes=grace_minutes):
        return "IN PROGRESS"
    return "MISSED"


def _fmt_ts(ts) -> str:
    if ts is None:
        return ""
    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            # Assume UTC (DuckDB stores UTC)
            ts = ts.replace(tzinfo=UTC)
        ts_et = ts.astimezone(ET)
        return ts_et.strftime("%Y-%m-%d %H:%M %Z")
    return str(ts)[:19]


# ---------------------------------------------------------------------------
# Today's orders + fills queries
# ---------------------------------------------------------------------------

@st.cache_data(ttl=30)
def _today_intended_orders(today_str: str) -> pd.DataFrame:
    con = _ro_conn()
    try:
        tables = {t for (t,) in con.execute("SHOW TABLES").fetchall()}
        if "intended_orders" not in tables:
            return pd.DataFrame()
        return con.execute(
            """
            SELECT ticker, side, target_shares, status, alpaca_order_id, error
            FROM intended_orders
            WHERE asof_date = ?
            ORDER BY ticker
            """,
            [today_str],
        ).fetchdf()
    finally:
        con.close()


@st.cache_data(ttl=30)
def _today_fills(today_str: str) -> pd.DataFrame:
    fill_asof = _reconciled_fill_asof(today_str)
    if fill_asof is None:
        return pd.DataFrame()
    con = _ro_conn()
    try:
        tables = {t for (t,) in con.execute("SHOW TABLES").fetchall()}
        if "paper_fills" not in tables:
            return pd.DataFrame()
        return con.execute(
            """
            SELECT ticker, side, filled_shares, fill_price, commission, fees,
                   status, submitted_at, filled_at, alpaca_order_id
            FROM paper_fills
            WHERE asof_date = ?
            ORDER BY ticker
            """,
            [fill_asof],
        ).fetchdf()
    finally:
        con.close()


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

def render() -> None:  # noqa: PLR0912
    st.header("Pipeline")
    st.caption(
        "Live status of today's scheduled jobs, derived from DuckDB. "
        "Sentinel files (Phase 5.5) not yet generated -- status inferred from DB state."
    )

    now_et = _now_et()
    today_str = now_et.strftime("%Y-%m-%d")
    weekday_name = now_et.strftime("%A")
    trading_day = _is_trading_day(now_et)

    # ── a) Today header ────────────────────────────────────────
    col_date, col_badge = st.columns([3, 1])
    with col_date:
        date_str = now_et.strftime("%Y-%m-%d")
        time_str = now_et.strftime("%H:%M %Z")
        st.subheader(f"{weekday_name}, {date_str}  --  {time_str}")
    with col_badge:
        if trading_day:
            st.success("TRADING DAY")
        else:
            st.warning("NON-TRADING DAY")

    # ── a2) Data-freshness banner ──────────────────────────────
    # The per-job table below is today-scoped (each job reads PENDING until its
    # fire time), so a multi-day freeze stays invisible all morning. This banner
    # is schedule-independent: it alarms the moment our data falls behind the
    # most recent completed NYSE session — the 6/1-6/3 freeze would have shown
    # red here every morning instead of a benign gray PENDING.
    latest_price_date = _latest_price_date()
    stale_days = _data_staleness_trading_days(latest_price_date, now_et)
    expected = _most_recent_completed_trading_day(now_et)
    if latest_price_date is None:
        st.error("DATA STALE -- prices table is empty; ingest has never run.")
    elif stale_days == 0:
        st.success(
            f"DATA FRESH -- prices current through {latest_price_date} "
            f"(latest completed session: {expected})."
        )
    else:
        plural = "s" if stale_days > 1 else ""
        st.error(
            f"DATA STALE -- prices {stale_days} trading day{plural} behind: "
            f"latest {latest_price_date}, expected through {expected}. "
            "The pipeline may be frozen (check ingest)."
        )

    st.divider()

    # ── b) Pipeline status table ───────────────────────────────
    st.subheader("Pipeline status")

    ingest_status, ingest_last, ingest_out = _ingest_status(today_str, now_et)
    predict_status, predict_last, predict_out = _predict_status(today_str, now_et)
    agents_status, agents_last, agents_out = _agents_status(today_str, now_et)
    decide_status, decide_last, decide_out = _decide_status(today_str, now_et)
    reconcile_status, reconcile_last, reconcile_out = _reconcile_status(today_str, now_et)
    stop_loss_status, stop_loss_last, stop_loss_out = _stop_loss_status(today_str, now_et)
    retrain_status, retrain_last, retrain_out = _retrain_status()

    job_rows = [
        {
            "Job": "ingest",
            "Scheduled (ET)": _scheduled_display(INGEST_LABEL),
            "Status": ingest_status,
            "Last completed": ingest_last,
            "Output": ingest_out,
        },
        {
            "Job": "predict",
            "Scheduled (ET)": _scheduled_display(PREDICT_LABEL),
            "Status": predict_status,
            "Last completed": predict_last,
            "Output": predict_out,
        },
        {
            "Job": "agents",
            "Scheduled (ET)": _scheduled_display(AGENTS_LABEL),
            "Status": agents_status,
            "Last completed": agents_last,
            "Output": agents_out,
        },
        {
            "Job": "decide",
            "Scheduled (ET)": _scheduled_display(DECIDE_LABEL),
            "Status": decide_status,
            "Last completed": decide_last,
            "Output": decide_out,
        },
        {
            "Job": "reconcile",
            "Scheduled (ET)": _scheduled_display(RECONCILE_LABEL),
            "Status": reconcile_status,
            "Last completed": reconcile_last,
            "Output": reconcile_out,
        },
        {
            "Job": "stop-loss",
            "Scheduled (ET)": _scheduled_display(STOP_LOSS_LABEL),
            "Status": stop_loss_status,
            "Last completed": stop_loss_last,
            "Output": stop_loss_out,
        },
        {
            "Job": "retrain",
            "Scheduled (ET)": _scheduled_display(RETRAIN_LABEL),
            "Status": retrain_status,
            "Last completed": retrain_last,
            "Output": retrain_out,
        },
    ]

    status_df = pd.DataFrame(job_rows)

    # Color-code status column via highlighting
    def _color_status(val: str) -> str:
        colors = {
            "DONE": "background-color: #1e7e34; color: white",
            "IN PROGRESS": "background-color: #856404; color: white",
            "PENDING": "background-color: #495057; color: white",
            "MISSED": "background-color: #721c24; color: white",
            "STALE (repredict pending)": "background-color: #721c24; color: white",
            "TBD": "background-color: #495057; color: white",
        }
        return colors.get(val, "")

    styled = status_df.style.map(_color_status, subset=["Status"])
    st.dataframe(styled, hide_index=True, width="stretch")

    st.divider()

    # ── c) Today's intended orders ─────────────────────────────
    orders_df = _today_intended_orders(today_str)
    st.subheader(f"Today's intended orders ({len(orders_df)} rows)")
    if orders_df.empty:
        st.info("No intended orders for today.")
    else:
        st.dataframe(orders_df, hide_index=True, width="stretch")

    st.divider()

    # ── d) Today's fills ───────────────────────────────────────
    fills_df = _today_fills(today_str)
    st.subheader(f"Today's fills ({len(fills_df)} rows)")
    if fills_df.empty:
        st.info("No fills for today.")
    else:
        st.dataframe(fills_df, hide_index=True, width="stretch")

    st.divider()

    # ── e) Refresh hint ────────────────────────────────────────
    st.caption(
        "This view does NOT auto-refresh. "
        "Click the refresh button in the top-right Streamlit toolbar to update."
    )
