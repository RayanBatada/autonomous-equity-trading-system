"""Paper-trading tab: Phase 5 live module visibility.

Displays:
- Last decide / stop-loss / reconcile fire times
- Account equity curve from account_snapshots
- Recent intended_orders (last 14 days)
- Recent paper_fills (last 14 days)
- Drift indicators (no-fill rate, partial fills, catastrophic-loss flag)

All reads are read-only DuckDB queries. No Alpaca API calls — this tab shows
what was persisted by the live module's reconcile job, not live broker state.
For real-time positions, log into the Alpaca paper dashboard at
https://app.alpaca.markets/paper/dashboard/overview.
"""

import duckdb
import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.data import DB_PATH
from sma.db_connect import read_only_connect
from sma.readiness import sentinel_lineage_stale
from sma.sentinels import read_sentinel

INGEST_LABEL = "com.sma.ingest.daily"
PREDICT_LABEL = "com.sma.model.predict.daily"


def _ro_conn() -> duckdb.DuckDBPyConnection:
    if not DB_PATH.exists():
        st.error(f"DuckDB not found at {DB_PATH}.")
        st.stop()
    return read_only_connect(DB_PATH)


@st.cache_data(ttl=15)
def _has_phase5_tables() -> bool:
    con = _ro_conn()
    try:
        names = {t for (t,) in con.execute("SHOW TABLES").fetchall()}
        return {"intended_orders", "paper_fills", "account_snapshots"}.issubset(names)
    finally:
        con.close()


@st.cache_data(ttl=15)
def _last_fire(source: str) -> dict | None:
    con = _ro_conn()
    try:
        row = con.execute(
            "SELECT MAX(created_at), COUNT(*) FROM intended_orders WHERE source = ?",
            [source],
        ).fetchone()
    finally:
        con.close()
    if row is None or row[0] is None:
        return None
    return {"ts": row[0], "count": int(row[1] or 0)}


@st.cache_data(ttl=15)
def _last_snapshot() -> dict | None:
    con = _ro_conn()
    try:
        # The single latest snapshot ROW — independent MAX() aggregates could mix
        # equity from one day with position_count from another (2026-06-05 audit).
        row = con.execute(
            "SELECT asof_date, created_at, equity, position_count "
            "FROM account_snapshots ORDER BY asof_date DESC, created_at DESC LIMIT 1"
        ).fetchone()
    finally:
        con.close()
    if row is None or row[0] is None:
        return None
    return {
        "asof_date": row[0],
        "ts": row[1],
        "equity": float(row[2] or 0),
        "position_count": int(row[3] or 0),
    }


@st.cache_data(ttl=15)
def _equity_curve_df() -> pd.DataFrame:
    con = _ro_conn()
    try:
        df = con.execute("""
            SELECT asof_date, equity, cash, long_market_value, position_count
            FROM account_snapshots
            ORDER BY asof_date
        """).df()
    finally:
        con.close()
    if not df.empty:
        df["asof_date"] = pd.to_datetime(df["asof_date"])
    return df


def _prediction_lineage_stale(asof) -> bool:
    try:
        predict_sentinel = read_sentinel(label=PREDICT_LABEL, asof=asof)
        ingest_sentinel = read_sentinel(label=INGEST_LABEL, asof=asof)
    except Exception:
        return False
    return sentinel_lineage_stale(consumer=predict_sentinel, upstream=ingest_sentinel)


@st.cache_data(ttl=30)
def _strategy_snapshot() -> dict:
    """Compare latest model top-K against currently-held positions.

    Returns a dict with:
      - asof: the latest prediction date in the DB
      - top: DataFrame[ticker, predicted_value] for top-10 picks
      - held: set of currently-held tickers (from paper_fills net)
      - new_buys: tickers in top-10 NOT currently held
      - force_sells: held tickers NOT in top-10 (likely SELL targets)
      - keepers: top-10 AND held
    """
    con = _ro_conn()
    try:
        latest_date = con.execute(
            "SELECT MAX(asof_date) FROM predictions"
        ).fetchone()[0]
        if latest_date is None:
            return {"asof": None, "top": pd.DataFrame()}
        # Top-10 predictions on the latest asof_date — using the latest
        # model_id only (tie-break by computed_at DESC).
        latest_model = con.execute(
            "SELECT model_id FROM predictions WHERE asof_date = ? "
            "ORDER BY computed_at DESC LIMIT 1", [latest_date],
        ).fetchone()[0]
        top = con.execute(
            "SELECT ticker, predicted_value FROM predictions "
            "WHERE asof_date = ? AND model_id = ? "
            "ORDER BY predicted_value DESC LIMIT 10",
            [latest_date, latest_model],
        ).df()
        held_rows = con.execute("""
            SELECT ticker
            FROM paper_fills
            WHERE filled_shares > 0
            GROUP BY ticker
            HAVING SUM(CASE WHEN side='BUY' THEN filled_shares ELSE -filled_shares END) > 0
        """).fetchall()
    finally:
        con.close()
    held = {r[0] for r in held_rows}
    top_set = set(top["ticker"]) if not top.empty else set()
    return {
        "asof": latest_date,
        "model_id": latest_model,
        "prediction_stale": _prediction_lineage_stale(latest_date),
        "top": top,
        "held": held,
        "new_buys": sorted(top_set - held),
        "force_sells": sorted(held - top_set),
        "keepers": sorted(top_set & held),
    }


@st.cache_data(ttl=15)
def _bench_returns_df(start_date) -> pd.DataFrame:
    """Return DataFrame with date + cumulative % return for SPY since start_date.

    Uses prices.SPY (preferring yfinance for split-adjusted closes). Aligned
    to trading days only — gaps are handled by pandas' ffill so the chart
    plots smoothly over weekends.
    """
    con = _ro_conn()
    try:
        df = con.execute("""
            SELECT date, adj_close
            FROM (
              SELECT date, adj_close,
                     ROW_NUMBER() OVER (
                       PARTITION BY date
                       ORDER BY CASE source WHEN 'yfinance' THEN 0
                                            WHEN 'alpaca' THEN 1
                                            ELSE 2 END
                     ) AS rn
              FROM prices
              WHERE ticker = 'SPY' AND date >= ? AND adj_close IS NOT NULL
            ) t WHERE rn = 1 ORDER BY date
        """, [start_date]).df()
    finally:
        con.close()
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    base = df["adj_close"].iloc[0]
    df["spy_return_pct"] = (df["adj_close"] / base - 1.0) * 100.0
    return df[["date", "spy_return_pct"]]


@st.cache_data(ttl=15)
def _recent_intended_orders(days: int = 14) -> pd.DataFrame:
    con = _ro_conn()
    try:
        df = con.execute(f"""
            SELECT asof_date, ticker, side, target_shares, target_weight,
                   last_price, source, status, error, alpaca_order_id, created_at
            FROM intended_orders
            WHERE asof_date >= CURRENT_DATE - INTERVAL '{days} days'
            ORDER BY created_at DESC
        """).df()
    finally:
        con.close()
    return df


@st.cache_data(ttl=15)
def _recent_fills(days: int = 14) -> pd.DataFrame:
    con = _ro_conn()
    try:
        df = con.execute(f"""
            SELECT asof_date, ticker, side, filled_shares, fill_price,
                   commission, fees, status, submitted_at, filled_at,
                   alpaca_order_id
            FROM paper_fills
            WHERE asof_date >= CURRENT_DATE - INTERVAL '{days} days'
            ORDER BY filled_at DESC
        """).df()
    finally:
        con.close()
    return df


@st.cache_data(ttl=15)
def _drift_summary(days: int = 14) -> dict:
    """Compute no-fill rate, partial-fill count for the last N days."""
    con = _ro_conn()
    try:
        # Key off "order was placed at the broker" (alpaca_order_id IS NOT NULL),
        # NOT status='submitted' — reconcile now updates status to terminal
        # (filled/expired/...) so 'submitted' would only match still-pending
        # orders and collapse the denominator. Matches reconcile._detect_no_fill_drift.
        intended = con.execute(f"""
            SELECT COUNT(*) FROM intended_orders
            WHERE asof_date >= CURRENT_DATE - INTERVAL '{days} days'
              AND source = 'decide'
              AND alpaca_order_id IS NOT NULL
        """).fetchone()[0] or 0
        no_fill = con.execute(f"""
            SELECT COUNT(*) FROM intended_orders i
            LEFT JOIN paper_fills f ON i.alpaca_order_id = f.alpaca_order_id
            WHERE i.asof_date >= CURRENT_DATE - INTERVAL '{days} days'
              AND i.source = 'decide'
              AND i.alpaca_order_id IS NOT NULL
              AND f.alpaca_order_id IS NULL
        """).fetchone()[0] or 0
        partial = con.execute(f"""
            SELECT COUNT(*) FROM intended_orders i
            JOIN paper_fills f ON i.alpaca_order_id = f.alpaca_order_id
            WHERE i.asof_date >= CURRENT_DATE - INTERVAL '{days} days'
              AND f.filled_shares < i.target_shares * 0.9
        """).fetchone()[0] or 0
    finally:
        con.close()
    no_fill_pct = (no_fill / intended) if intended > 0 else 0.0
    return {
        "intended": int(intended),
        "no_fill": int(no_fill),
        "no_fill_pct": no_fill_pct,
        "partial": int(partial),
    }


def _fmt_ts(ts) -> str:
    if ts is None:
        return "never"
    if hasattr(ts, "astimezone"):
        return ts.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    return str(ts)


def _alpha_trust_banner() -> None:
    """Excess return vs SPY + how much to trust it (sample size). The bot is fully
    long, so absolute return is mostly market beta — only EXCESS over SPY is
    evidence of edge, and at a tiny live sample even that is noise. Surfacing this
    stops early P&L from being over-read as skill."""
    from sma.eval.performance import alpha_vs_benchmark, trust_level

    con = _ro_conn()
    try:
        snaps = con.execute(
            "SELECT asof_date, equity FROM account_snapshots ORDER BY asof_date"
        ).fetchall()
        dates = [str(r[0]) for r in snaps]
        eq = [float(r[1]) for r in snaps]
        spy = [
            con.execute(
                "SELECT adj_close FROM prices WHERE ticker='SPY' AND date=? "
                "AND source='yfinance' LIMIT 1",
                [d],
            ).fetchone()
            for d in dates
        ]
    finally:
        con.close()

    pairs = [(e, float(s[0])) for e, s in zip(eq, spy, strict=True) if s is not None]
    if len(pairs) < 2:
        return
    a = alpha_vs_benchmark([e for e, _ in pairs], [s for _, s in pairs])
    lvl, msg = trust_level(len(eq))
    line = (
        f"**Strategy {a['strategy_return']:+.2%}** vs **SPY {a['benchmark_return']:+.2%}** "
        f"→ **excess (alpha) {a['excess_return']:+.2%}** over {len(eq)} live days.  \n"
        f"Trust **{lvl}** — {msg}"
    )
    render_fn = {"LOW": st.error, "MEDIUM": st.warning, "HIGH": st.success}[lvl]
    render_fn(line)


def render() -> None:
    st.header("Paper Trading")

    if not _has_phase5_tables():
        st.warning(
            "Phase 5 schema not yet applied to this DuckDB. "
            "Run any sma.live CLI command (e.g., `python -m sma.live status`) "
            "to trigger schema migration v4."
        )
        return

    _alpha_trust_banner()

    # --- Strategy snapshot: what does the model want vs what we hold?
    # This is the "what will decide do tonight" preview.
    st.subheader("Strategy snapshot — model wants vs held")
    snap = _strategy_snapshot()
    if snap.get("asof") is None:
        st.info("No predictions in the DB yet — predict.daily hasn't fired.")
    else:
        prediction_status = (
            "STALE (repredict pending)" if snap.get("prediction_stale") else "DONE"
        )
        if snap.get("prediction_stale"):
            st.warning(
                f"Latest predictions: `{snap['asof']}` from model "
                f"`{snap['model_id']}` — {prediction_status}."
            )
        else:
            st.caption(
                f"Latest predictions: `{snap['asof']}` from model "
                f"`{snap['model_id']}` — {prediction_status}. "
                "Compare to current Alpaca holdings."
            )
        c1, c2, c3 = st.columns(3)
        c1.metric("Keep (in top-10 + held)", len(snap["keepers"]))
        c2.metric("Likely BUY (top-10 not held)", len(snap["new_buys"]))
        c3.metric("Likely SELL (held, not top-10)", len(snap["force_sells"]))
        col_a, col_b = st.columns(2)
        with col_a:
            st.write("**Model top-10**")
            top = snap["top"].copy()
            top["held"] = top["ticker"].isin(snap["held"]).map(
                {True: "✓", False: "—"}
            )
            top["predicted_value"] = top["predicted_value"].map("{:+.4f}".format)
            st.dataframe(top, hide_index=True, width="stretch")
        with col_b:
            st.write("**Force-sell candidates (held but dropped)**")
            if snap["force_sells"]:
                fs_df = pd.DataFrame({"ticker": snap["force_sells"]})
                st.dataframe(fs_df, hide_index=True, width="stretch")
            else:
                st.caption("_None — all held positions still in top-10._")

    st.divider()

    # --- Top section: launchd job heartbeat
    st.subheader("Job heartbeat")
    col1, col2, col3 = st.columns(3)
    last_decide = _last_fire("decide")
    last_stop_loss = _last_fire("stop-loss")
    last_snap = _last_snapshot()

    with col1:
        if last_decide:
            st.metric(
                "Last decide fire",
                _fmt_ts(last_decide["ts"]),
                f"{last_decide['count']} intended orders",
            )
        else:
            st.metric("Last decide fire", "never", "no fires yet")

    with col2:
        if last_stop_loss:
            st.metric(
                "Last stop-loss fire",
                _fmt_ts(last_stop_loss["ts"]),
                f"{last_stop_loss['count']} stop-sells",
            )
        else:
            st.metric("Last stop-loss fire", "never", "rail disabled per config")

    with col3:
        if last_snap:
            st.metric(
                "Last reconcile (snapshot)",
                _fmt_ts(last_snap["ts"]),
                f"asof={last_snap['asof_date']}, equity=${last_snap['equity']:,.2f}",
            )
        else:
            st.metric("Last reconcile (snapshot)", "never", "no fires yet")

    if last_snap is None:
        st.info(
            "**Paper trading hasn't started yet.** Once `sma.live decide` "
            "+ `sma.live reconcile` jobs fire, fill data shows up here. "
            "See the paper-trading go-live runbook in the project specs."
        )
        st.subheader("Live broker state")
        st.markdown(
            "Real-time positions + P&L: "
            "https://app.alpaca.markets/paper/dashboard/overview"
        )
        return

    # --- Equity curve
    st.subheader("Account equity curve (account_snapshots)")
    eq_df = _equity_curve_df()
    if not eq_df.empty:
        long = eq_df.melt(
            id_vars=["asof_date"],
            value_vars=["equity", "cash", "long_market_value"],
            var_name="metric", value_name="dollars",
        )
        fig = px.line(
            long, x="asof_date", y="dollars", color="metric",
            title="Equity / cash / long market value over time",
        )
        st.plotly_chart(fig, width="stretch")
        st.dataframe(eq_df.tail(10), width="stretch")

        # --- Performance vs SPY (the question that matters)
        st.subheader("Performance vs SPY benchmark")
        first_date = eq_df["asof_date"].iloc[0]
        base_equity = eq_df["equity"].iloc[0]
        portfolio = eq_df[["asof_date", "equity"]].copy()
        portfolio["return_pct"] = (portfolio["equity"] / base_equity - 1.0) * 100.0
        portfolio = portfolio.rename(columns={"asof_date": "date"})
        portfolio["metric"] = "Portfolio"
        spy = _bench_returns_df(first_date.date())
        if spy.empty:
            st.info("No SPY price data — can't render benchmark.")
        else:
            spy_chart = spy.rename(columns={"spy_return_pct": "return_pct"})
            spy_chart["metric"] = "SPY"
            combined = pd.concat(
                [portfolio[["date", "return_pct", "metric"]],
                 spy_chart[["date", "return_pct", "metric"]]],
                ignore_index=True,
            )
            fig2 = px.line(
                combined, x="date", y="return_pct", color="metric",
                title="Cumulative return: portfolio vs SPY (% from first snapshot)",
                color_discrete_map={"Portfolio": "#1f77b4", "SPY": "#ff7f0e"},
            )
            fig2.add_hline(y=0, line_dash="dot", line_color="gray")
            st.plotly_chart(fig2, width="stretch")
            # Side-by-side latest reading
            portfolio_latest = portfolio["return_pct"].iloc[-1]
            spy_latest = spy["spy_return_pct"].iloc[-1]
            alpha = portfolio_latest - spy_latest
            c1, c2, c3 = st.columns(3)
            c1.metric("Portfolio", f"{portfolio_latest:+.2f}%")
            c2.metric("SPY", f"{spy_latest:+.2f}%")
            c3.metric(
                "Alpha (portfolio − SPY)", f"{alpha:+.2f}%",
                delta=f"{alpha:+.2f}pp",
                delta_color=("normal" if alpha > 0 else "inverse"),
            )
    else:
        st.info("No account snapshots yet.")

    # --- Drift indicators
    st.subheader("Drift indicators (last 14 days)")
    drift = _drift_summary(days=14)
    col1, col2, col3 = st.columns(3)
    col1.metric("Intended orders submitted", drift["intended"])
    col2.metric(
        "Orders with no fill",
        f"{drift['no_fill']} ({drift['no_fill_pct']:.0%})",
        delta=("⚠️ >20%" if drift["no_fill_pct"] > 0.20 else "OK"),
        delta_color=("inverse" if drift["no_fill_pct"] > 0.20 else "off"),
    )
    col3.metric(
        "Partial fills (<90%)",
        drift["partial"],
        delta=("⚠️" if drift["partial"] > 0 else "OK"),
        delta_color=("inverse" if drift["partial"] > 0 else "off"),
    )

    # --- Recent intended orders
    st.subheader("Intended orders (last 14 days)")
    intended = _recent_intended_orders(days=14)
    if not intended.empty:
        if "status" in intended.columns:
            color_map = {"submitted": "#2ecc71", "submission_failed": "#e74c3c"}
            intended["_color"] = intended["status"].map(color_map).fillna("#7f8c8d")
            display = intended.drop(columns=["_color"])
        else:
            display = intended
        st.dataframe(display, width="stretch", height=300)
    else:
        st.info("No intended orders in the last 14 days.")

    # --- Recent fills
    st.subheader("Paper fills (last 14 days)")
    fills = _recent_fills(days=14)
    if not fills.empty:
        st.dataframe(fills, width="stretch", height=300)
        if "fill_price" in fills.columns and "filled_shares" in fills.columns:
            fills_summary = (
                fills.assign(notional=fills["fill_price"] * fills["filled_shares"])
                     .groupby("asof_date")["notional"].sum().reset_index()
            )
            if not fills_summary.empty:
                fig = px.bar(
                    fills_summary, x="asof_date", y="notional",
                    title="Daily fill notional ($)",
                )
                st.plotly_chart(fig, width="stretch")
    else:
        st.info("No paper fills in the last 14 days.")

    st.subheader("Live broker dashboard")
    st.markdown(
        "Authoritative real-time positions + P&L: "
        "https://app.alpaca.markets/paper/dashboard/overview"
    )
