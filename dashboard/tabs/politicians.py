"""Politicians tab — Senate + House trade activity feeding politician_flow_30d.

Surfaces:
- Recent trades (last 30 days, both chambers, sortable)
- Top tickers by 30-day and 90-day net flow (the same windowing the
  politician_flow_30d feature uses, plus a longer-horizon view)
- Top politicians by trade count + dollar volume
- Coverage stats: row counts per chamber, parse status, freshness
- Direct query into a single ticker's politician trade history

The data source is the `politician_trades` table populated by
`sma.ingest.sources.politician_trades` (House, FD.zip + PDF parse) and
`sma.ingest.sources.senate_trades` (Senate, EFD HTML scrape). Both
schedules fire Sundays — Senate 10:00 ET, House 11:00 ET — and the
politician_flow_30d feature picks up the fresh data in Monday's retrain.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import streamlit as st

from dashboard.data import DB_PATH
from sma.ingest.store import Store


def render() -> None:
    st.header("Politicians")
    st.caption(
        "Senate + House periodic transaction reports (PTRs) feeding the "
        "`politician_flow_30d` model feature. Senate refreshes Sun 10:00 ET, "
        "House refreshes Sun 11:00 ET."
    )

    if not DB_PATH.exists():
        st.error(f"DuckDB not found at {DB_PATH}.")
        return
    store = Store(path=DB_PATH).connect(read_only=True)
    conn = store.conn

    _render_coverage(conn)
    st.divider()
    _render_top_tickers_by_flow(conn)
    st.divider()
    _render_top_politicians(conn)
    st.divider()
    _render_recent_trades(conn)
    st.divider()
    _render_ticker_lookup(conn)


def _render_coverage(conn) -> None:
    st.subheader("Coverage")
    rows = conn.execute(
        """
        SELECT chamber,
               COUNT(*) AS trades,
               COUNT(DISTINCT doc_id) AS docs,
               COUNT(DISTINCT ticker) AS tickers,
               MIN(transaction_date) AS earliest_trade,
               MAX(transaction_date) AS latest_trade,
               MAX(filing_date) AS latest_filing
        FROM politician_trades
        WHERE transaction_date IS NOT NULL
        GROUP BY chamber
        ORDER BY chamber
        """,
    ).fetchall()
    if not rows:
        st.info("No politician_trades data yet.")
        return
    df = pd.DataFrame(
        rows,
        columns=[
            "chamber", "trades", "docs", "tickers",
            "earliest_trade", "latest_trade", "latest_filing",
        ],
    )
    st.dataframe(df, hide_index=True, width="stretch")


def _net_flow_sql(window_days: int) -> str:
    return f"""
        WITH cutoff AS (SELECT (CURRENT_DATE - INTERVAL '{window_days} days') AS d)
        SELECT
            ticker,
            COUNT(*) AS trades,
            COUNT(DISTINCT last_name) AS politicians,
            SUM(CASE
                WHEN transaction_type = 'P'
                    THEN (amount_min + COALESCE(amount_max, amount_min)) / 2
                WHEN transaction_type LIKE 'S%'
                    THEN -1 * (amount_min + COALESCE(amount_max, amount_min)) / 2
                ELSE 0
            END) AS net_flow
        FROM politician_trades, cutoff
        WHERE ticker IS NOT NULL
          -- window on filing_date (disclosure), matching the model's
          -- politician_flow_30d feature — NOT transaction_date (2026-06-05 audit).
          AND filing_date >= cutoff.d
        GROUP BY ticker
        ORDER BY ABS(net_flow) DESC
        LIMIT 20
    """


def _render_top_tickers_by_flow(conn) -> None:
    st.subheader("Top tickers by net politician flow")
    st.caption(
        "Buys add midpoint of disclosed range; sells subtract it. "
        "Sorted by |net_flow| so both inflows (PRIORITY for the model) "
        "and outflows (signal to trim) surface."
    )
    col30, col90 = st.columns(2)
    with col30:
        st.markdown("**Last 30 days** (matches `politician_flow_30d` feature window)")
        rows = conn.execute(_net_flow_sql(30)).fetchall()
        if rows:
            df = pd.DataFrame(
                rows, columns=["ticker", "trades", "politicians", "net_flow"],
            )
            df["net_flow"] = df["net_flow"].astype(float)
            st.dataframe(df, hide_index=True, width="stretch")
        else:
            st.info("No trades in the last 30 days.")
    with col90:
        st.markdown("**Last 90 days** (longer horizon to spot accumulation)")
        rows = conn.execute(_net_flow_sql(90)).fetchall()
        if rows:
            df = pd.DataFrame(
                rows, columns=["ticker", "trades", "politicians", "net_flow"],
            )
            df["net_flow"] = df["net_flow"].astype(float)
            st.dataframe(df, hide_index=True, width="stretch")
        else:
            st.info("No trades in the last 90 days.")


def _render_top_politicians(conn) -> None:
    st.subheader("Most-active politicians (last 90 days)")
    rows = conn.execute(
        """
        SELECT
            chamber,
            COALESCE(first_name || ' ' || last_name, last_name) AS name,
            state_dst AS state,
            COUNT(*) AS trades,
            COUNT(DISTINCT ticker) AS distinct_tickers,
            SUM((amount_min + COALESCE(amount_max, amount_min)) / 2) AS gross_volume
        FROM politician_trades
        WHERE transaction_date >= (CURRENT_DATE - INTERVAL '90 days')
          AND last_name IS NOT NULL
        GROUP BY chamber, name, state
        ORDER BY trades DESC
        LIMIT 15
        """,
    ).fetchall()
    if not rows:
        st.info("No politician activity in the last 90 days.")
        return
    df = pd.DataFrame(
        rows,
        columns=["chamber", "name", "state", "trades", "distinct_tickers", "gross_volume"],
    )
    df["gross_volume"] = df["gross_volume"].astype(float)
    st.dataframe(df, hide_index=True, width="stretch")


def _render_recent_trades(conn) -> None:
    st.subheader("Recent trades (last 30 days)")
    chamber = st.radio(
        "Chamber",
        ["Both", "house", "senate"],
        horizontal=True,
        key="politicians_chamber",
    )
    where_chamber = ""
    if chamber != "Both":
        where_chamber = f"AND chamber = '{chamber}'"
    rows = conn.execute(
        f"""
        SELECT
            transaction_date AS trade_date,
            chamber,
            COALESCE(first_name || ' ' || last_name, last_name) AS politician,
            state_dst AS state,
            ticker,
            transaction_type AS type,
            amount_min,
            amount_max,
            filing_date
        FROM politician_trades
        WHERE transaction_date >= (CURRENT_DATE - INTERVAL '30 days')
          AND ticker IS NOT NULL
          {where_chamber}
        ORDER BY transaction_date DESC, filing_date DESC
        LIMIT 200
        """,
    ).fetchall()
    if not rows:
        st.info("No trades in the last 30 days for this chamber filter.")
        return
    df = pd.DataFrame(
        rows,
        columns=[
            "trade_date", "chamber", "politician", "state", "ticker",
            "type", "amount_min", "amount_max", "filing_date",
        ],
    )
    st.dataframe(df, hide_index=True, width="stretch", height=420)


def _render_ticker_lookup(conn) -> None:
    st.subheader("Ticker lookup — full politician history")
    with st.form("politician_ticker_lookup"):
        ticker = st.text_input(
            "Ticker", value="NVDA", help="Uppercase, e.g. NVDA / MSFT / GOOGL"
        ).strip().upper()
        submitted = st.form_submit_button("Search")
    if not submitted:
        return
    if not ticker:
        st.warning("Enter a ticker.")
        return
    rows = conn.execute(
        """
        SELECT
            transaction_date AS trade_date,
            chamber,
            COALESCE(first_name || ' ' || last_name, last_name) AS politician,
            state_dst AS state,
            transaction_type AS type,
            amount_min,
            amount_max,
            filing_date
        FROM politician_trades
        WHERE ticker = ?
        ORDER BY transaction_date DESC, filing_date DESC
        LIMIT 200
        """,
        [ticker],
    ).fetchall()
    if not rows:
        st.info(f"No politician trade rows for {ticker}.")
        return
    df = pd.DataFrame(
        rows,
        columns=[
            "trade_date", "chamber", "politician", "state",
            "type", "amount_min", "amount_max", "filing_date",
        ],
    )
    st.dataframe(df, hide_index=True, width="stretch", height=420)

    # 30-day net flow for this ticker
    asof = date.today()
    cutoff = asof - timedelta(days=30)
    # Window on filing_date (disclosure) and use the model's exact sign rule:
    # P=+1, S%=-1, everything else (e.g. 'E' exchange) = 0 (2026-06-05 audit —
    # previously windowed on trade_date and counted non-sells as buys).
    sub = df[df["filing_date"] >= cutoff]
    if not sub.empty:
        midpoint = (
            sub["amount_min"].astype(float)
            + sub["amount_max"].fillna(sub["amount_min"]).astype(float)
        ) / 2.0
        sign = sub["type"].astype(str).map(
            lambda t: 1.0 if t == "P" else (-1.0 if t.startswith("S") else 0.0),
        )
        net_flow_30d = float((midpoint * sign).sum())
        st.metric(
            f"{ticker} net politician flow (last 30 days)",
            f"${net_flow_30d:,.0f}",
            help="Same window the politician_flow_30d model feature uses.",
        )
