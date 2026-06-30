"""Cached read-only accessors for data/sma.duckdb.

Each function opens its own connection and closes it when done. Read-only
mode prevents accidental writes during interactive sessions.
"""

from pathlib import Path

import duckdb
import pandas as pd
import streamlit as st

from sma.db_connect import read_only_connect

DB_PATH = Path("data/sma.duckdb")


def read_only_conn() -> duckdb.DuckDBPyConnection:
    """Lock-tolerant read-only connection to the vault DB.

    Use everywhere in the dashboard instead of a raw
    `duckdb.connect(read_only=True)`: DuckDB's file lock makes a raw read-only
    open CRASH while a scheduled writer (ingest/predict/agents/decide/reconcile)
    holds the lock. `read_only_connect` retries with capped backoff instead.
    """
    return read_only_connect(DB_PATH)


def _conn() -> duckdb.DuckDBPyConnection:
    if not DB_PATH.exists():
        st.error(
            f"DuckDB not found at {DB_PATH}. "
            "Run `uv run python -m sma.ingest run` to populate."
        )
        st.stop()
    return read_only_conn()


@st.cache_data(ttl=60)
def list_tables() -> list[tuple[str, int]]:
    con = _conn()
    try:
        rows = []
        for (t,) in con.execute("SHOW TABLES").fetchall():
            n = con.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
            rows.append((t, int(n)))
        return rows
    finally:
        con.close()


@st.cache_data(ttl=60)
def coverage_summary() -> pd.DataFrame:
    """Per-table row count, date range, ticker count, source count where applicable."""
    con = _conn()
    try:
        rows = []

        # prices
        r = con.execute(
            "SELECT COUNT(*), MIN(date), MAX(date),"
            " COUNT(DISTINCT ticker), COUNT(DISTINCT source) FROM prices"
        ).fetchone()
        rows.append(
            {
                "table": "prices",
                "rows": r[0],
                "date_min": str(r[1]),
                "date_max": str(r[2]),
                "tickers": r[3],
                "sources": r[4],
            }
        )

        # news
        r = con.execute(
            "SELECT COUNT(*), MIN(published_at), MAX(published_at),"
            " COUNT(DISTINCT ticker), COUNT(DISTINCT source) FROM news"
        ).fetchone()
        rows.append(
            {
                "table": "news",
                "rows": r[0],
                "date_min": str(r[1])[:10] if r[1] else "N/A",
                "date_max": str(r[2])[:10] if r[2] else "N/A",
                "tickers": r[3],
                "sources": r[4],
            }
        )

        # other tables: just row counts
        for tname in ("filings", "fundamentals", "earnings", "ingest_log"):
            n = con.execute(f'SELECT COUNT(*) FROM "{tname}"').fetchone()[0]
            rows.append(
                {
                    "table": tname,
                    "rows": n,
                    "date_min": "N/A",
                    "date_max": "N/A",
                    "tickers": "N/A",
                    "sources": "N/A",
                }
            )

        df = pd.DataFrame(rows)
        # Arrow can't serialize a column with mixed int + str values. The
        # filings/fundamentals/earnings/ingest_log rows leave tickers/sources
        # as the string "N/A" while prices/news leave them as ints. Normalize.
        for col in ("tickers", "sources"):
            df[col] = df[col].astype(str)
        return df
    finally:
        con.close()


@st.cache_data(ttl=60)
def ingest_log(limit: int = 200) -> pd.DataFrame:
    con = _conn()
    try:
        return con.execute(
            f"SELECT run_id, source, status, rows_inserted, started_at, finished_at, error "
            f"FROM ingest_log ORDER BY started_at DESC LIMIT {limit}"
        ).fetchdf()
    finally:
        con.close()


@st.cache_data(ttl=60)
def ingest_log_for_chart(limit: int = 30) -> pd.DataFrame:
    """Last N ingest runs with all columns needed for the bar chart."""
    con = _conn()
    try:
        return con.execute(
            f"""
            SELECT run_id, source, status, rows_inserted, started_at
            FROM ingest_log
            ORDER BY started_at DESC
            LIMIT {limit}
            """
        ).fetchdf()
    finally:
        con.close()


@st.cache_data(ttl=60)
def list_tickers() -> list[str]:
    con = _conn()
    try:
        return [
            r[0]
            for r in con.execute(
                "SELECT DISTINCT ticker FROM prices ORDER BY ticker"
            ).fetchall()
        ]
    finally:
        con.close()


@st.cache_data(ttl=60)
def ohlcv(ticker: str, source: str = "yfinance") -> pd.DataFrame:
    """Full OHLCV history for a ticker/source combination."""
    con = _conn()
    try:
        df = con.execute(
            """
            SELECT date, open, high, low, close, adj_close, volume
            FROM prices
            WHERE ticker = $ticker AND source = $source
            ORDER BY date
            """,
            {"ticker": ticker, "source": source},
        ).fetchdf()
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
        return df
    finally:
        con.close()


@st.cache_data(ttl=60)
def news_for_ticker(ticker: str, limit: int = 100) -> pd.DataFrame:
    """Recent news for a single ticker, newest first."""
    con = _conn()
    try:
        return con.execute(
            """
            SELECT ticker, published_at, headline, url, source, body_excerpt
            FROM news
            WHERE ticker = $ticker
            ORDER BY published_at DESC
            LIMIT $limit
            """,
            {"ticker": ticker, "limit": limit},
        ).fetchdf()
    finally:
        con.close()


@st.cache_data(ttl=60)
def all_news(limit: int = 500) -> pd.DataFrame:
    con = _conn()
    try:
        return con.execute(
            f"""
            SELECT ticker, published_at, headline, url, source, body_excerpt
            FROM news
            ORDER BY published_at DESC
            LIMIT {limit}
            """
        ).fetchdf()
    finally:
        con.close()


@st.cache_data(ttl=60)
def news_filtered(
    tickers: list[str] | None,
    sources: list[str] | None,
    date_start: str | None,
    date_end: str | None,
    limit: int = 200,
) -> tuple[pd.DataFrame, int]:
    """Filtered news; returns (page_df, total_count)."""
    con = _conn()
    try:
        clauses = []
        params: dict = {}

        if tickers:
            clauses.append("ticker = ANY($tickers)")
            params["tickers"] = tickers
        if sources:
            clauses.append("source = ANY($sources)")
            params["sources"] = sources
        if date_start:
            clauses.append("published_at >= CAST($date_start AS DATE)")
            params["date_start"] = date_start
        if date_end:
            clauses.append("published_at < CAST($date_end AS DATE) + INTERVAL 1 DAY")
            params["date_end"] = date_end

        where = "WHERE " + " AND ".join(clauses) if clauses else ""

        total = con.execute(
            f"SELECT COUNT(*) FROM news {where}", params
        ).fetchone()[0]

        df = con.execute(
            f"""
            SELECT ticker, published_at, headline, url, source, body_excerpt
            FROM news
            {where}
            ORDER BY published_at DESC
            LIMIT {limit}
            """,
            params,
        ).fetchdf()

        return df, int(total)
    finally:
        con.close()


@st.cache_data(ttl=60)
def distinct_news_sources() -> list[str]:
    con = _conn()
    try:
        return [
            r[0]
            for r in con.execute(
                "SELECT DISTINCT source FROM news ORDER BY source"
            ).fetchall()
        ]
    finally:
        con.close()


@st.cache_data(ttl=60)
def news_tickers() -> list[str]:
    con = _conn()
    try:
        return [
            r[0]
            for r in con.execute(
                "SELECT DISTINCT ticker FROM news ORDER BY ticker"
            ).fetchall()
        ]
    finally:
        con.close()
