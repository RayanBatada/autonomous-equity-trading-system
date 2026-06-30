"""Historical earnings backfill: Finnhub primary, yfinance fallback.

Daily ingest fetches FORWARD-looking earnings via earnings_calendar.
This module fetches HISTORICAL earnings (past N quarters) so the
earnings_blackout rail has data to score against.

Two providers:
- `backfill_earnings`: Finnhub's company_earnings endpoint. Capped at 4
  quarters on the free tier and bound by daily quota.
- `backfill_earnings_yfinance`: Yahoo Finance scrape via yfinance's
  Ticker.get_earnings_dates. No quota, but HTML-scraped so individual
  ticker calls can fail without warning.

Both write `eps_estimate` and `eps_actual`; revenue columns are NULL
because neither source surfaces them in this code path. The Phase 3
blackout rail only needs `report_date`, so this is sufficient.
"""

from collections.abc import Callable
from datetime import date
from typing import Any

import finnhub
import pandas as pd
import yfinance as yf
from loguru import logger

from sma.ingest.ratelimit import TokenBucket
from sma.ingest.store import Store


def backfill_earnings(
    api_key: str,
    tickers: list[str],
    store: Store,
    run_id: int,
    quarters: int = 8,
    rate_limiter: TokenBucket | None = None,
    client: Any | None = None,
) -> int:
    """Fetch the last `quarters` quarters of earnings for each ticker.

    Returns the number of rows persisted (after dedup via INSERT OR REPLACE).
    """
    if client is None:
        client = finnhub.Client(api_key=api_key)

    rows: list[tuple] = []
    for ticker in tickers:
        if rate_limiter is not None:
            rate_limiter.acquire()
        try:
            items = client.company_earnings(symbol=ticker, limit=quarters) or []
        except Exception as e:
            logger.warning("company_earnings failed for {}: {}", ticker, e)
            continue

        for item in items:
            symbol = item.get("symbol") or ticker
            period_str = item.get("period")
            if not period_str:
                logger.warning(
                    "skipping earnings row with missing period for {}: {}",
                    ticker, item,
                )
                continue
            try:
                report_date = date.fromisoformat(period_str)
            except (ValueError, TypeError) as e:
                logger.warning(
                    "skipping earnings row with invalid period {!r} for {}: {}",
                    period_str, ticker, e,
                )
                continue
            rows.append((
                symbol,
                report_date,
                item.get("estimate"),
                item.get("actual"),
                None,
                None,
                "finnhub_earnings_backfill",
                run_id,
            ))

    if rows:
        store.conn.executemany(
            "INSERT OR REPLACE INTO earnings "
            "(ticker, report_date, eps_estimate, eps_actual, revenue_estimate, "
            " revenue_actual, source, run_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )

    return len(rows)


def backfill_earnings_yfinance(
    tickers: list[str],
    store: Store,
    run_id: int,
    quarters: int = 8,
    ticker_factory: Callable[[str], Any] | None = None,
    only_missing: bool = False,
) -> int:
    """Fetch the last `quarters` quarters of earnings via Yahoo Finance.

    Uses yfinance's Ticker.get_earnings_dates which scrapes Yahoo's
    earnings page. No API key, no quota, but per-ticker calls can fail
    on transient HTTP errors or page-layout changes; failures are caught
    per-ticker so one bad ticker doesn't kill the run.

    Args:
        tickers: full ticker list.
        store: connected DuckDB Store.
        run_id: allocated run id (logged on each row).
        quarters: number of past reported quarters to keep per ticker.
        ticker_factory: yf.Ticker by default; injectable for tests.
        only_missing: if True, skip tickers that already have any rows
            in the earnings table (use for filling Finnhub gaps without
            overwriting Finnhub-sourced rows).

    Returns the number of rows persisted (after dedup via INSERT OR REPLACE).
    """
    if ticker_factory is None:
        ticker_factory = yf.Ticker

    if only_missing:
        existing = {
            r[0] for r in store.conn.execute(
                "SELECT DISTINCT ticker FROM earnings"
            ).fetchall()
        }
        tickers = [t for t in tickers if t not in existing]

    rows: list[tuple] = []
    for ticker in tickers:
        try:
            t_obj = ticker_factory(ticker)
            df = t_obj.get_earnings_dates(limit=max(quarters * 2, 12))
        except Exception as e:
            logger.warning("yfinance earnings_dates failed for {}: {}", ticker, e)
            continue

        if df is None or len(df) == 0:
            continue

        # Drop future rows that haven't reported yet (NaN actual).
        if "Reported EPS" in df.columns:
            df = df[df["Reported EPS"].notna()]
        else:
            logger.warning(
                "yfinance returned unexpected schema for {}: cols={}",
                ticker, list(df.columns),
            )
            continue

        df = df.head(quarters)

        for ts, row in df.iterrows():
            try:
                report_date = ts.date() if hasattr(ts, "date") else ts
            except Exception as e:
                logger.warning(
                    "skipping invalid earnings date for {}: {}", ticker, e,
                )
                continue

            est = row.get("EPS Estimate")
            act = row.get("Reported EPS")
            est_clean = (
                None if est is None or pd.isna(est) else float(est)
            )
            act_clean = (
                None if act is None or pd.isna(act) else float(act)
            )
            rows.append((
                ticker,
                report_date,
                est_clean,
                act_clean,
                None,
                None,
                "yfinance_earnings_backfill",
                run_id,
            ))

    if rows:
        store.conn.executemany(
            "INSERT OR REPLACE INTO earnings "
            "(ticker, report_date, eps_estimate, eps_actual, revenue_estimate, "
            " revenue_actual, source, run_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )

    return len(rows)
