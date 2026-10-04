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

import time
from collections.abc import Callable
from datetime import date
from typing import Any

import finnhub
import pandas as pd
import requests
import yfinance as yf
from loguru import logger

from sma.ingest.ratelimit import TokenBucket
from sma.ingest.sources._finnhub_earnings_symbols import earnings_vendor_aliases
from sma.ingest.sources._finnhub_retry import fetch_with_retry
from sma.ingest.store import Store

# The finnhub SDK hardcodes a 10s DEFAULT_TIMEOUT (finnhub.client.Client).
# The 2026-08-03 backfill hit repeated company_earnings read timeouts against
# it (transient Finnhub slowness, not a rate limit) -- 30s gives real
# responses room to land. Client.DEFAULT_TIMEOUT is a class attribute read as
# `self.DEFAULT_TIMEOUT` in `_request`, so setting it on the instance shadows
# the class default for this client only.
CLIENT_TIMEOUT_S = 30


def backfill_earnings(
    api_key: str,
    tickers: list[str],
    store: Store,
    run_id: int,
    quarters: int = 8,
    rate_limiter: TokenBucket | None = None,
    client: Any | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> int:
    """Fetch the last `quarters` quarters of earnings for each ticker.

    Returns the number of rows persisted (after dedup via INSERT OR REPLACE).
    """
    if client is None:
        client = finnhub.Client(api_key=api_key)
        client.DEFAULT_TIMEOUT = CLIENT_TIMEOUT_S

    rows: list[tuple] = []
    for ticker in tickers:
        if rate_limiter is not None:
            rate_limiter.acquire()
        # Finnhub uses dot share-class notation (BRK.B) where the universe is
        # yfinance-canonical dash notation (BRK-B) -- same translation
        # alpaca_prices/alpaca_news/finnhub_news already do. Query in vendor
        # notation; storage below always uses the canonical `ticker`, not
        # whatever symbol the response echoes back (see comment there).
        vendor_symbol = earnings_vendor_aliases(ticker)[0]
        # One retry on a read timeout only (transient Finnhub slowness, not a
        # rate limit) -- mirrors the finnhub_news/finnhub_fundamentals 429
        # retry-in-place pattern. Other exceptions (incl. a 429
        # FinnhubAPIException, which the `--provider both` CLI path relies on
        # falling straight through to the yfinance gap-fill) are not retried.
        items = fetch_with_retry(
            "company_earnings", ticker,
            lambda vendor_symbol=vendor_symbol: client.company_earnings(
                symbol=vendor_symbol, limit=quarters
            ) or [],
            is_retryable=lambda e: isinstance(e, requests.exceptions.ReadTimeout),
            reason="read timeout",
            rate_limiter=rate_limiter,
            sleep_fn=sleep_fn,
            max_retries=1,
        )
        if items is None:
            continue

        for item in items:
            # Regression (verified live 2026-08-16): Finnhub reports
            # company-level earnings under ITS OWN symbol regardless of query
            # spelling -- company_earnings("BRK-B") and company_earnings
            # ("BRK.B") both return rows with symbol="BRK.A". Trusting
            # `item.get("symbol")` stored 4 rows under ticker='BRK.A', a
            # symbol outside the universe (canonical is BRK-B), so nothing
            # downstream ever read them. This call is per-ticker, so there is
            # no batch ambiguity about which canonical ticker an item
            # belongs to -- always store the canonical `ticker` we requested.
            symbol = ticker
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
