"""Finnhub fundamentals + earnings calendar ingestion.

Two endpoints, one source: both come from finnhub and write per-ticker
rows. company_basic_financials returns a metric dict; earnings_calendar
returns upcoming earnings in a date window.
"""

import time
from collections.abc import Callable
from datetime import date, timedelta

import finnhub
from loguru import logger

from sma.ingest.ratelimit import TokenBucket
from sma.ingest.sources._finnhub_earnings_symbols import earnings_reverse_map
from sma.ingest.sources._finnhub_retry import (
    DEFAULT_RETRY_SLEEP_BUDGET_S,
    RetrySleepBudget,
    fetch_with_429_retry,
)
from sma.ingest.sources.base import IngestResult
from sma.ingest.store import Store


class FinnhubFundamentalsSource:
    name = "finnhub_fundamentals"

    def __init__(
        self,
        api_key: str,
        earnings_window_days: int = 30,
        rate_limiter: TokenBucket | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
        retry_sleep_budget_s: float = DEFAULT_RETRY_SLEEP_BUDGET_S,
    ):
        self.api_key = api_key
        self.earnings_window_days = earnings_window_days
        self._client = finnhub.Client(api_key=api_key)
        self._rate_limiter = rate_limiter
        self._sleep_fn = sleep_fn
        self._retry_sleep_budget_s = retry_sleep_budget_s

    def fetch(
        self,
        tickers: list[str],
        asof_date: date,
        store: Store,
        run_id: int,
    ) -> IngestResult:
        # One budget per fetch() call: a 429 storm can cost this run at most
        # `retry_sleep_budget_s` of sleeping inside the global writer lock,
        # after which rate-limited tickers are skipped with a warning.
        budget = RetrySleepBudget(self._retry_sleep_budget_s)

        fund_rows = []
        for t in tickers:
            if self._rate_limiter is not None:
                self._rate_limiter.acquire()
            data = fetch_with_429_retry(
                self.name, t,
                lambda t=t: self._client.company_basic_financials(t, "all") or {},
                rate_limiter=self._rate_limiter,
                sleep_fn=self._sleep_fn,
                sleep_budget=budget,
            )
            if data is None:
                continue
            m = data.get("metric") or {}
            fund_rows.append((
                t,
                asof_date,
                m.get("peNormalizedAnnual"),
                m.get("pbAnnual"),
                m.get("roeRfy"),
                m.get("totalDebt/totalEquityAnnual"),
                m.get("netProfitMarginAnnual"),
                m.get("revenueGrowth5Y"),
                m.get("marketCapitalization"),
                "finnhub",
                run_id,
            ))

        if fund_rows:
            store.conn.executemany(
                "INSERT OR REPLACE INTO fundamentals "
                "(ticker, asof_date, pe, pb, roe, debt_to_equity, profit_margin, "
                " revenue_growth_yoy, market_cap, source, run_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                fund_rows,
            )

        earn_rows = []
        try:
            if self._rate_limiter is not None:
                self._rate_limiter.acquire()
            window_end = asof_date + timedelta(days=self.earnings_window_days)
            cal = self._client.earnings_calendar(
                _from=asof_date.isoformat(),
                to=window_end.isoformat(),
                symbol="",
                international=False,
            ) or {}
            # Finnhub may key a batch response entry under a different symbol
            # than any ticker we requested with -- e.g. verified live
            # 2026-08-16, Berkshire events come back as symbol="BRK.A" for a
            # requested "BRK-B". A plain `sym in ticker_set` (as this used to
            # be) always dropped those rows, leaving BRK-B with zero forward
            # earnings from this source. The reverse map is one-to-many (see
            # _finnhub_earnings_symbols), so one vendor row can legitimately
            # fan out to more than one canonical ticker in the batch.
            reverse_map = earnings_reverse_map(tickers)
            for e in cal.get("earningsCalendar", []) or []:
                sym = e.get("symbol")
                canonical_tickers = reverse_map.get(sym)
                if not canonical_tickers:
                    continue
                report_date_str = e.get("date")
                if not report_date_str:
                    continue
                report_date = date.fromisoformat(report_date_str)
                for canonical in canonical_tickers:
                    earn_rows.append((
                        canonical,
                        report_date,
                        e.get("epsEstimate"),
                        e.get("epsActual"),
                        e.get("revenueEstimate"),
                        e.get("revenueActual"),
                        "finnhub",
                        run_id,
                    ))
        except Exception as e:
            from sma.ingest.sources._finnhub_retry import redact_secrets

            logger.warning("finnhub earnings_calendar failed: {}", redact_secrets(e))

        if earn_rows:
            store.conn.executemany(
                "INSERT OR REPLACE INTO earnings "
                "(ticker, report_date, eps_estimate, eps_actual, revenue_estimate, "
                " revenue_actual, source, run_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                earn_rows,
            )

        total = len(fund_rows) + len(earn_rows)
        return IngestResult(self.name, total, "ok", None)
