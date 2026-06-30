"""Finnhub fundamentals + earnings calendar ingestion.

Two endpoints, one source: both come from finnhub and write per-ticker
rows. company_basic_financials returns a metric dict; earnings_calendar
returns upcoming earnings in a date window.
"""

from datetime import date, timedelta

import finnhub
from loguru import logger

from sma.ingest.ratelimit import TokenBucket
from sma.ingest.sources.base import IngestResult
from sma.ingest.store import Store


class FinnhubFundamentalsSource:
    name = "finnhub_fundamentals"

    def __init__(
        self,
        api_key: str,
        earnings_window_days: int = 30,
        rate_limiter: TokenBucket | None = None,
    ):
        self.api_key = api_key
        self.earnings_window_days = earnings_window_days
        self._client = finnhub.Client(api_key=api_key)
        self._rate_limiter = rate_limiter

    def fetch(
        self,
        tickers: list[str],
        asof_date: date,
        store: Store,
        run_id: int,
    ) -> IngestResult:
        fund_rows = []
        for t in tickers:
            if self._rate_limiter is not None:
                self._rate_limiter.acquire()
            try:
                data = self._client.company_basic_financials(t, "all") or {}
            except Exception as e:
                logger.warning("finnhub_fundamentals failed for {}: {}", t, e)
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
            ticker_set = set(tickers)
            for e in cal.get("earningsCalendar", []) or []:
                sym = e.get("symbol")
                if sym not in ticker_set:
                    continue
                report_date_str = e.get("date")
                if not report_date_str:
                    continue
                report_date = date.fromisoformat(report_date_str)
                earn_rows.append((
                    sym,
                    report_date,
                    e.get("epsEstimate"),
                    e.get("epsActual"),
                    e.get("revenueEstimate"),
                    e.get("revenueActual"),
                    "finnhub",
                    run_id,
                ))
        except Exception as e:
            logger.warning("finnhub earnings_calendar failed: {}", e)

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
