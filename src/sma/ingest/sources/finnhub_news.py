"""Finnhub company news ingestion.

Pulls the last N days of company news per ticker and writes deduped rows
to the news table. Dedup key is sha256(headline + url + body[:500]).
"""

import hashlib
import time
from collections.abc import Callable
from datetime import date, datetime, timedelta

import finnhub

from sma.ingest.ratelimit import TokenBucket
from sma.ingest.sources._finnhub_retry import (
    DEFAULT_RETRY_SLEEP_BUDGET_S,
    RetrySleepBudget,
    fetch_with_429_retry,
)
from sma.ingest.sources.base import IngestResult
from sma.ingest.store import Store


def _news_hash(headline: str, url: str, body: str | None) -> str:
    body_part = (body or "")[:500]
    raw = f"{headline}|{url}|{body_part}".encode()
    return hashlib.sha256(raw).hexdigest()


class FinnhubNewsSource:
    name = "finnhub_news"

    def __init__(
        self,
        api_key: str,
        lookback_days: int = 7,
        rate_limiter: TokenBucket | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
        retry_sleep_budget_s: float = DEFAULT_RETRY_SLEEP_BUDGET_S,
    ):
        self.api_key = api_key
        self.lookback_days = lookback_days
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
        start = (asof_date - timedelta(days=self.lookback_days)).isoformat()
        end = asof_date.isoformat()

        # One budget per fetch() call: a 429 storm can cost this run at most
        # `retry_sleep_budget_s` of sleeping inside the global writer lock,
        # after which rate-limited tickers are skipped with a warning.
        budget = RetrySleepBudget(self._retry_sleep_budget_s)

        seen_hashes_per_ticker: dict[str, set[str]] = {}
        rows = []
        for t in tickers:
            if self._rate_limiter is not None:
                self._rate_limiter.acquire()
            # Finnhub keys share classes with a DOT (BRK.B) where the rest of the
            # system is yfinance-canonical (BRK-B). Verified against the live API
            # 2026-07-29: company-news BRK-B returns 0 articles, BRK.B returns 32.
            # Passing the raw dashed ticker silently discarded every Berkshire
            # article, leaving BRK-B with 0 news rows all time — the sole cause of
            # `news_per_ticker_minimum` failing nightly. Ask in the vendor's
            # notation; store under the canonical `t` below.
            vendor_symbol = t.replace("-", ".")
            items = fetch_with_429_retry(
                self.name, t,
                lambda vendor_symbol=vendor_symbol: self._client.company_news(
                    vendor_symbol, _from=start, to=end
                ) or [],
                rate_limiter=self._rate_limiter,
                sleep_fn=self._sleep_fn,
                sleep_budget=budget,
            )
            if items is None:
                continue
            seen = seen_hashes_per_ticker.setdefault(t, set())
            for it in items:
                headline = it.get("headline") or ""
                url = it.get("url") or ""
                body = it.get("summary") or ""
                if not headline or not url:
                    continue
                h = _news_hash(headline, url, body)
                if h in seen:
                    continue
                seen.add(h)
                # A missing/invalid Finnhub `datetime` must NOT be coerced to
                # the 1970-01-01 epoch and stored as a real article time.
                try:
                    ts = int(it.get("datetime", 0))
                except (TypeError, ValueError):
                    ts = 0
                if ts <= 0:
                    continue
                published_at = datetime.utcfromtimestamp(ts)
                rows.append((
                    t,
                    published_at,
                    "finnhub",
                    headline,
                    url,
                    body[:500] if body else None,
                    h,
                    run_id,
                ))

        if rows:
            store.conn.executemany(
                "INSERT OR REPLACE INTO news "
                "(ticker, published_at, source, headline, url, body_excerpt, hash, run_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
        return IngestResult(self.name, len(rows), "ok", None)
