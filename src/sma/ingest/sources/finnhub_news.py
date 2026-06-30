"""Finnhub company news ingestion.

Pulls the last N days of company news per ticker and writes deduped rows
to the news table. Dedup key is sha256(headline + url + body[:500]).
"""

import hashlib
from datetime import date, datetime, timedelta

import finnhub
from loguru import logger

from sma.ingest.ratelimit import TokenBucket
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
    ):
        self.api_key = api_key
        self.lookback_days = lookback_days
        self._client = finnhub.Client(api_key=api_key)
        self._rate_limiter = rate_limiter

    def fetch(
        self,
        tickers: list[str],
        asof_date: date,
        store: Store,
        run_id: int,
    ) -> IngestResult:
        start = (asof_date - timedelta(days=self.lookback_days)).isoformat()
        end = asof_date.isoformat()

        seen_hashes_per_ticker: dict[str, set[str]] = {}
        rows = []
        for t in tickers:
            if self._rate_limiter is not None:
                self._rate_limiter.acquire()
            try:
                items = self._client.company_news(t, _from=start, to=end) or []
            except Exception as e:
                logger.warning("finnhub_news failed for {}: {}", t, e)
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
