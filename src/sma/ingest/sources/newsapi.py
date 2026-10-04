"""NewsAPI ingestion.

Free tier: 100 requests/day. We make one request per ticker per day, so the
universe must stay under 100 tickers. Rate-limit responses (HTTP 429) are
returned as status='rate_limited' so the runner can skip the source for the
rest of the day without crashing.
"""

import hashlib
from datetime import UTC, date, datetime, timedelta

import httpx
from loguru import logger

from sma.ingest.sources._finnhub_retry import redact_secrets
from sma.ingest.sources.base import IngestResult
from sma.ingest.store import Store


def _news_hash(headline: str, url: str, body: str | None) -> str:
    body_part = (body or "")[:500]
    return hashlib.sha256(f"{headline}|{url}|{body_part}".encode()).hexdigest()


class NewsAPISource:
    name = "newsapi"
    BASE_URL = "https://newsapi.org/v2/everything"

    def __init__(self, api_key: str, lookback_days: int = 7):
        self.api_key = api_key
        self.lookback_days = lookback_days
        self._client = httpx.Client(timeout=20.0)

    def fetch(
        self,
        tickers: list[str],
        asof_date: date,
        store: Store,
        run_id: int,
    ) -> IngestResult:
        from_date = (asof_date - timedelta(days=self.lookback_days)).isoformat()
        rows = []
        for t in tickers:
            try:
                resp = self._client.get(
                    self.BASE_URL,
                    params={
                        "q": t,
                        "from": from_date,
                        "language": "en",
                        "sortBy": "publishedAt",
                        "pageSize": 50,
                        "apiKey": self.api_key,
                    },
                )
            except Exception as e:
                # apiKey is a query param, and httpx quotes the URL in its errors.
                logger.warning("newsapi request failed for {}: {}", t, redact_secrets(e))
                continue

            if resp.status_code == 429:
                logger.warning("newsapi rate limited; aborting source for this run")
                return IngestResult(self.name, len(rows), "rate_limited", "HTTP 429")
            if resp.status_code != 200:
                logger.warning("newsapi {} returned {}: {}",
                               t, resp.status_code, resp.text[:200])
                continue

            data = resp.json()
            for art in data.get("articles") or []:
                title = art.get("title") or ""
                url = art.get("url") or ""
                desc = art.get("description") or ""
                published_at_str = art.get("publishedAt")
                if not (title and url and published_at_str):
                    continue
                try:
                    published_at = (
                        datetime.fromisoformat(published_at_str.replace("Z", "+00:00"))
                        .astimezone(UTC)  # normalize a non-UTC offset to UTC first
                        .replace(tzinfo=None)
                    )
                except ValueError:
                    continue
                h = _news_hash(title, url, desc)
                rows.append((
                    t, published_at, "newsapi", title, url,
                    desc[:500] if desc else None, h, run_id,
                ))

        if rows:
            store.conn.executemany(
                "INSERT OR REPLACE INTO news "
                "(ticker, published_at, source, headline, url, body_excerpt, hash, run_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
        return IngestResult(self.name, len(rows), "ok", None)
