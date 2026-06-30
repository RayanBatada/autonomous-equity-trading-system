"""Alpaca News ingestion.

Pulls articles from Alpaca's News API (https://data.alpaca.markets/v1beta1/news),
which is free on paper-account credentials. Articles are sourced primarily from
Benzinga and tagged with one or more tickers via the `symbols[]` field.

Cross-ticker emission: a single article can mention multiple symbols
(e.g. an "AAPL/AMZN earnings" piece). For each ticker in the requested batch
that also appears in the article's `symbols[]`, we emit a separate news row,
so each ticker can find the article in its own news view.

Free-tier rate limit: 200 requests/minute (well above ingest needs).
"""

import hashlib
from datetime import UTC, date, datetime, timedelta

import httpx
from loguru import logger

from sma.ingest.ratelimit import TokenBucket
from sma.ingest.sources.base import IngestResult
from sma.ingest.store import Store


def _news_hash(headline: str, url: str, body: str | None) -> str:
    body_part = (body or "")[:500]
    return hashlib.sha256(f"{headline}|{url}|{body_part}".encode()).hexdigest()


class AlpacaNewsSource:
    name = "alpaca_news"
    BASE_URL = "https://data.alpaca.markets/v1beta1/news"

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        lookback_days: int = 7,
        rate_limiter: TokenBucket | None = None,
        batch_size: int = 10,
        page_limit: int = 50,
    ):
        if not api_key or not api_secret:
            raise ValueError(
                "ALPACA_API_KEY/SECRET required for alpaca_news source"
            )
        self.api_key = api_key
        self.api_secret = api_secret
        self.lookback_days = lookback_days
        self._rate_limiter = rate_limiter
        self.batch_size = batch_size
        self.page_limit = page_limit
        self._client = httpx.Client(
            timeout=20.0,
            headers={
                "APCA-API-KEY-ID": api_key,
                "APCA-API-SECRET-KEY": api_secret,
            },
        )

    def fetch(
        self,
        tickers: list[str],
        asof_date: date,
        store: Store,
        run_id: int,
    ) -> IngestResult:
        start_dt = (
            datetime.combine(asof_date - timedelta(days=self.lookback_days),
                             datetime.min.time())
            .isoformat() + "Z"
        )
        end_dt = (
            datetime.combine(asof_date + timedelta(days=1),
                             datetime.min.time())
            .isoformat() + "Z"
        )

        rows: list[tuple] = []
        seen: set[tuple[str, str]] = set()

        for i in range(0, len(tickers), self.batch_size):
            batch = tickers[i:i + self.batch_size]
            batch_set = set(batch)
            symbols_csv = ",".join(batch)
            page_token: str | None = None

            while True:
                params = {
                    "symbols": symbols_csv,
                    "start": start_dt,
                    "end": end_dt,
                    "limit": self.page_limit,
                    "sort": "desc",
                    "include_content": "false",
                    "exclude_contentless": "false",
                }
                if page_token:
                    params["page_token"] = page_token

                if self._rate_limiter is not None:
                    self._rate_limiter.acquire()

                try:
                    resp = self._client.get(self.BASE_URL, params=params)
                except Exception as e:
                    logger.warning(
                        "alpaca_news request failed for batch {}: {}", batch, e
                    )
                    break

                if resp.status_code == 429:
                    logger.warning(
                        "alpaca_news rate limited; aborting source for this run"
                    )
                    if rows:
                        store.conn.executemany(
                            "INSERT OR REPLACE INTO news "
                            "(ticker, published_at, source, headline, url, "
                            "body_excerpt, hash, run_id) "
                            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                            rows,
                        )
                    return IngestResult(
                        self.name, len(rows), "rate_limited", "HTTP 429"
                    )
                if resp.status_code != 200:
                    logger.warning(
                        "alpaca_news batch {} returned {}: {}",
                        batch, resp.status_code, resp.text[:200],
                    )
                    break

                data = resp.json()
                articles = data.get("news") or []
                for art in articles:
                    headline = art.get("headline") or ""
                    url = art.get("url") or ""
                    summary = art.get("summary") or ""
                    created_at_str = art.get("created_at")
                    art_symbols = art.get("symbols") or []
                    if not (headline and url and created_at_str):
                        continue
                    try:
                        published_at = (
                            datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
                            .astimezone(UTC)  # normalize a non-UTC offset to UTC first
                            .replace(tzinfo=None)
                        )
                    except ValueError:
                        continue
                    h = _news_hash(headline, url, summary)
                    src_label = (
                        f"alpaca:{art['source']}"
                        if art.get("source")
                        else "alpaca"
                    )
                    body_excerpt = summary[:500] if summary else None
                    matching = [s for s in art_symbols if s in batch_set]
                    for ticker in matching:
                        key = (ticker, h)
                        if key in seen:
                            continue
                        seen.add(key)
                        rows.append((
                            ticker,
                            published_at,
                            src_label,
                            headline,
                            url,
                            body_excerpt,
                            h,
                            run_id,
                        ))

                page_token = data.get("next_page_token")
                if not page_token:
                    break

        if rows:
            store.conn.executemany(
                "INSERT OR REPLACE INTO news "
                "(ticker, published_at, source, headline, url, body_excerpt, "
                "hash, run_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
        return IngestResult(self.name, len(rows), "ok", None)
