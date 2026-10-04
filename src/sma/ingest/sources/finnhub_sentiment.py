"""Finnhub news-sentiment ingestion.

Stores a single sentiment score per (ticker, asof_date), where the score is
bullishPercent - bearishPercent (range -1..1).
"""

from datetime import date

import finnhub
from loguru import logger

from sma.ingest.sources._finnhub_retry import redact_secrets
from sma.ingest.sources.base import IngestResult
from sma.ingest.store import Store


class FinnhubSentimentSource:
    name = "finnhub_sentiment"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self._client = finnhub.Client(api_key=api_key)

    def fetch(
        self,
        tickers: list[str],
        asof_date: date,
        store: Store,
        run_id: int,
    ) -> IngestResult:
        rows = []
        for t in tickers:
            try:
                resp = self._client.news_sentiment(t) or {}
            except Exception as e:
                logger.warning("finnhub_sentiment failed for {}: {}", t, redact_secrets(e))
                continue
            sent = resp.get("sentiment") or {}
            buzz = resp.get("buzz") or {}
            if not sent:
                continue
            score = float(sent.get("bullishPercent", 0)) - float(sent.get("bearishPercent", 0))
            num_articles = int(buzz.get("articlesInLastWeek") or 0)
            rows.append((t, asof_date, score, "finnhub", num_articles, run_id))

        if rows:
            store.conn.executemany(
                "INSERT OR REPLACE INTO sentiment "
                "(ticker, date, score, source, num_articles, run_id) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                rows,
            )
        return IngestResult(self.name, len(rows), "ok", None)
