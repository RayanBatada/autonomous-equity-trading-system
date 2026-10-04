"""Keyed APIs whose key rides in the URL never leak it to a log or ingest_log
(follow-up to the 2026-10-04 Finnhub redaction). Finnhub uses ?token=,
NewsAPI ?apiKey=; requests/httpx exception text quotes the URL."""

from datetime import date
from unittest.mock import MagicMock

from loguru import logger

KEY = "d0abcDEF123secretKEY456"


def _capture():
    lines = []
    sink = logger.add(lambda m: lines.append(str(m)), level="DEBUG")
    return lines, sink


def test_finnhub_sentiment_failure_log_is_redacted():
    from sma.ingest.sources.finnhub_sentiment import FinnhubSentimentSource

    src = FinnhubSentimentSource(api_key="x")
    src._client = MagicMock()
    src._client.news_sentiment.side_effect = ConnectionError(
        f"Max retries exceeded with url: /api/v1/news-sentiment?symbol=AAPL&token={KEY}"
    )
    lines, sink = _capture()
    try:
        src.fetch(["AAPL"], date(2026, 10, 2), MagicMock(), 1)
    finally:
        logger.remove(sink)
    assert any("finnhub_sentiment failed" in ln for ln in lines)
    assert not any(KEY in ln for ln in lines)


def test_newsapi_failure_log_is_redacted():
    from sma.ingest.sources.newsapi import NewsAPISource

    src = NewsAPISource(api_key=KEY)
    src._client = MagicMock()
    src._client.get.side_effect = ConnectionError(
        f"error for url https://newsapi.org/v2/everything?q=AAPL&apiKey={KEY}"
    )
    lines, sink = _capture()
    try:
        src.fetch(["AAPL"], date(2026, 10, 2), MagicMock(), 1)
    finally:
        logger.remove(sink)
    assert any("newsapi request failed" in ln for ln in lines)
    assert not any(KEY in ln for ln in lines)


def test_runner_source_crash_redacts_log_and_ingest_log():
    from sma.ingest.runner import IngestRunner

    src = MagicMock()
    src.name = "newsapi"
    src.fetch.side_effect = RuntimeError(f"boom https://x/?apiKey={KEY}")
    store = MagicMock()
    runner = IngestRunner(store=store, sources=[src], universe=["AAPL"])
    lines, sink = _capture()
    try:
        runner._fetch_one(1, date(2026, 10, 2), src, log_start=True)
    finally:
        logger.remove(sink)
    assert not any(KEY in ln for ln in lines)
    logged = [str(c) for c in store.log_run_end.call_args_list]
    assert logged and not any(KEY in c for c in logged)
