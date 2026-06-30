from datetime import date
from unittest.mock import patch

import httpx
import pytest

from sma.ingest.sources.newsapi import NewsAPISource
from sma.ingest.store import Store


@pytest.fixture
def store():
    s = Store(":memory:").connect()
    yield s
    s.close()


def _fake_response(articles):
    payload = {"status": "ok", "totalResults": len(articles), "articles": articles}
    request = httpx.Request("GET", "https://newsapi.org/v2/everything")
    return httpx.Response(200, json=payload, request=request)


def test_newsapi_inserts_articles(store):
    src = NewsAPISource(api_key="fake")
    run_id = store.allocate_run_id()
    store.log_run_start(run_id, source="newsapi")

    articles = [
        {
            "publishedAt": "2026-04-25T15:00:00Z",
            "title": "AAPL surges on strong earnings",
            "url": "https://example.com/aapl-1",
            "description": "Apple beat estimates.",
            "source": {"name": "Example"},
        },
        {
            "publishedAt": "2026-04-24T09:00:00Z",
            "title": "AAPL announces buyback",
            "url": "https://example.com/aapl-2",
            "description": "Plans $10B buyback.",
            "source": {"name": "Example"},
        },
    ]

    with patch.object(src._client, "get",
                      return_value=_fake_response(articles)):
        result = src.fetch(["AAPL"], date(2026, 4, 26), store, run_id)

    assert result.status == "ok"
    assert result.rows_inserted == 2
    rows = store.conn.execute("SELECT source FROM news").fetchall()
    assert all(r[0] == "newsapi" for r in rows)


def test_newsapi_handles_429_rate_limit(store):
    src = NewsAPISource(api_key="fake")
    run_id = store.allocate_run_id()
    store.log_run_start(run_id, source="newsapi")

    request = httpx.Request("GET", "https://newsapi.org/v2/everything")
    rate_limited = httpx.Response(429, json={"status": "error", "message": "rate limit"},
                                  request=request)

    with patch.object(src._client, "get", return_value=rate_limited):
        result = src.fetch(["AAPL"], date(2026, 4, 26), store, run_id)

    assert result.status == "rate_limited"
    assert result.rows_inserted == 0
