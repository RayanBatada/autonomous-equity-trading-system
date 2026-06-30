from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from sma.ingest.sources.finnhub_news import FinnhubNewsSource
from sma.ingest.store import Store


@pytest.fixture
def store():
    s = Store(":memory:").connect()
    yield s
    s.close()


def _fake_news(ticker: str):
    return [
        {
            "datetime": 1745625600,
            "headline": f"{ticker} hits new high",
            "url": f"https://example.com/{ticker}/1",
            "summary": "Great quarter, strong guidance.",
            "source": "ExampleNews",
        },
        {
            "datetime": 1745539200,
            "headline": f"{ticker} CEO interview",
            "url": f"https://example.com/{ticker}/2",
            "summary": "Discusses long-term strategy.",
            "source": "ExampleNews",
        },
    ]


def test_finnhub_news_inserts_rows(store):
    src = FinnhubNewsSource(api_key="fake")
    run_id = store.allocate_run_id()
    store.log_run_start(run_id, source="finnhub_news")

    fake_client = MagicMock()
    fake_client.company_news.side_effect = lambda symbol, _from, to: _fake_news(symbol)

    with patch.object(src, "_client", fake_client):
        result = src.fetch(["AAPL", "MSFT"], date(2026, 4, 26), store, run_id)

    assert result.status == "ok"
    assert result.rows_inserted == 4
    rows = store.conn.execute(
        "SELECT ticker, source FROM news ORDER BY ticker"
    ).fetchall()
    assert {r[0] for r in rows} == {"AAPL", "MSFT"}
    assert all(r[1] == "finnhub" for r in rows)


def test_finnhub_news_dedups_identical_articles(store):
    src = FinnhubNewsSource(api_key="fake")
    run_id = store.allocate_run_id()
    store.log_run_start(run_id, source="finnhub_news")

    duplicate = _fake_news("AAPL")[0]
    fake_client = MagicMock()
    fake_client.company_news.return_value = [duplicate, duplicate]

    with patch.object(src, "_client", fake_client):
        result = src.fetch(["AAPL"], date(2026, 4, 26), store, run_id)

    assert result.rows_inserted == 1
