from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from sma.ingest.sources.finnhub_sentiment import FinnhubSentimentSource
from sma.ingest.store import Store


@pytest.fixture
def store():
    s = Store(":memory:").connect()
    yield s
    s.close()


def test_finnhub_sentiment_inserts_row(store):
    src = FinnhubSentimentSource(api_key="fake")
    run_id = store.allocate_run_id()
    store.log_run_start(run_id, source="finnhub_sentiment")

    fake_client = MagicMock()
    fake_client.news_sentiment.return_value = {
        "buzz": {"articlesInLastWeek": 12, "buzz": 0.85, "weeklyAverage": 10},
        "companyNewsScore": 0.62,
        "sectorAverageBullishPercent": 0.55,
        "sentiment": {"bearishPercent": 0.30, "bullishPercent": 0.70},
        "symbol": "AAPL",
    }

    with patch.object(src, "_client", fake_client):
        result = src.fetch(["AAPL"], date(2026, 4, 26), store, run_id)

    assert result.status == "ok"
    assert result.rows_inserted == 1
    row = store.conn.execute(
        "SELECT ticker, score, source, num_articles FROM sentiment"
    ).fetchone()
    assert row[0] == "AAPL"
    assert row[2] == "finnhub"
    assert row[3] == 12


def test_finnhub_sentiment_skips_tickers_with_no_data(store):
    src = FinnhubSentimentSource(api_key="fake")
    run_id = store.allocate_run_id()
    store.log_run_start(run_id, source="finnhub_sentiment")

    fake_client = MagicMock()
    fake_client.news_sentiment.return_value = {}

    with patch.object(src, "_client", fake_client):
        result = src.fetch(["UNKNOWN"], date(2026, 4, 26), store, run_id)

    assert result.rows_inserted == 0
    assert result.status == "ok"
