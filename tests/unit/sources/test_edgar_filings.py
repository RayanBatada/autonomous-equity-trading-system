from datetime import date
from unittest.mock import patch

import httpx
import pytest

from sma.ingest.sources.edgar_filings import EdgarFilingsSource
from sma.ingest.store import Store


@pytest.fixture
def store():
    s = Store(":memory:").connect()
    yield s
    s.close()


_TICKERS_LOOKUP = {
    "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
    "1": {"cik_str": 789019, "ticker": "MSFT", "title": "Microsoft Corp"},
}


_SUBMISSIONS = {
    "filings": {
        "recent": {
            "accessionNumber": ["0000320193-26-000001", "0000320193-26-000002"],
            "filingDate":      ["2026-04-25",          "2026-03-15"],
            "form":            ["8-K",                 "10-Q"],
            "primaryDocument": ["aapl-20260425.htm",   "aapl-20260315.htm"],
        }
    }
}


def _resp(json_data, status=200):
    request = httpx.Request("GET", "https://www.sec.gov")
    return httpx.Response(status, json=json_data, request=request)


def test_edgar_pulls_recent_filings(store):
    src = EdgarFilingsSource(user_agent="Test (t@example.com)")
    run_id = store.allocate_run_id()
    store.log_run_start(run_id, source="edgar")

    def fake_get(url, **kw):
        if "company_tickers" in url:
            return _resp(_TICKERS_LOOKUP)
        if "submissions" in url:
            return _resp(_SUBMISSIONS)
        return _resp({}, status=404)

    with patch.object(src._client, "get", side_effect=fake_get):
        result = src.fetch(["AAPL"], date(2026, 4, 26), store, run_id)

    assert result.status == "ok"
    rows = store.conn.execute(
        "SELECT ticker, filing_type, accession_no FROM filings ORDER BY filed_at DESC"
    ).fetchall()
    assert ("AAPL", "8-K", "0000320193-26-000001") in rows


def test_edgar_skips_unknown_ticker(store):
    src = EdgarFilingsSource(user_agent="Test (t@example.com)")
    run_id = store.allocate_run_id()
    store.log_run_start(run_id, source="edgar")

    def fake_get(url, **kw):
        if "company_tickers" in url:
            return _resp(_TICKERS_LOOKUP)
        return _resp({}, status=404)

    with patch.object(src._client, "get", side_effect=fake_get):
        result = src.fetch(["NOPE"], date(2026, 4, 26), store, run_id)

    assert result.status == "ok"
    assert result.rows_inserted == 0
