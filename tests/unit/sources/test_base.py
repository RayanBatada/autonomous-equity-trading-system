
from sma.ingest.sources.base import IngestResult


def test_ingest_result_ok_constructs_with_minimum_fields():
    r = IngestResult(source="yfinance", rows_inserted=120, status="ok", error=None)
    assert r.source == "yfinance"
    assert r.rows_inserted == 120
    assert r.status == "ok"
    assert r.error is None


def test_ingest_result_error_carries_error_message():
    r = IngestResult(source="finnhub_news", rows_inserted=0, status="error", error="rate limited")
    assert r.error == "rate limited"
    assert r.status == "error"
