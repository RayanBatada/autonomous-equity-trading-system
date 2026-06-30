from datetime import date
from unittest.mock import patch

import pandas as pd
import pytest

from sma.ingest.sources.yfinance_prices import YFinancePricesSource
from sma.ingest.store import Store


@pytest.fixture
def store():
    s = Store(":memory:").connect()
    yield s
    s.close()


def _fake_yf_dataframe():
    return pd.DataFrame(
        {
            "Open":      [150.0, 151.0],
            "High":      [152.0, 153.0],
            "Low":       [149.0, 150.0],
            "Close":     [151.5, 152.5],
            "Adj Close": [151.5, 152.5],
            "Volume":    [10_000_000, 11_000_000],
        },
        index=pd.to_datetime(["2026-04-22", "2026-04-23"]),
    )


def test_fetch_inserts_rows(store):
    src = YFinancePricesSource(lookback_days=5)
    run_id = store.allocate_run_id()
    store.log_run_start(run_id, source="yfinance")

    with patch("sma.ingest.sources.yfinance_prices.yf.download",
               return_value=_fake_yf_dataframe()):
        result = src.fetch(["AAPL"], date(2026, 4, 23), store, run_id)

    assert result.status == "ok"
    assert result.rows_inserted == 2
    rows = store.conn.execute(
        "SELECT ticker, date, open, close, source FROM prices ORDER BY date"
    ).fetchall()
    assert len(rows) == 2
    assert rows[0][0] == "AAPL"
    assert rows[0][4] == "yfinance"


def test_fetch_handles_empty_dataframe(store):
    src = YFinancePricesSource(lookback_days=5)
    run_id = store.allocate_run_id()
    store.log_run_start(run_id, source="yfinance")

    with patch("sma.ingest.sources.yfinance_prices.yf.download",
               return_value=pd.DataFrame()):
        result = src.fetch(["FAKE"], date(2026, 4, 23), store, run_id)

    assert result.rows_inserted == 0
    assert result.status == "ok"


def _fake_yf_dataframe_multiindex(ticker: str = "AAPL"):
    """Mimics newer yfinance behavior where columns are MultiIndex even for one ticker."""
    df = _fake_yf_dataframe()
    df.columns = pd.MultiIndex.from_tuples([(c, ticker) for c in df.columns])
    return df


def test_fetch_handles_multiindex_columns(store):
    """Newer yfinance versions return MultiIndex columns by default. We must flatten."""
    src = YFinancePricesSource(lookback_days=5)
    run_id = store.allocate_run_id()
    store.log_run_start(run_id, source="yfinance")

    with patch("sma.ingest.sources.yfinance_prices.yf.download",
               return_value=_fake_yf_dataframe_multiindex("AAPL")):
        result = src.fetch(["AAPL"], date(2026, 4, 23), store, run_id)

    assert result.status == "ok"
    assert result.rows_inserted == 2


def test_fetch_falls_back_to_per_ticker_on_bulk_failure(store):
    src = YFinancePricesSource(lookback_days=5)
    run_id = store.allocate_run_id()
    store.log_run_start(run_id, source="yfinance")

    call_count = {"n": 0}
    def fake_download(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("bulk download flaked")
        return _fake_yf_dataframe()

    with patch("sma.ingest.sources.yfinance_prices.yf.download",
               side_effect=fake_download):
        result = src.fetch(["AAPL", "MSFT"], date(2026, 4, 23), store, run_id)

    assert result.status == "ok"
    assert call_count["n"] >= 3


def test_fetch_returns_error_when_no_rows_inserted(store):
    """All yfinance fetches failing (e.g. DNS down) → rows=0 → status MUST be
    'error', not 'ok'. status='ok' with 0 rows let enough_sources_succeeded pass
    on a dead network, masking a total price-ingest failure — the exact bug that
    hid the 6/1-6/3 bot freeze (decide blocked on stale prices, no alert)."""
    src = YFinancePricesSource(lookback_days=5)
    run_id = store.allocate_run_id()
    store.log_run_start(run_id, source="yfinance")
    with patch("sma.ingest.sources.yfinance_prices.yf.download",
               side_effect=Exception("nodename nor servname provided, or not known")):
        result = src.fetch(["AAPL"], date(2026, 4, 23), store, run_id)
    assert result.rows_inserted == 0
    assert result.status == "error"
