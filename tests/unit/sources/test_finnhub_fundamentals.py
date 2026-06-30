from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from sma.ingest.sources.finnhub_fundamentals import FinnhubFundamentalsSource
from sma.ingest.store import Store


@pytest.fixture
def store():
    s = Store(":memory:").connect()
    yield s
    s.close()


def _fake_metric(ticker: str):
    return {
        "metric": {
            "peNormalizedAnnual": 28.5,
            "pbAnnual": 12.3,
            "roeRfy": 0.55,
            "totalDebt/totalEquityAnnual": 1.5,
            "netProfitMarginAnnual": 0.25,
            "revenueGrowth5Y": 0.08,
            "marketCapitalization": 3_000_000.0,
        }
    }


def _fake_earnings():
    return {
        "earningsCalendar": [
            {
                "symbol": "AAPL",
                "date": "2026-05-01",
                "epsEstimate": 1.55,
                "epsActual": None,
                "revenueEstimate": 95_000_000_000,
                "revenueActual": None,
            }
        ]
    }


def test_fundamentals_inserts_metric_row(store):
    src = FinnhubFundamentalsSource(api_key="fake")
    run_id = store.allocate_run_id()
    store.log_run_start(run_id, source="finnhub_fundamentals")

    fake_client = MagicMock()
    fake_client.company_basic_financials.side_effect = \
        lambda symbol, metric: _fake_metric(symbol)
    fake_client.earnings_calendar.return_value = _fake_earnings()

    with patch.object(src, "_client", fake_client):
        result = src.fetch(["AAPL"], date(2026, 4, 26), store, run_id)

    assert result.status == "ok"
    rows = store.conn.execute(
        "SELECT ticker, pe, pb, roe, source FROM fundamentals"
    ).fetchall()
    assert rows == [("AAPL", 28.5, 12.3, 0.55, "finnhub")]
    earnings = store.conn.execute(
        "SELECT ticker, report_date, eps_estimate FROM earnings"
    ).fetchall()
    assert earnings == [("AAPL", date(2026, 5, 1), 1.55)]
