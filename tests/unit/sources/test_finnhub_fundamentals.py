from datetime import date
from unittest.mock import MagicMock, patch

import pytest
from finnhub.exceptions import FinnhubAPIException

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


def test_earnings_calendar_maps_brk_a_response_to_canonical_brk_b(store):
    """Regression (verified live 2026-08-16): Finnhub's earnings_calendar
    returns Berkshire events under symbol="BRK.A" even when BRK-B (or
    BRK.B) is the ticker being tracked. Before this fix, the `sym in
    ticker_set` filter compared "BRK.A" against a set holding "BRK-B" and
    always dropped it, so BRK-B got zero forward earnings from this source.
    """
    src = FinnhubFundamentalsSource(api_key="fake")
    run_id = store.allocate_run_id()
    store.log_run_start(run_id, source="finnhub_fundamentals")

    fake_client = MagicMock()
    fake_client.company_basic_financials.side_effect = \
        lambda symbol, metric: _fake_metric(symbol)
    fake_client.earnings_calendar.return_value = {
        "earningsCalendar": [
            {
                "symbol": "BRK.A",
                "date": "2026-05-02",
                "epsEstimate": 5.1,
                "epsActual": None,
                "revenueEstimate": 90_000_000_000,
                "revenueActual": None,
            },
        ]
    }

    with patch.object(src, "_client", fake_client):
        result = src.fetch(["BRK-B"], date(2026, 4, 26), store, run_id)

    assert result.status == "ok"
    earnings = store.conn.execute(
        "SELECT ticker, report_date, eps_estimate FROM earnings"
    ).fetchall()
    assert earnings == [("BRK-B", date(2026, 5, 2), 5.1)]


def test_earnings_calendar_still_drops_unrelated_symbols(store):
    """The alias translation must not become a blanket pass -- a vendor
    symbol that genuinely isn't in the requested batch is still filtered
    out, exactly as `sym in ticker_set` did before."""
    src = FinnhubFundamentalsSource(api_key="fake")
    run_id = store.allocate_run_id()
    store.log_run_start(run_id, source="finnhub_fundamentals")

    fake_client = MagicMock()
    fake_client.company_basic_financials.side_effect = \
        lambda symbol, metric: _fake_metric(symbol)
    fake_client.earnings_calendar.return_value = {
        "earningsCalendar": [
            {
                "symbol": "MSFT",
                "date": "2026-05-02",
                "epsEstimate": 3.0,
                "epsActual": None,
                "revenueEstimate": None,
                "revenueActual": None,
            },
        ]
    }

    with patch.object(src, "_client", fake_client):
        src.fetch(["AAPL"], date(2026, 4, 26), store, run_id)

    earnings = store.conn.execute("SELECT ticker FROM earnings").fetchall()
    assert earnings == []


def _fake_429():
    return FinnhubAPIException(MagicMock(status_code=429, text="rate"))


def test_fundamentals_429_then_success_retries_and_inserts(store):
    """A 429 within a run must retry the SAME ticker, not drop it for the
    night. Sleep is injected so the test doesn't actually wait 15-30s.
    """
    src = FinnhubFundamentalsSource(api_key="fake", sleep_fn=lambda s: None)
    run_id = store.allocate_run_id()
    store.log_run_start(run_id, source="finnhub_fundamentals")

    calls = {"n": 0}

    def side_effect(symbol, metric):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _fake_429()
        return _fake_metric(symbol)

    fake_client = MagicMock()
    fake_client.company_basic_financials.side_effect = side_effect
    fake_client.earnings_calendar.return_value = {"earningsCalendar": []}

    with patch.object(src, "_client", fake_client):
        result = src.fetch(["AAPL"], date(2026, 4, 26), store, run_id)

    assert calls["n"] == 2
    assert result.status == "ok"
    rows = store.conn.execute("SELECT ticker, pe FROM fundamentals").fetchall()
    assert rows == [("AAPL", 28.5)]


def test_fundamentals_3x429_gives_up_with_warning(store):
    """Three consecutive 429s (initial + 2 retries) give up; ticker is
    skipped but the run continues for other tickers.
    """
    src = FinnhubFundamentalsSource(api_key="fake", sleep_fn=lambda s: None)
    run_id = store.allocate_run_id()
    store.log_run_start(run_id, source="finnhub_fundamentals")

    calls = {"n": 0}

    def side_effect(symbol, metric):
        calls["n"] += 1
        if symbol == "BAD":
            raise _fake_429()
        return _fake_metric(symbol)

    fake_client = MagicMock()
    fake_client.company_basic_financials.side_effect = side_effect
    fake_client.earnings_calendar.return_value = {"earningsCalendar": []}

    with patch.object(src, "_client", fake_client):
        src.fetch(["BAD", "AAPL"], date(2026, 4, 26), store, run_id)

    # BAD: initial + 2 retries = 3 calls; AAPL: 1 call.
    assert calls["n"] == 4
    tickers = {r[0] for r in store.conn.execute("SELECT ticker FROM fundamentals").fetchall()}
    assert tickers == {"AAPL"}


def test_fundamentals_non_429_exception_not_retried(store):
    src = FinnhubFundamentalsSource(api_key="fake", sleep_fn=lambda s: None)
    run_id = store.allocate_run_id()
    store.log_run_start(run_id, source="finnhub_fundamentals")

    calls = {"n": 0}

    def side_effect(symbol, metric):
        calls["n"] += 1
        raise RuntimeError("connection reset")

    fake_client = MagicMock()
    fake_client.company_basic_financials.side_effect = side_effect
    fake_client.earnings_calendar.return_value = {"earningsCalendar": []}

    with patch.object(src, "_client", fake_client):
        result = src.fetch(["AAPL"], date(2026, 4, 26), store, run_id)

    assert calls["n"] == 1
    rows = store.conn.execute("SELECT ticker FROM fundamentals").fetchall()
    assert rows == []
    assert result.status == "ok"
