from datetime import date
from unittest.mock import MagicMock, patch

import pytest
from finnhub.exceptions import FinnhubAPIException

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


def test_finnhub_news_translates_dash_symbols_and_stores_canonical(store):
    """Finnhub keys share classes with a DOT (BRK.B); the system is
    yfinance-canonical (BRK-B). Request must translate; storage stays canonical.

    Regression (found 2026-07-29, verified against the live Finnhub API):
        company-news BRK-B -> 0 articles
        company-news BRK.B -> 32 articles
    `finnhub_news` passed the raw dashed ticker straight through, so every
    Berkshire article was silently discarded and BRK-B ended up with 0 news rows
    all time — the sole cause of `news_per_ticker_minimum` failing nightly.
    (Alpaca was ALSO asking with the wrong notation, fixed separately, but Alpaca
    carries no BRK coverage on this tier at all — Finnhub is where the data is.)
    """
    src = FinnhubNewsSource(api_key="fake")
    run_id = store.allocate_run_id()
    store.log_run_start(run_id, source="finnhub_news")

    fake_client = MagicMock()
    fake_client.company_news.return_value = _fake_news("BRK.B")

    with patch.object(src, "_client", fake_client):
        src.fetch(["BRK-B"], date(2026, 4, 26), store, run_id)

    # asked Finnhub in ITS dotted notation
    assert fake_client.company_news.call_args[0][0] == "BRK.B"
    # stored under the canonical dashed form
    tickers = {
        r[0] for r in store.conn.execute("SELECT ticker FROM news").fetchall()
    }
    assert tickers == {"BRK-B"}


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


def _fake_429():
    return FinnhubAPIException(MagicMock(status_code=429, text="rate"))


def test_finnhub_news_429_then_success_retries_and_inserts(store):
    """A 429 within a run must retry the SAME ticker, not drop it for the
    night. Sleep is injected so the test doesn't actually wait 15-30s.
    """
    src = FinnhubNewsSource(api_key="fake", sleep_fn=lambda s: None)
    run_id = store.allocate_run_id()
    store.log_run_start(run_id, source="finnhub_news")

    calls = {"n": 0}

    def side_effect(symbol, _from, to):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _fake_429()
        return _fake_news(symbol)

    fake_client = MagicMock()
    fake_client.company_news.side_effect = side_effect

    with patch.object(src, "_client", fake_client):
        result = src.fetch(["AAPL"], date(2026, 4, 26), store, run_id)

    assert calls["n"] == 2
    assert result.rows_inserted == 2
    tickers = {r[0] for r in store.conn.execute("SELECT ticker FROM news").fetchall()}
    assert tickers == {"AAPL"}


def test_finnhub_news_3x429_gives_up_with_warning(store):
    """Three consecutive 429s (initial + 2 retries) give up; ticker is
    skipped but the run continues for other tickers.
    """
    src = FinnhubNewsSource(api_key="fake", sleep_fn=lambda s: None)
    run_id = store.allocate_run_id()
    store.log_run_start(run_id, source="finnhub_news")

    calls = {"n": 0}

    def side_effect(symbol, _from, to):
        calls["n"] += 1
        if symbol == "BAD.CO":
            raise _fake_429()
        return _fake_news(symbol)

    fake_client = MagicMock()
    fake_client.company_news.side_effect = side_effect

    with patch.object(src, "_client", fake_client):
        result = src.fetch(["BAD.CO", "AAPL"], date(2026, 4, 26), store, run_id)

    # BAD.CO: initial + 2 retries = 3 calls; AAPL: 1 call.
    assert calls["n"] == 4
    tickers = {r[0] for r in store.conn.execute("SELECT ticker FROM news").fetchall()}
    assert tickers == {"AAPL"}
    assert result.rows_inserted == 2


def test_finnhub_news_retry_sleep_budget_caps_a_429_storm(store):
    """A 429-storm night must not hold the writer lock for hours. With a 40s
    budget the run sleeps at most 40s TOTAL, then skips the rest of the
    rate-limited tickers with one attempt each."""
    sleeps = []
    src = FinnhubNewsSource(
        api_key="fake", sleep_fn=sleeps.append, retry_sleep_budget_s=40.0,
    )
    run_id = store.allocate_run_id()
    store.log_run_start(run_id, source="finnhub_news")

    fake_client = MagicMock()
    fake_client.company_news.side_effect = lambda symbol, _from, to: (_ for _ in ()).throw(
        _fake_429()
    )

    tickers = [f"T{i}" for i in range(40)]
    with patch.object(src, "_client", fake_client):
        result = src.fetch(tickers, date(2026, 4, 26), store, run_id)

    assert sum(sleeps) <= 40.0, "cumulative retry sleep must respect the budget"
    # Unbounded, 40 tickers x 2 retries would be 80 sleeps of 15-30s (~30 min).
    assert len(sleeps) <= 4
    assert result.rows_inserted == 0


def test_finnhub_news_retry_sleep_budget_resets_per_fetch(store):
    """The budget is per RUN, not per process: the next night's fetch() gets a
    full one, so one bad night cannot permanently disable retry-in-place."""
    sleeps = []
    src = FinnhubNewsSource(
        api_key="fake", sleep_fn=sleeps.append, retry_sleep_budget_s=40.0,
    )
    run_id = store.allocate_run_id()
    store.log_run_start(run_id, source="finnhub_news")

    fake_client = MagicMock()
    fake_client.company_news.side_effect = lambda symbol, _from, to: (_ for _ in ()).throw(
        _fake_429()
    )

    tickers = [f"T{i}" for i in range(40)]
    with patch.object(src, "_client", fake_client):
        src.fetch(tickers, date(2026, 4, 26), store, run_id)
    # Sleeps are clamped to what's left, so a fully-spent budget sums to
    # exactly total_s regardless of where the 15-30s draws landed.
    assert sum(sleeps) == pytest.approx(40.0)

    sleeps.clear()
    with patch.object(src, "_client", fake_client):
        src.fetch(tickers, date(2026, 4, 27), store, run_id)

    assert sum(sleeps) == pytest.approx(40.0), "a fresh fetch() must get a fresh budget"


def test_finnhub_news_non_429_exception_not_retried(store):
    src = FinnhubNewsSource(api_key="fake", sleep_fn=lambda s: None)
    run_id = store.allocate_run_id()
    store.log_run_start(run_id, source="finnhub_news")

    calls = {"n": 0}

    def side_effect(symbol, _from, to):
        calls["n"] += 1
        raise RuntimeError("connection reset")

    fake_client = MagicMock()
    fake_client.company_news.side_effect = side_effect

    with patch.object(src, "_client", fake_client):
        result = src.fetch(["AAPL"], date(2026, 4, 26), store, run_id)

    assert calls["n"] == 1
    assert result.rows_inserted == 0
