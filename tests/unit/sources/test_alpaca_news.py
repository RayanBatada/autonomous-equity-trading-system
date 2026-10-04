from datetime import date
from unittest.mock import patch

import httpx
import pytest

from sma.ingest.sources.alpaca_news import AlpacaNewsSource
from sma.ingest.store import Store


@pytest.fixture
def store():
    s = Store(":memory:").connect()
    yield s
    s.close()


def _resp(payload: dict, status: int = 200) -> httpx.Response:
    request = httpx.Request("GET", "https://data.alpaca.markets/v1beta1/news")
    return httpx.Response(status, json=payload, request=request)


def _article(
    article_id: int = 1,
    symbols: list[str] | None = None,
    source: str = "benzinga",
    url: str | None = None,
    headline: str | None = None,
) -> dict:
    return {
        "id": article_id,
        "headline": headline or f"Headline {article_id}",
        "summary": f"Summary for article {article_id}.",
        "url": url or f"https://www.benzinga.com/article/{article_id}",
        "source": source,
        "symbols": symbols or ["AAPL"],
        "author": "Test Author",
        "created_at": "2026-04-25T15:00:00Z",
        "updated_at": "2026-04-25T15:00:01Z",
    }


def test_alpaca_news_inserts_one_row_per_intersecting_ticker(store):
    src = AlpacaNewsSource(api_key="fake", api_secret="fake")
    run_id = store.allocate_run_id()
    store.log_run_start(run_id, source="alpaca_news")

    payload = {
        "news": [_article(article_id=1, symbols=["AAPL", "AMZN"])],
        "next_page_token": None,
    }

    with patch.object(src._client, "get", return_value=_resp(payload)):
        result = src.fetch(["AAPL", "AMZN"], date(2026, 4, 26), store, run_id)

    assert result.status == "ok"
    assert result.rows_inserted == 2
    rows = store.conn.execute(
        "SELECT ticker, url FROM news ORDER BY ticker"
    ).fetchall()
    assert {r[0] for r in rows} == {"AAPL", "AMZN"}
    # both rows point at the same article
    assert rows[0][1] == rows[1][1]


def test_alpaca_news_translates_dash_symbols_and_stores_canonical(store):
    """Alpaca wants dot share-class notation (BRK.B); the system is
    yfinance-canonical (BRK-B). The request must translate and the response must
    map back, exactly as alpaca_prices already does.

    Regression (found 2026-07-29): BRK-B had **0 news rows all time** while every
    other ticker had news, so `news_per_ticker_minimum` failed every single day
    with `missing_news=['BRK-B']`. Two halves to the bug, and BOTH are needed:
      1. the request sent the literal `BRK-B`, which Alpaca does not recognize;
      2. the response matcher compared Alpaca's `BRK.B` against a batch set
         holding `BRK-B`, so even a returned Berkshire article could never match.
    This is the same class of bug that killed the ENTIRE alpaca_prices bulk
    request from 6/12 until the 2026-07-01 review.
    """
    src = AlpacaNewsSource(api_key="fake", api_secret="fake")
    run_id = store.allocate_run_id()
    store.log_run_start(run_id, source="alpaca_news")

    # Alpaca answers in ITS notation (dotted).
    payload = {
        "news": [_article(article_id=1, symbols=["BRK.B", "AAPL"])],
        "next_page_token": None,
    }

    with patch.object(src._client, "get", return_value=_resp(payload)) as get:
        result = src.fetch(["BRK-B", "AAPL"], date(2026, 4, 26), store, run_id)

    # requested in Alpaca's dot form
    sent = get.call_args.kwargs["params"]["symbols"]
    assert "BRK.B" in sent, f"request must translate to dot form, got {sent!r}"
    assert "BRK-B" not in sent, f"dashed form must not be sent, got {sent!r}"

    # stored under the canonical dash form
    assert result.rows_inserted == 2
    tickers = {
        r[0] for r in store.conn.execute("SELECT ticker FROM news").fetchall()
    }
    assert tickers == {"BRK-B", "AAPL"}


def test_alpaca_news_dash_dot_aliases_in_one_batch_both_get_rows(store):
    """Both alias spellings in one batch must each get their own news row.

    Codex review 2026-07-29: the dash-to-dot translation built its reverse map as
    `{v: k for k, v in to_alpaca.items()}`, which is NON-INJECTIVE — `BRK-B` and
    `BRK.B` both translate to `BRK.B`, so the reverse dict kept only whichever
    came last and the other ticker's article was silently misattributed or
    dropped. `universe.yaml` currently holds only `BRK-B`, but a `BRK.B` zombie
    HAS been in the universe before (deduped out during the 2026-07-01 review),
    so this is a live regression risk, not a hypothetical one.
    """
    src = AlpacaNewsSource(api_key="fake", api_secret="fake")
    run_id = store.allocate_run_id()
    store.log_run_start(run_id, source="alpaca_news")

    payload = {
        "news": [_article(article_id=1, symbols=["BRK.B"])],
        "next_page_token": None,
    }

    with patch.object(src._client, "get", return_value=_resp(payload)):
        result = src.fetch(["BRK-B", "BRK.B"], date(2026, 4, 26), store, run_id)

    tickers = {
        r[0] for r in store.conn.execute("SELECT ticker FROM news").fetchall()
    }
    assert tickers == {"BRK-B", "BRK.B"}, (
        f"both alias spellings must be attributed, got {tickers}"
    )
    assert result.rows_inserted == 2


def test_alpaca_news_emits_only_for_requested_tickers(store):
    src = AlpacaNewsSource(api_key="fake", api_secret="fake")
    run_id = store.allocate_run_id()
    store.log_run_start(run_id, source="alpaca_news")

    payload = {
        "news": [_article(
            article_id=1, symbols=["AAPL", "AMZN", "GOOGL"]
        )],
        "next_page_token": None,
    }

    with patch.object(src._client, "get", return_value=_resp(payload)):
        result = src.fetch(["AAPL"], date(2026, 4, 26), store, run_id)

    assert result.status == "ok"
    assert result.rows_inserted == 1
    rows = store.conn.execute("SELECT ticker FROM news").fetchall()
    assert [r[0] for r in rows] == ["AAPL"]


def test_alpaca_news_handles_pagination(store):
    src = AlpacaNewsSource(api_key="fake", api_secret="fake")
    run_id = store.allocate_run_id()
    store.log_run_start(run_id, source="alpaca_news")

    page1 = {
        "news": [_article(article_id=1, symbols=["AAPL"])],
        "next_page_token": "abc",
    }
    page2 = {
        "news": [_article(article_id=2, symbols=["AAPL"])],
        "next_page_token": None,
    }

    responses = [_resp(page1), _resp(page2)]
    with patch.object(src._client, "get", side_effect=responses) as mock_get:
        result = src.fetch(["AAPL"], date(2026, 4, 26), store, run_id)

    assert result.status == "ok"
    assert result.rows_inserted == 2
    # Confirm pagination param was passed on second call
    second_call_params = mock_get.call_args_list[1].kwargs["params"]
    assert second_call_params.get("page_token") == "abc"


def test_alpaca_news_handles_429(store):
    src = AlpacaNewsSource(api_key="fake", api_secret="fake")
    run_id = store.allocate_run_id()
    store.log_run_start(run_id, source="alpaca_news")

    rate_limited = _resp({"message": "rate limit"}, status=429)

    with patch.object(src._client, "get", return_value=rate_limited):
        result = src.fetch(["AAPL"], date(2026, 4, 26), store, run_id)

    assert result.status == "rate_limited"
    assert result.rows_inserted == 0


def test_alpaca_news_handles_empty_response(store):
    src = AlpacaNewsSource(api_key="fake", api_secret="fake")
    run_id = store.allocate_run_id()
    store.log_run_start(run_id, source="alpaca_news")

    payload = {"news": [], "next_page_token": None}

    with patch.object(src._client, "get", return_value=_resp(payload)):
        result = src.fetch(["AAPL"], date(2026, 4, 26), store, run_id)

    assert result.status == "ok"
    assert result.rows_inserted == 0


def test_alpaca_news_dedups_within_run(store):
    src = AlpacaNewsSource(api_key="fake", api_secret="fake")
    run_id = store.allocate_run_id()
    store.log_run_start(run_id, source="alpaca_news")

    article = _article(article_id=1, symbols=["AAPL"])
    page1 = {"news": [article], "next_page_token": "abc"}
    page2 = {"news": [article], "next_page_token": None}

    responses = [_resp(page1), _resp(page2)]
    with patch.object(src._client, "get", side_effect=responses):
        result = src.fetch(["AAPL"], date(2026, 4, 26), store, run_id)

    assert result.status == "ok"
    assert result.rows_inserted == 1
    rows = store.conn.execute("SELECT ticker FROM news").fetchall()
    assert len(rows) == 1


def test_alpaca_news_raises_on_missing_credentials():
    with pytest.raises(ValueError):
        AlpacaNewsSource(api_key="", api_secret="fake")
    with pytest.raises(ValueError):
        AlpacaNewsSource(api_key="fake", api_secret="")


def test_alpaca_news_source_name_set(store):
    src = AlpacaNewsSource(api_key="fake", api_secret="fake")
    run_id = store.allocate_run_id()
    store.log_run_start(run_id, source="alpaca_news")

    payload = {
        "news": [_article(article_id=1, symbols=["AAPL"], source="benzinga")],
        "next_page_token": None,
    }

    with patch.object(src._client, "get", return_value=_resp(payload)):
        result = src.fetch(["AAPL"], date(2026, 4, 26), store, run_id)

    assert result.rows_inserted == 1
    row = store.conn.execute("SELECT source FROM news").fetchone()
    assert row[0].startswith("alpaca")
    assert row[0] == "alpaca:benzinga"
