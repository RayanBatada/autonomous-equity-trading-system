from datetime import date, datetime
from unittest.mock import MagicMock, patch

import pytest

from sma.ingest.sources.alpaca_prices import AlpacaPricesSource
from sma.ingest.store import Store


@pytest.fixture
def store():
    s = Store(":memory:").connect()
    yield s
    s.close()


class FakeBar:
    def __init__(self, t, o, h, l, c, v):
        self.timestamp = t
        self.open = o; self.high = h; self.low = l; self.close = c
        self.volume = v


def _fake_bars(ticker: str):
    return [
        FakeBar(datetime(2026, 4, 22, 16, 0), 150.0, 152.0, 149.0, 151.5, 10_000_000),
        FakeBar(datetime(2026, 4, 23, 16, 0), 151.0, 153.0, 150.0, 152.5, 11_000_000),
    ]


def test_alpaca_fetch_inserts_rows(store):
    src = AlpacaPricesSource(
        api_key="k", api_secret="s",
        base_url="https://paper-api.alpaca.markets",
        lookback_days=5,
    )
    run_id = store.allocate_run_id()
    store.log_run_start(run_id, source="alpaca")

    fake_client = MagicMock()
    fake_client.get_stock_bars.return_value.data = {"AAPL": _fake_bars("AAPL")}

    with patch.object(src, "_client", fake_client):
        result = src.fetch(["AAPL"], date(2026, 4, 23), store, run_id)

    assert result.status == "ok"
    assert result.rows_inserted == 2
    rows = store.conn.execute(
        "SELECT ticker, source FROM prices ORDER BY date"
    ).fetchall()
    assert all(r[1] == "alpaca" for r in rows)


def test_alpaca_fetch_returns_zero_when_empty(store):
    src = AlpacaPricesSource(
        api_key="k", api_secret="s",
        base_url="https://paper-api.alpaca.markets",
        lookback_days=5,
    )
    run_id = store.allocate_run_id()
    store.log_run_start(run_id, source="alpaca")

    fake_client = MagicMock()
    fake_client.get_stock_bars.return_value.data = {}

    with patch.object(src, "_client", fake_client):
        result = src.fetch(["UNKNOWN"], date(2026, 4, 23), store, run_id)

    assert result.rows_inserted == 0
    assert result.status == "ok"
