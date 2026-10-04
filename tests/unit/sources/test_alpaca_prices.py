from datetime import date, datetime
from unittest.mock import MagicMock, patch

import pytest

import sma.ingest.sources.alpaca_prices as alpaca_prices_mod
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


def test_alpaca_translates_dash_symbols_and_stores_canonical(store):
    """2026-07-01 review: Alpaca wants dot class notation (BRK.B); the system is
    yfinance-canonical (BRK-B). Request must translate, storage must map back —
    one dashed symbol used to kill the ENTIRE bulk request (source dead 6/12+)."""
    src = AlpacaPricesSource(
        api_key="k", api_secret="s",
        base_url="https://paper-api.alpaca.markets", lookback_days=5,
    )
    run_id = store.allocate_run_id()
    store.log_run_start(run_id, source="alpaca")

    fake_client = MagicMock()
    fake_client.get_stock_bars.return_value.data = {"BRK.B": _fake_bars("BRK.B")}

    with patch.object(src, "_client", fake_client):
        result = src.fetch(["BRK-B"], date(2026, 4, 23), store, run_id)

    # requested in Alpaca's dot form
    req = fake_client.get_stock_bars.call_args[0][0]
    assert req.symbol_or_symbols == ["BRK.B"]
    # stored under the canonical dash form
    assert result.rows_inserted == 2
    tickers = {r[0] for r in store.conn.execute("SELECT ticker FROM prices").fetchall()}
    assert tickers == {"BRK-B"}


def test_alpaca_drops_named_invalid_symbol_and_retries(store):
    """If Alpaca rejects a symbol by name, retry once without it instead of
    losing the whole source for the night."""
    src = AlpacaPricesSource(
        api_key="k", api_secret="s",
        base_url="https://paper-api.alpaca.markets", lookback_days=5,
    )
    run_id = store.allocate_run_id()
    store.log_run_start(run_id, source="alpaca")

    fake_client = MagicMock()
    ok = MagicMock()
    ok.data = {"AAPL": _fake_bars("AAPL")}
    fake_client.get_stock_bars.side_effect = [
        Exception('{"message":"invalid symbol: BOGUS"}'),
        ok,
    ]

    with patch.object(src, "_client", fake_client):
        result = src.fetch(["AAPL", "BOGUS"], date(2026, 4, 23), store, run_id)

    assert result.status == "ok"
    assert result.rows_inserted == 2
    retry_req = fake_client.get_stock_bars.call_args[0][0]
    assert retry_req.symbol_or_symbols == ["AAPL"]


# ---------------------------------------------------------------------------
# 2026-09-18 incident audit (yfinance_prices.py's insert path inserted NaN
# closes unconditionally): same defensive guard for alpaca, even though a NaN
# bar.close is not the observed failure mode here -- never insert a row
# whose close is NaN/None, drop it, and log one WARNING per ticker-run with
# the count.
# ---------------------------------------------------------------------------


def test_alpaca_drops_bar_with_nan_close(store):
    src = AlpacaPricesSource(
        api_key="k", api_secret="s",
        base_url="https://paper-api.alpaca.markets", lookback_days=5,
    )
    run_id = store.allocate_run_id()
    store.log_run_start(run_id, source="alpaca")

    bars = [
        FakeBar(datetime(2026, 9, 18, 16, 0), 150.0, 152.0, 149.0, float("nan"), 10_000_000),
    ]
    fake_client = MagicMock()
    fake_client.get_stock_bars.return_value.data = {"AAPL": bars}

    with patch.object(src, "_client", fake_client):
        result = src.fetch(["AAPL"], date(2026, 9, 18), store, run_id)

    assert result.rows_inserted == 0
    rows = store.conn.execute("SELECT * FROM prices WHERE ticker='AAPL'").fetchall()
    assert rows == [], "a NaN-close bar must never be inserted"


def test_alpaca_drops_bar_with_none_close(store):
    src = AlpacaPricesSource(
        api_key="k", api_secret="s",
        base_url="https://paper-api.alpaca.markets", lookback_days=5,
    )
    run_id = store.allocate_run_id()
    store.log_run_start(run_id, source="alpaca")

    bars = [FakeBar(datetime(2026, 9, 18, 16, 0), 150.0, 152.0, 149.0, None, 10_000_000)]
    fake_client = MagicMock()
    fake_client.get_stock_bars.return_value.data = {"AAPL": bars}

    with patch.object(src, "_client", fake_client):
        result = src.fetch(["AAPL"], date(2026, 9, 18), store, run_id)

    assert result.rows_inserted == 0


def test_alpaca_drop_logs_one_warning_with_count(store, monkeypatch):
    warnings = []
    monkeypatch.setattr(
        alpaca_prices_mod.logger, "warning",
        lambda *a, **kw: warnings.append(a),
    )
    src = AlpacaPricesSource(
        api_key="k", api_secret="s",
        base_url="https://paper-api.alpaca.markets", lookback_days=5,
    )
    run_id = store.allocate_run_id()
    store.log_run_start(run_id, source="alpaca")

    bars = [
        FakeBar(datetime(2026, 9, 17, 16, 0), 150.0, 152.0, 149.0, float("nan"), 10_000_000),
        FakeBar(datetime(2026, 9, 18, 16, 0), 150.0, 152.0, 149.0, float("nan"), 10_000_000),
    ]
    fake_client = MagicMock()
    fake_client.get_stock_bars.return_value.data = {"AAPL": bars}

    with patch.object(src, "_client", fake_client):
        src.fetch(["AAPL"], date(2026, 9, 18), store, run_id)

    drop_warnings = [w for w in warnings if "AAPL" in str(w) and "NaN" in str(w)]
    assert len(drop_warnings) == 1, f"expected exactly one drop warning, got: {warnings}"
    assert "2" in str(drop_warnings[0])


def test_alpaca_keeps_good_bars_alongside_dropped_nan_bar(store):
    """A NaN-close bar for one day must not sink the other good bars for the
    same ticker/run — best-effort insert, same as yfinance."""
    src = AlpacaPricesSource(
        api_key="k", api_secret="s",
        base_url="https://paper-api.alpaca.markets", lookback_days=5,
    )
    run_id = store.allocate_run_id()
    store.log_run_start(run_id, source="alpaca")

    bars = [
        FakeBar(datetime(2026, 9, 17, 16, 0), 150.0, 152.0, 149.0, 151.0, 10_000_000),
        FakeBar(datetime(2026, 9, 18, 16, 0), 150.0, 152.0, 149.0, float("nan"), 10_000_000),
    ]
    fake_client = MagicMock()
    fake_client.get_stock_bars.return_value.data = {"AAPL": bars}

    with patch.object(src, "_client", fake_client):
        result = src.fetch(["AAPL"], date(2026, 9, 18), store, run_id)

    assert result.rows_inserted == 1
    assert result.status == "ok"
