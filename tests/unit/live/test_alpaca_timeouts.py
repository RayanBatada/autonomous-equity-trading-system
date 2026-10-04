"""Every Alpaca HTTP call carries a timeout (flaw hunt 2026-10-01, A2).

alpaca-py's RESTClient calls requests.Session.request(method, url, **opts)
with no timeout, so a connection that died after it was established blocked
forever. That is how the watchdog hung from 15:53 on 2026-10-01 (PID 935,
`launchctl kickstart -k` needed), and inside ingest or decide the same hang
would hold the writer lock and block the whole evening.

With a timeout the hang becomes requests.exceptions.Timeout: read-only calls
already retry it (_retry_on_transient_network_error), and decide already
turns a failed submit into status='submission_failed' that reconcile adopts
by client_order_id if the broker did accept it.
"""

from unittest.mock import MagicMock

import pytest
import requests

from sma.live import alpaca_client as ac


@pytest.fixture
def recorded(monkeypatch):
    calls = []

    def fake_request(self, method, url, **kwargs):
        calls.append(kwargs)
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.text = "[]"
        resp.json.return_value = []
        return resp

    monkeypatch.setattr(requests.Session, "request", fake_request)
    return calls


def test_trading_client_requests_carry_the_default_timeout(recorded):
    from datetime import date

    client = ac.AlpacaClient.paper_from_env(api_key="k", secret_key="s")
    assert client.sessions_between(start=date(2026, 10, 2), end=date(2026, 10, 2)) == []
    assert recorded and recorded[-1]["timeout"] == ac.ALPACA_HTTP_TIMEOUT_S


def test_live_client_gets_the_same_timeout(recorded):
    client = ac.AlpacaClient.live_from_env(api_key="k", secret_key="s")
    client.tc._session.request("GET", "https://example.invalid/v2/clock")
    assert recorded[-1]["timeout"] == ac.ALPACA_HTTP_TIMEOUT_S


def test_market_data_client_gets_the_timeout(recorded):
    client = ac.AlpacaClient.paper_from_env(api_key="k", secret_key="s")
    client.data._session.request("GET", "https://example.invalid/v2/stocks/bars")
    assert recorded[-1]["timeout"] == ac.ALPACA_HTTP_TIMEOUT_S


def test_ingest_alpaca_prices_client_gets_the_timeout(recorded):
    from sma.ingest.sources.alpaca_prices import AlpacaPricesSource

    src = AlpacaPricesSource(api_key="k", api_secret="s", base_url="https://x")
    src._client._session.request("GET", "https://example.invalid/v2/stocks/bars")
    assert recorded[-1]["timeout"] == ac.ALPACA_HTTP_TIMEOUT_S


def test_an_explicit_timeout_is_not_overridden(recorded):
    client = ac.AlpacaClient.paper_from_env(api_key="k", secret_key="s")
    client.tc._session.request("GET", "https://example.invalid/", timeout=5)
    assert recorded[-1]["timeout"] == 5


def test_installing_twice_does_not_stack(recorded):
    client = ac.AlpacaClient.paper_from_env(api_key="k", secret_key="s")
    ac.with_default_timeout(client.tc)
    ac.with_default_timeout(client.tc)
    client.tc._session.request("GET", "https://example.invalid/")
    assert len(recorded) == 1
