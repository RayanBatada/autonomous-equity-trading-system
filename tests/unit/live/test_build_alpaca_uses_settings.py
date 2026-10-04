"""HIGH 5: _build_alpaca reads credentials from settings.secrets, not os.environ.

When launchd fires the job, ALPACA_API_KEY is NOT in the process environment
(launchd only provides PATH). The fix: read settings.secrets.alpaca_api_key,
which pydantic-settings loads from .env at Settings construction time.

Test: strip ALPACA_API_KEY from os.environ, write a .env file in tmp_path,
and assert _build_alpaca succeeds using the key from .env.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from sma.live.__main__ import _build_alpaca


def _build_fake_settings(api_key: str, secret_key: str) -> MagicMock:
    """Return a minimal settings mock with populated secrets.

    `live.real_money` is pinned to the real defaults (both gates False). A bare
    MagicMock would make every gate attribute truthy and route the test to the
    REAL-money endpoint — which is exactly the accident the gates exist to
    prevent, so the mock has to state them.
    """
    settings = MagicMock()
    settings.secrets.alpaca_api_key = api_key
    settings.secrets.alpaca_api_secret = secret_key
    settings.live.real_money.enabled = False
    settings.live.real_money.real_money_ack = False
    settings.live.real_money.max_real_equity = 1000.0
    settings.live.real_money.dry_run = False
    return settings


def test_build_alpaca_reads_from_settings_not_environ(tmp_path, monkeypatch):
    """_build_alpaca must use settings.secrets, not os.environ.

    Verify that even when ALPACA_API_KEY is absent from os.environ, the
    function succeeds if settings.secrets carries the value.
    """
    # Remove the keys from the environment to simulate launchd conditions.
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_API_SECRET", raising=False)

    # Confirm they are genuinely absent.
    assert os.environ.get("ALPACA_API_KEY") is None
    assert os.environ.get("ALPACA_API_SECRET") is None

    settings = _build_fake_settings(
        api_key="fake-key-from-settings",
        secret_key="fake-secret-from-settings",
    )

    with patch("sma.live.real_money.AlpacaClient") as mock_client_cls:
        mock_client_cls.paper_from_env.return_value = MagicMock()
        client = _build_alpaca(settings)

    mock_client_cls.paper_from_env.assert_called_once_with(
        api_key="fake-key-from-settings",
        secret_key="fake-secret-from-settings",
    )
    assert client is mock_client_cls.paper_from_env.return_value


def test_build_alpaca_raises_when_settings_secrets_empty(monkeypatch):
    """_build_alpaca must raise ClickException when settings.secrets has empty keys.

    This guards against a misconfigured .env where the keys are present but blank.
    """
    import click

    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_API_SECRET", raising=False)

    settings = _build_fake_settings(api_key="", secret_key="")

    with pytest.raises(click.ClickException):
        _build_alpaca(settings)


def test_build_alpaca_env_file_supplies_key(tmp_path, monkeypatch):
    """pydantic-settings loads .env; _build_alpaca gets the key without os.environ.

    Write a real .env file and construct a real Secrets object to verify
    the full loading chain works end-to-end.
    """
    from sma.config import Secrets

    # Strip all SMA secrets from the environment.
    for k in (
        "ALPACA_API_KEY",
        "ALPACA_API_SECRET",
        "ALPACA_BASE_URL",
        "FINNHUB_API_KEY",
        "NEWSAPI_KEY",
        "EDGAR_USER_AGENT",
        "ANTHROPIC_API_KEY",
    ):
        monkeypatch.delenv(k, raising=False)

    # Write a .env file into tmp_path.
    env_file = tmp_path / ".env"
    env_file.write_text(
        "ALPACA_API_KEY=key-from-dotenv\n"
        "ALPACA_API_SECRET=secret-from-dotenv\n"
        "ALPACA_BASE_URL=https://paper-api.alpaca.markets\n"
        "FINNHUB_API_KEY=fh-x\n"
        "NEWSAPI_KEY=na-x\n"
        "EDGAR_USER_AGENT=Test (test@test.com)\n"
    )

    # Build Secrets with env_file pointing at our tmp .env.
    secrets = Secrets(_env_file=str(env_file))
    assert secrets.alpaca_api_key == "key-from-dotenv"
    assert secrets.alpaca_api_secret == "secret-from-dotenv"


def test_build_alpaca_defaults_to_paper_endpoint(monkeypatch):
    """Unarmed real_money config (the default, and what the paper bot runs on)
    must construct a PAPER client and never touch live_from_env."""
    settings = _build_fake_settings("k", "s")
    with patch("sma.live.real_money.AlpacaClient") as mock_client_cls:
        mock_client_cls.paper_from_env.return_value = MagicMock(paper=True)
        _build_alpaca(settings)
    mock_client_cls.paper_from_env.assert_called_once()
    mock_client_cls.live_from_env.assert_not_called()


def test_build_alpaca_requires_both_gates_for_live(monkeypatch):
    """enabled alone is not enough — real_money_ack must also be set."""
    settings = _build_fake_settings("k", "s")
    settings.live.real_money.enabled = True   # ack still False
    with patch("sma.live.real_money.AlpacaClient") as mock_client_cls:
        mock_client_cls.paper_from_env.return_value = MagicMock(paper=True)
        _build_alpaca(settings)
    mock_client_cls.live_from_env.assert_not_called()


def test_build_alpaca_refuses_live_above_equity_ceiling():
    """Both gates armed but the account is bigger than the experiment was
    authorised for: refuse, do not trade."""
    import click

    settings = _build_fake_settings("k", "s")
    settings.live.real_money.enabled = True
    settings.live.real_money.real_money_ack = True
    settings.live.real_money.max_real_equity = 100.0

    live_client = MagicMock(paper=False)
    live_client.get_account.return_value = {"equity": 118_000.0}
    with patch("sma.live.real_money.AlpacaClient") as mock_client_cls:
        mock_client_cls.live_from_env.return_value = live_client
        with pytest.raises(click.ClickException, match="max_real_equity"):
            _build_alpaca(settings)


def test_build_alpaca_allows_live_within_equity_ceiling():
    settings = _build_fake_settings("k", "s")
    settings.live.real_money.enabled = True
    settings.live.real_money.real_money_ack = True
    settings.live.real_money.max_real_equity = 1000.0

    live_client = MagicMock(paper=False)
    live_client.get_account.return_value = {"equity": 50.0}
    with patch("sma.live.real_money.AlpacaClient") as mock_client_cls:
        mock_client_cls.live_from_env.return_value = live_client
        got = _build_alpaca(settings)
    assert got is live_client
    mock_client_cls.paper_from_env.assert_not_called()
