"""A dead thesis layer pages the night it happens.

2026-09-30 19:48 the Anthropic credit ran out; 10/1 the agents job failed
23 of 23 tickers and wrote quality.passed=false, exited 0, and nothing
reached a phone. decide treats agents as advisory (correct: a dead overlay
must not freeze trading), so it traded on the model alone and the only trace
was a WARNING in live.decide.err.log. The ingest-time health check could not
catch it either: it reads the agents sentinel for the same date at 18:30,
before agents run at 19:45 (flaw hunt 2026-10-01 B7).
"""

from datetime import date
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from sma.agents.__main__ import cli
from tests.integration.agents.test_agents_writes_sentinel import (
    _fake_settings,
    _make_writer_lock_factory,
)

CREDIT = (
    "Error code: 400 - {'type': 'error', 'error': {'type': 'invalid_request_error', "
    "'message': 'Your credit balance is too low to access the Anthropic API.'}}"
)


def _run(tmp_path, monkeypatch, side_effect, tickers=("AAPL", "MSFT")):
    lock_path = tmp_path / ".sma-writer.lock"
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path / "sentinels"))
    monkeypatch.setattr("sma.locks.DEFAULT_LOCK_PATH", lock_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text("{}\n")
    universe_path = tmp_path / "universe.yaml"
    universe_path.write_text(f"tickers: [{', '.join(tickers)}]\n")
    fake_pipeline = MagicMock()
    fake_pipeline.run.side_effect = side_effect
    pages = []
    with (
        patch("sma.agents.__main__.load_settings", return_value=_fake_settings()),
        patch("sma.agents.__main__._build_pipeline", return_value=fake_pipeline),
        patch("sma.agents.__main__._build_context", return_value=MagicMock()),
        patch("sma.agents.__main__.load_universe", return_value=list(tickers)),
        patch("sma.agents.__main__.writer_lock", _make_writer_lock_factory(lock_path)),
        patch(
            "sma.agents.__main__.notify_failure",
            side_effect=lambda **kw: pages.append(kw),
        ),
    ):
        result = CliRunner().invoke(
            cli,
            [
                "run", "--asof-date", date(2026, 10, 1).isoformat(), "--force-full",
                "--db", str(tmp_path / "test.duckdb"), "--config", str(config_path),
                "--universe", str(universe_path),
            ],
            catch_exceptions=False,
        )
    assert result.exit_code == 0, result.output
    return pages


def test_all_tickers_failing_pages_with_the_error(tmp_path, monkeypatch):
    pages = _run(tmp_path, monkeypatch, RuntimeError(CREDIT))
    assert len(pages) == 1
    assert "agents" in pages[0]["title"].lower()
    msg = pages[0]["message"]
    assert "0 of 2" in msg
    assert "credit balance is too low" in msg
    assert "without fresh theses" in msg


def test_majority_failing_pages(tmp_path, monkeypatch):
    pages = _run(
        tmp_path,
        monkeypatch,
        [MagicMock(conviction="bullish", score=0.6), RuntimeError("boom"), RuntimeError("boom")],
        tickers=("AAPL", "MSFT", "GOOG"),
    )
    assert len(pages) == 1
    assert "1 of 3" in pages[0]["message"]


def test_healthy_run_does_not_page(tmp_path, monkeypatch):
    pages = _run(
        tmp_path,
        monkeypatch,
        [MagicMock(conviction="bullish", score=0.6), MagicMock(conviction="neutral", score=0.0)],
    )
    assert pages == []


def test_a_minority_of_failures_does_not_page(tmp_path, monkeypatch):
    pages = _run(
        tmp_path,
        monkeypatch,
        [
            MagicMock(conviction="bullish", score=0.6),
            MagicMock(conviction="bullish", score=0.6),
            RuntimeError("boom"),
        ],
        tickers=("AAPL", "MSFT", "GOOG"),
    )
    assert pages == []
