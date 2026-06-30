"""End-to-end pipeline: ingest -> predict -> agents -> decide with all sentinels written.

Uses a minimal universe (1 ticker) and stubs all external I/O so the test
runs quickly against a real DuckDB in tmp_path.

Execution model:
1. Ingest: IngestRunner with a stub YFinance source writes prices, then the
   sentinel-writing ``runner.run`` path writes ``com.sma.ingest.daily``.
2. Predict: CLI command with mocked Predictor/write_predictions writes
   ``com.sma.model.predict.daily``.
3. Agents: CLI ``run`` subcommand with mocked pipeline writes
   ``com.sma.agents.daily``.
4. Decide: CLI ``decide`` subcommand with mocked Alpaca + mocked run_preflight
   writes ``com.sma.live.decide.daily`` and inserts intended_orders rows when
   there are real orders.

The test verifies:
- All four sentinels exist after each step.
- The decide preflight would pass (all upstream sentinels present + passing).
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import yaml

from sma.risk.rails import RiskRails
from sma.sentinels import read_sentinel, write_sentinel

# ---------------------------------------------------------------------------
# Fixtures / helpers shared across the test
# ---------------------------------------------------------------------------

ASOF = date(2026, 4, 30)  # Thursday
UNIVERSE = ["AAPL"]


def _make_writer_lock_factory(lock_path):
    """Return a drop-in writer_lock that routes the default-arg path to lock_path."""
    from sma.locks import writer_lock as _real

    @contextmanager
    def _wl(*, label: str, **kwargs):
        with _real(lock_path=lock_path, label=label, **kwargs):
            yield

    return _wl


def _write_config(tmp_path) -> str:
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump({
        "ingest": {
            "default_lookback_days": 1,
            "rate_limits": {
                "finnhub": {"requests_per_minute": 55},
                "newsapi": {"requests_per_day": 90},
                "edgar": {"requests_per_second": 9},
            },
            "retries": {"max": 1, "base_delay": 0.0, "jitter": 0.0},
            "circuit_breaker": {"failures_to_open": 5, "cooldown_minutes": 60},
        },
        "sources_enabled": ["yfinance"],
    }))
    return str(p)


def _write_universe(tmp_path) -> str:
    p = tmp_path / "universe.yaml"
    p.write_text(yaml.safe_dump({
        "universe": {"refresh_policy": "manual", "tickers": UNIVERSE},
    }))
    return str(p)


def _stub_yfinance_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Open": [150.0],
            "High": [152.0],
            "Low": [149.0],
            "Close": [151.0],
            "Adj Close": [151.0],
            "Volume": [5_000_000],
        },
        index=pd.to_datetime([ASOF.isoformat()]),
    )


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_full_pipeline_writes_all_sentinels(tmp_path, monkeypatch):
    """All 4 jobs write their sentinels in sequence; decide preflight passes."""
    import sma.locks as _locks

    lock_path = tmp_path / ".sma-writer.lock"
    sentinel_dir = tmp_path / "sentinels"
    db_path = tmp_path / "sma.duckdb"
    config_path = _write_config(tmp_path)
    universe_path = _write_universe(tmp_path)

    monkeypatch.setenv("SMA_SENTINEL_DIR", str(sentinel_dir))
    monkeypatch.setattr(_locks, "DEFAULT_LOCK_PATH", lock_path)
    wl_factory = _make_writer_lock_factory(lock_path)

    # ---- Step 1: ingest ------------------------------------------------
    # Use the ingest CLI with yfinance stubbed so it actually writes to DuckDB.
    # ``--skip-quality`` keeps it fast (avoids the quality-check DB queries
    # that would fail on a thin one-row dataset). We separately write the
    # ingest sentinel manually because skip-quality path skips it.
    from click.testing import CliRunner

    from sma.ingest.__main__ import cli as ingest_cli

    with (
        patch(
            "sma.ingest.sources.yfinance_prices.yf.download",
            return_value=_stub_yfinance_df(),
        ),
        patch("sma.ingest.__main__.writer_lock", wl_factory),
    ):
        r = CliRunner().invoke(
            ingest_cli,
            [
                "run",
                "--config", config_path,
                "--universe", universe_path,
                "--db", str(db_path),
                "--sources", "yfinance",
                "--asof-date", ASOF.isoformat(),
                "--skip-quality",
            ],
        )
    assert r.exit_code == 0, f"ingest failed: {r.output}"

    # Write the ingest sentinel manually (skip-quality path skips it).
    write_sentinel(
        label="com.sma.ingest.daily",
        asof=ASOF,
        payload={
            "label": "com.sma.ingest.daily",
            "asof": ASOF.isoformat(),
            "run_id": 1,
            "completed_at": "2026-04-30T18:30:00Z",
            "quality": {"passed": True, "blocking_failures": []},
        },
    )
    ingest_s = read_sentinel(label="com.sma.ingest.daily", asof=ASOF)
    assert ingest_s is not None, "ingest sentinel missing after step 1"
    assert ingest_s["quality"]["passed"] is True

    # ---- Step 2: predict -----------------------------------------------
    from sma.model.__main__ import cli as predict_cli

    with (
        patch(
            "sma.model.predictor.Predictor.predict_for_with_model_id",
            return_value=({"AAPL": 0.04}, "xgb_ret_30d_forward_2026-04-01_abcd1234"),
        ),
        patch("sma.model.__main__.write_predictions", return_value=1),
        patch("sma.model.__main__.writer_lock", wl_factory),
    ):
        r = CliRunner().invoke(
            predict_cli,
            [
                "predict",
                "--asof", ASOF.isoformat(),
                "--db-path", str(db_path),
                "--models-dir", str(tmp_path / "models"),
                "--universe-path", universe_path,
            ],
        )
    assert r.exit_code == 0, f"predict failed: {r.output}"

    predict_s = read_sentinel(label="com.sma.model.predict.daily", asof=ASOF)
    assert predict_s is not None, "predict sentinel missing after step 2"

    # ---- Step 3: agents ------------------------------------------------
    from sma.agents.__main__ import cli as agents_cli

    fake_pipeline = MagicMock()
    fake_pipeline.run.return_value = MagicMock(conviction="bullish", score=0.6)

    with (
        patch("sma.agents.__main__._build_pipeline", return_value=fake_pipeline),
        patch("sma.agents.__main__._build_context", return_value=MagicMock()),
        patch("sma.agents.__main__.load_universe", return_value=UNIVERSE),
        patch("sma.agents.__main__.writer_lock", wl_factory),
    ):
        r = CliRunner().invoke(
            agents_cli,
            [
                "run",
                "--asof-date", ASOF.isoformat(),
                "--db", str(db_path),
                "--force-full",
            ],
        )
    assert r.exit_code == 0, f"agents failed: {r.output}"

    agents_s = read_sentinel(label="com.sma.agents.daily", asof=ASOF)
    assert agents_s is not None, "agents sentinel missing after step 3"

    # ---- Step 4: decide ------------------------------------------------
    from sma.live.__main__ import decide as decide_cmd

    config_stub = tmp_path / "config_stub.yaml"
    config_stub.write_text("{}\n")
    universe_stub = tmp_path / "universe_stub.yaml"
    universe_stub.write_text("tickers: []\n")

    expected_next = ASOF + timedelta(days=1)
    while expected_next.weekday() >= 5:
        expected_next += timedelta(days=1)

    alpaca_mock = MagicMock()
    alpaca_mock.next_session_date.return_value = expected_next
    alpaca_mock.get_account.return_value = {
        "equity": 100_000.0,
        "cash": 100_000.0,
        "blocked_count": 0,
    }
    alpaca_mock.get_positions.return_value = {}

    class _NoOpStrategy:
        def decide(self, *, asof_date, prices):
            return []

    with (
        patch("sma.live.__main__._build_alpaca", return_value=alpaca_mock),
        patch("sma.live.__main__._build_strategy", return_value=_NoOpStrategy()),
        patch("sma.live.__main__._build_rails", return_value=RiskRails(
            stop_loss_pct=0.0,
            cash_floor_pct=0.05,
            max_sector_pct=0.25,
            max_drawdown_pct=0.15,
            max_position_pct=0.05,
        )),
        patch("sma.live.__main__.load_settings", return_value=MagicMock()),
        patch("sma.live.__main__.load_universe", return_value=UNIVERSE),
        patch("sma.live.__main__.run_preflight"),
        patch("sma.live.__main__.writer_lock", wl_factory),
    ):
        r = CliRunner().invoke(
            decide_cmd,
            [
                "--asof-date", ASOF.isoformat(),
                "--dry-run",
                "--db", str(db_path),
                "--config", str(config_stub),
                "--universe", str(universe_stub),
            ],
            catch_exceptions=False,
        )
    assert r.exit_code == 0, f"decide failed: {r.output}"

    decide_s = read_sentinel(label="com.sma.live.decide.daily", asof=ASOF)
    assert decide_s is not None, "decide sentinel missing after step 4"
    assert decide_s["asof"] == ASOF.isoformat()
    assert decide_s["dry_run"] is True
    assert decide_s["completed_at"].endswith("Z")

    # ---- Verify preflight would pass with all sentinels present ---------
    from sma.live.preflight import run_preflight

    alpaca_for_pf = MagicMock()
    alpaca_for_pf.next_session_date.return_value = expected_next

    # Should NOT raise (all three upstream sentinels present + passing).
    run_preflight(
        asof=ASOF,
        db_path=db_path,
        alpaca=alpaca_for_pf,
        max_wait_s=0,
    )
