"""Codex branch review (2026-06-09):

1. The holiday-skip ingest sentinel is quality.passed=True with run_id=None.
   Launchd still fires predict/decide on weekday market holidays; the lineage
   chain treated the holiday sentinel as READY (missing run_ids are
   lineage-tolerant), so predict would write stale-feature predictions and
   decide would submit orders on a closed day. Both must skip QUIETLY
   (exit 0, no sentinel, no page — monitoring/watchdog already have their own
   holiday guards).

2. The late-start grace window had no absolute cutoff: a RunAtLoad/manual
   decide hours past its deadline could still submit. Bound it at
   deadline + 6h (symmetric with the watchdog's too-late-to-kick rule).
"""

from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from sma import schedule as sched
from sma.live.preflight import (
    PreflightAbortError,
    PreflightHolidaySkipped,
    run_preflight,
)
from sma.sentinels import read_sentinel, write_sentinel

ASOF = date(2026, 4, 30)  # Thursday


def _mock_alpaca():
    alpaca = MagicMock()
    alpaca.next_session_date.return_value = date(2026, 5, 1)
    return alpaca


def _write_holiday_ingest_sentinel(asof=ASOF):
    write_sentinel(
        label="com.sma.ingest.daily",
        asof=asof,
        payload={
            "label": "com.sma.ingest.daily",
            "asof": asof.isoformat(),
            "completed_at": "2026-04-30T22:35:00Z",
            "quality": {"passed": True, "blocking_failures": [], "checks": []},
            "holiday_skipped": True,
            "run_id": None,
        },
    )


# --- decide preflight ----------------------------------------------------------


def test_preflight_raises_holiday_skip_on_holiday_ingest_sentinel(tmp_path, monkeypatch):
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path / "s"))
    _write_holiday_ingest_sentinel()
    with pytest.raises(PreflightHolidaySkipped):
        run_preflight(asof=ASOF, db_path=Path("x"), alpaca=_mock_alpaca(), max_wait_s=0)


def test_decide_cli_exits_zero_and_quiet_on_holiday(tmp_path, monkeypatch):
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path / "s"))
    config = tmp_path / "config.yaml"
    config.write_text("{}\n")
    universe = tmp_path / "u.yaml"
    universe.write_text("tickers: []\n")
    from sma.live.__main__ import decide as decide_cmd

    notifications = []
    with (
        patch("sma.live.__main__._build_alpaca", return_value=MagicMock()),
        patch("sma.live.__main__._build_rails", return_value=MagicMock()),
        patch("sma.live.__main__.load_settings", return_value=MagicMock()),
        patch("sma.live.__main__.load_universe", return_value=["AAPL"]),
        patch(
            "sma.live.__main__.run_preflight",
            side_effect=PreflightHolidaySkipped("market holiday"),
        ),
        patch(
            "sma.live.__main__.notify_failure",
            side_effect=lambda title, message: notifications.append((title, message)),
        ),
    ):
        result = CliRunner().invoke(
            decide_cmd,
            [
                "--asof-date", ASOF.isoformat(),
                "--db", str(tmp_path / "t.duckdb"),
                "--config", str(config),
                "--universe", str(universe),
            ],
        )
    assert result.exit_code == 0, result.output
    assert "holiday" in result.output.lower()
    assert notifications == []  # a closed market must not page anyone
    assert read_sentinel(label="com.sma.live.decide.daily", asof=ASOF) is None


# --- predict gate ---------------------------------------------------------------


def test_predict_skips_quietly_on_holiday_sentinel(tmp_path, monkeypatch):
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path / "s"))
    _write_holiday_ingest_sentinel()
    from sma.model import __main__ as mod

    notifications = []
    monkeypatch.setattr(
        mod, "notify_failure",
        lambda title, message: notifications.append((title, message)),
    )
    predictor = MagicMock()
    monkeypatch.setattr(mod, "Predictor", lambda **kw: predictor)
    monkeypatch.setattr(mod, "load_universe", lambda p: ["AAPL"])

    from sma.model.__main__ import cli

    result = CliRunner().invoke(
        cli,
        [
            "predict",
            "--asof", ASOF.isoformat(),
            "--db-path", str(tmp_path / "t.duckdb"),
            "--models-dir", str(tmp_path / "m"),
            "--universe-path", str(tmp_path / "u.yaml"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "holiday" in result.output.lower()
    assert notifications == []
    assert read_sentinel(label="com.sma.model.predict.daily", asof=ASOF) is None
    predictor.predict_for_with_model_id.assert_not_called()


# --- absolute late cutoff -------------------------------------------------------


def test_preflight_aborts_when_started_past_absolute_cutoff(tmp_path, monkeypatch):
    """decide deadline is 21:00; deadline+6h = 03:00 next day. A 03:30 start
    (RunAtLoad recovery, manual run) must fail closed instead of submitting."""
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path / "s"))
    for label in (
        "com.sma.ingest.daily",
        "com.sma.model.predict.daily",
        "com.sma.agents.daily",
    ):
        write_sentinel(
            label=label,
            asof=ASOF,
            payload={
                "run_id": 1,
                "quality": {"passed": True, "blocking_failures": []},
            },
        )
    way_late = datetime.combine(
        ASOF + timedelta(days=1), datetime.min.time(), tzinfo=sched.NY_TZ
    ).replace(hour=3, minute=30)
    with pytest.raises(PreflightAbortError, match="too late"):
        run_preflight(
            asof=ASOF,
            db_path=Path("x"),
            alpaca=_mock_alpaca(),
            max_wait_s=600,
            now_fn=lambda: way_late,
        )
