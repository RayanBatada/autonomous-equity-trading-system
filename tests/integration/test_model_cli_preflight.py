"""predict must gate on the ingest sentinel and record lineage.

2026-06-09: ingest failed (DNS, no 6/09 prices), predict at 19:30 happily
wrote 188 predictions off stale 6/08 features with a healthy-looking
sentinel. Decide gates on ingest directly, but once ingest healed later that
evening the stale predictions were the ones decide would trade on. predict
must (a) refuse to run against a failed/missing ingest (and write NO
sentinel, so the watchdog re-kicks it after ingest heals), and (b) stamp the
consumed ingest run_id into its sentinel for decide's lineage check.
"""

from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

from sma.sentinels import read_sentinel, write_sentinel

ASOF = date(2026, 4, 30)  # Thursday


def _write_ingest_sentinel(*, passed: bool, run_id: int = 42):
    write_sentinel(
        label="com.sma.ingest.daily",
        asof=ASOF,
        payload={
            "run_id": run_id,
            "quality": {
                "passed": passed,
                "blocking_failures": [] if passed else ["all_tickers_have_price"],
            },
        },
    )


def _patch_fast_predict(monkeypatch, tmp_path):
    from sma.model import __main__ as mod

    predictor = MagicMock()
    predictor.predict_for_with_model_id.return_value = ({"AAPL": 0.1}, "model_x")
    monkeypatch.setattr(mod, "Predictor", lambda **kw: predictor)
    monkeypatch.setattr(mod, "write_predictions", lambda **kw: 1)
    monkeypatch.setattr(mod, "load_universe", lambda p: ["AAPL"])
    return mod


def _invoke(mod, tmp_path, *extra):
    from sma.model.__main__ import cli

    return CliRunner().invoke(
        cli,
        [
            "predict",
            "--asof", ASOF.isoformat(),
            "--db-path", str(tmp_path / "t.duckdb"),
            "--models-dir", str(tmp_path / "models"),
            "--universe-path", str(tmp_path / "u.yaml"),
            *extra,
        ],
    )


def test_predict_refuses_on_failed_ingest_and_writes_no_sentinel(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    mod = _patch_fast_predict(monkeypatch, tmp_path)
    notifications = []
    monkeypatch.setattr(
        mod, "notify_failure",
        lambda title, message: notifications.append((title, message)),
    )
    _write_ingest_sentinel(passed=False)
    result = _invoke(mod, tmp_path)
    assert result.exit_code != 0
    assert read_sentinel(label="com.sma.model.predict.daily", asof=ASOF) is None, (
        "a refused predict must leave NO sentinel so the watchdog re-kicks it "
        "after ingest heals"
    )
    assert len(notifications) == 1


def test_predict_refuses_when_ingest_sentinel_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    mod = _patch_fast_predict(monkeypatch, tmp_path)
    monkeypatch.setattr(mod, "notify_failure", lambda title, message: None)
    # historical asof → the wait window is already past → instant refusal
    result = _invoke(mod, tmp_path)
    assert result.exit_code != 0
    assert read_sentinel(label="com.sma.model.predict.daily", asof=ASOF) is None


def test_predict_records_ingest_lineage_on_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    mod = _patch_fast_predict(monkeypatch, tmp_path)
    _write_ingest_sentinel(passed=True, run_id=42)
    result = _invoke(mod, tmp_path)
    assert result.exit_code == 0, result.output
    sentinel = read_sentinel(label="com.sma.model.predict.daily", asof=ASOF)
    assert sentinel is not None
    assert sentinel["ingest_run_id"] == 42


def test_predict_no_preflight_skips_gate_for_backfills(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    mod = _patch_fast_predict(monkeypatch, tmp_path)
    # no ingest sentinel at all — the eval/backfill path over history
    result = _invoke(mod, tmp_path, "--no-preflight")
    assert result.exit_code == 0, result.output
    sentinel = read_sentinel(label="com.sma.model.predict.daily", asof=ASOF)
    assert sentinel is not None
    assert sentinel["ingest_run_id"] is None
