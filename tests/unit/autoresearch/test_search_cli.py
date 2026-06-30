"""CLI test for `python -m sma.autoresearch search`.

Orchestration only: select_and_gate (the search + gate, whose real math is
covered in test_config_search.py) is mocked to a controlled outcome, so this
test verifies the command's WIRING — best config trained + saved on promote,
nothing saved on dry-run, and a sentinel written every time — without a DB or a
walk-forward CV.
"""
from unittest.mock import patch

import numpy as np
import pandas as pd
from click.testing import CliRunner

from sma.autoresearch.__main__ import cli
from sma.autoresearch.config_search import ConfigResult, SearchOutcome
from sma.autoresearch.promotion import PromotionDecision


def _synthetic(n_dates=12, n_names=20, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_dates).date
    rows, ys, asofs = [], [], []
    for d in dates:
        f0 = rng.normal(size=n_names)
        for v in f0:
            rows.append({"f0": float(v), "f1": float(rng.normal())})
            ys.append(float(v + rng.normal(scale=0.5)))
            asofs.append(d)
    return pd.DataFrame(rows), pd.Series(ys), pd.Series(asofs)


def _outcome(promote=True):
    best = ConfigResult(params={"max_depth": 4, "learning_rate": 0.05}, cv_ic=0.05, rank=0)
    other = ConfigResult(params={"max_depth": 6, "learning_rate": 0.1}, cv_ic=0.03, rank=1)
    return SearchOutcome(
        best=best,
        decision=PromotionDecision(promote=promote, reason="CV-IC +0.0500 beats incumbent"),
        incumbent_cv_ic=None,
        n_configs_evaluated=2,
        results=[best, other],
    )


def _run(tmp_path, extra_args=()):
    X, y, asof = _synthetic()
    with (
        patch("sma.ingest.universe.load_universe", return_value=["AAA", "BBB"]),
        patch("sma.model.__main__._load_prices_for_range", return_value=pd.DataFrame()),
        patch("sma.model.__main__._load_politician_trades", return_value=pd.DataFrame()),
        patch("sma.model.__main__._load_earnings_calendar", return_value=pd.DataFrame()),
        patch("sma.model.__main__._load_news_for_count_feature", return_value=pd.DataFrame()),
        patch("sma.model.loader.build_training_set", return_value=(X, y, asof)),
        patch("sma.autoresearch.__main__.writer_lock"),
        patch("sma.autoresearch.config_search.select_and_gate", return_value=_outcome()),
        patch(
            "sma.model.persistence.save_model",
            return_value=(tmp_path / "xgb_x.pkl", tmp_path / "xgb_x.json"),
        ) as save,
        patch("sma.sentinels.write_sentinel") as sentinel,
    ):
        result = CliRunner().invoke(
            cli,
            ["search", "--asof", "2026-06-10", "--n-configs", "3", "--seed", "1",
             "--models-dir", str(tmp_path), *extra_args],
        )
    return result, save, sentinel


def test_search_cli_promotes_and_writes_sentinel(tmp_path):
    result, save, sentinel = _run(tmp_path)
    assert result.exit_code == 0, result.output
    assert save.called  # promoted -> a model was trained + saved
    assert save.call_args.kwargs["promoted"] is True
    payload = sentinel.call_args.kwargs["payload"]
    # Sentinel label MUST match the launchd job label the watchdog/dashboard
    # already monitor (sma.schedule: com.sma.autoresearch.nightly), else the
    # watchdog never sees it and false-pages every Monday.
    assert sentinel.call_args.kwargs["label"] == "com.sma.autoresearch.nightly"
    assert payload["kind"] == "config_search"
    assert payload["promoted"] is True
    assert payload["best_cv_ic"] == 0.05
    # counterfactual record: every evaluated config is in the sentinel
    assert len(payload["configs"]) == 2
    assert payload["configs"][0]["rank"] == 0
    assert payload["configs"][0]["cv_ic"] == 0.05
    assert "params" in payload["configs"][1]
    assert payload["model_id"] == "xgb_x"


def test_search_cli_default_seed_is_date_derived(tmp_path):
    # omitting --seed must default the seed to the as-of date's ordinal, so a
    # weekly schedule explores different configs each run
    from datetime import date

    X, y, asof = _synthetic()
    with (
        patch("sma.ingest.universe.load_universe", return_value=["AAA", "BBB"]),
        patch("sma.model.__main__._load_prices_for_range", return_value=pd.DataFrame()),
        patch("sma.model.__main__._load_politician_trades", return_value=pd.DataFrame()),
        patch("sma.model.__main__._load_earnings_calendar", return_value=pd.DataFrame()),
        patch("sma.model.__main__._load_news_for_count_feature", return_value=pd.DataFrame()),
        patch("sma.model.loader.build_training_set", return_value=(X, y, asof)),
        patch("sma.autoresearch.__main__.writer_lock"),
        patch(
            "sma.autoresearch.config_search.select_and_gate",
            return_value=_outcome(promote=False),
        ) as gate,
        patch("sma.sentinels.write_sentinel"),
    ):
        result = CliRunner().invoke(
            cli,
            ["search", "--dry-run", "--asof", "2026-06-22", "--n-configs", "2",
             "--models-dir", str(tmp_path)],  # no --seed
        )
    assert result.exit_code == 0, result.output
    assert gate.call_args.kwargs["seed"] == date(2026, 6, 22).toordinal()


def test_search_cli_dry_run_writes_sentinel_but_never_saves(tmp_path):
    result, save, sentinel = _run(tmp_path, extra_args=("--dry-run",))
    assert result.exit_code == 0, result.output
    assert not save.called  # dry-run never trains/saves
    payload = sentinel.call_args.kwargs["payload"]
    assert payload["promoted"] is False
    assert payload["dry_run"] is True
    assert payload["model_id"] is None
