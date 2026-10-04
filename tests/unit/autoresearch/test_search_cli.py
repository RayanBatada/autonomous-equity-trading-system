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
            # --no-require-retrain: these tests exercise the search WIRING without
            # staging a weekly-retrain sentinel; the retrain precondition itself is
            # covered by test_search_cli_defers_when_retrain_incomplete below.
            ["search", "--asof", "2026-06-10", "--n-configs", "3", "--seed", "1",
             "--no-require-retrain",
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
             "--no-require-retrain",
             "--models-dir", str(tmp_path)],  # no --seed
        )
    assert result.exit_code == 0, result.output
    assert gate.call_args.kwargs["seed"] == date(2026, 6, 22).toordinal()


def test_search_cli_wires_a_deadline_guard_into_select_and_gate(tmp_path):
    # 2026-08-24: the search command must always hand select_and_gate a
    # deadline_reached callable — _search_deadline_reached itself is
    # responsible for staying a no-op on historical/manual asofs (see
    # test_search_deadline.py), not the caller.
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
            ["search", "--asof", "2026-06-10", "--n-configs", "2", "--seed", "1",
             "--no-require-retrain", "--models-dir", str(tmp_path)],
        )
    assert result.exit_code == 0, result.output
    assert callable(gate.call_args.kwargs["deadline_reached"])


def test_search_cli_dry_run_writes_sentinel_but_never_saves(tmp_path):
    result, save, sentinel = _run(tmp_path, extra_args=("--dry-run",))
    assert result.exit_code == 0, result.output
    assert not save.called  # dry-run never trains/saves
    payload = sentinel.call_args.kwargs["payload"]
    assert payload["promoted"] is False
    assert payload["dry_run"] is True
    assert payload["model_id"] is None


def test_search_cli_raw_labels_flag_forces_full_search(tmp_path):
    """--raw-labels is the deliberate manual override: it must be threaded to
    select_and_gate as force=True so an explicit raw-label run still fully
    evaluates every candidate even though it's known it can't promote against
    a demean incumbent."""
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
            ["search", "--asof", "2026-06-10", "--n-configs", "3", "--seed", "1",
             "--no-require-retrain", "--raw-labels",
             "--models-dir", str(tmp_path)],
        )
    assert result.exit_code == 0, result.output
    assert gate.call_args.kwargs["force"] is True
    assert gate.call_args.kwargs["label_type"] == "raw"


def test_search_cli_no_raw_labels_flag_does_not_force(tmp_path):
    """Without --raw-labels, force must be False (the default/scheduled path
    that should skip a known-mismatched search rather than run it)."""
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
            ["search", "--asof", "2026-06-10", "--n-configs", "3", "--seed", "1",
             "--no-require-retrain",
             "--models-dir", str(tmp_path)],
        )
    assert result.exit_code == 0, result.output
    assert gate.call_args.kwargs["force"] is False
    assert gate.call_args.kwargs["label_type"] == "demean"


def test_search_cli_defers_when_retrain_incomplete(tmp_path):
    """The 07:00 launchd fire must DEFER cleanly (exit 0, no work, no sentinel)
    when the weekly retrain has not written its sentinel for the as-of date, so
    the watchdog re-kicks once retrain lands.

    Regression (Mon 2026-07-13/20/27): the Mac slept through the 04:00 retrain,
    which then ran mid-afternoon/evening holding the heavy_job_lock; autoresearch
    fired at 07:00, blocked the full 2h heavy-lock timeout, and died exit-1 with a
    cry-wolf 'missed job' page EVERY Monday. Gating in-process — not only in the
    watchdog kick — stops the wasted run and the false page. No sentinel is
    written on a defer, precisely so the watchdog still sees autoresearch as
    not-done and re-kicks it after retrain completes."""
    X, y, asof = _synthetic()
    with (
        patch("sma.ingest.universe.load_universe", return_value=["AAA", "BBB"]),
        patch("sma.model.__main__._load_prices_for_range", return_value=pd.DataFrame()),
        patch("sma.model.loader.build_training_set", return_value=(X, y, asof)),
        patch("sma.autoresearch.__main__.writer_lock"),
        # No retrain sentinel for the as-of date.
        patch("sma.sentinels.read_sentinel", return_value=None),
        patch(
            "sma.autoresearch.config_search.select_and_gate", return_value=_outcome()
        ) as gate,
        patch("sma.sentinels.write_sentinel") as sentinel,
    ):
        result = CliRunner().invoke(
            cli,
            ["search", "--asof", "2026-06-10", "--n-configs", "3", "--seed", "1",
             "--models-dir", str(tmp_path)],  # require-retrain is the default
        )
    assert result.exit_code == 0, result.output
    assert not gate.called  # deferred BEFORE the expensive CV search
    assert not sentinel.called  # no sentinel → watchdog re-kicks after retrain
    assert "defer" in result.output.lower()


# ---------------------------------------------------------------------------
# Seed ensemble wiring (2026-08-24 gap-fix). 885bf98 added the 10-seed
# prediction ensemble to the weekly retrain but left autoresearch training its
# final fit at the DEFAULT single seed and saving into the SAME models_dir --
# latest_model_for_date breaks same-date ties by created_at, so a promotion
# would silently replace a 10-seed ensemble artifact with a single-seed model.
# These pin: (1) search_cmd reads model.ensemble_seeds from config.yaml (the
# --ensemble-seeds CLI plumbing already exists in `train`, not here -- this is
# read-only config wiring, no new flag), (2) that value reaches the FINAL fit
# and its recorded cv_rmse, and (3) the saved cv_ic is the ensemble-consistent
# gate number (SearchOutcome.promoted_cv_ic), not the single-seed search-stage
# score.
# ---------------------------------------------------------------------------


def test_search_cli_passes_ensemble_seeds_from_config_to_select_and_gate(tmp_path):
    from sma.config import ModelConfig

    X, y, asof = _synthetic()
    with (
        patch("sma.ingest.universe.load_universe", return_value=["AAA", "BBB"]),
        patch("sma.model.__main__._load_prices_for_range", return_value=pd.DataFrame()),
        patch("sma.model.__main__._load_politician_trades", return_value=pd.DataFrame()),
        patch("sma.model.__main__._load_earnings_calendar", return_value=pd.DataFrame()),
        patch("sma.model.__main__._load_news_for_count_feature", return_value=pd.DataFrame()),
        patch("sma.model.loader.build_training_set", return_value=(X, y, asof)),
        patch("sma.autoresearch.__main__.writer_lock"),
        patch("sma.model.__main__.load_model_config", return_value=ModelConfig(ensemble_seeds=6)),
        patch(
            "sma.autoresearch.config_search.select_and_gate",
            return_value=_outcome(promote=False),
        ) as gate,
        patch("sma.sentinels.write_sentinel"),
    ):
        result = CliRunner().invoke(
            cli,
            ["search", "--dry-run", "--asof", "2026-06-10", "--n-configs", "3",
             "--seed", "1", "--no-require-retrain", "--models-dir", str(tmp_path)],
        )
    assert result.exit_code == 0, result.output
    assert gate.call_args.kwargs["ensemble_seeds"] == 6


def test_search_cli_defaults_ensemble_seeds_to_one_when_config_absent(tmp_path):
    """A checkout without model.ensemble_seeds in config.yaml (or an unreadable
    config) must not break the search -- load_model_config already falls back
    to ModelConfig() defaults tolerantly; verify search_cmd passes THAT value
    through rather than hardcoding its own."""
    from sma.config import ModelConfig

    X, y, asof = _synthetic()
    with (
        patch("sma.ingest.universe.load_universe", return_value=["AAA", "BBB"]),
        patch("sma.model.__main__._load_prices_for_range", return_value=pd.DataFrame()),
        patch("sma.model.__main__._load_politician_trades", return_value=pd.DataFrame()),
        patch("sma.model.__main__._load_earnings_calendar", return_value=pd.DataFrame()),
        patch("sma.model.__main__._load_news_for_count_feature", return_value=pd.DataFrame()),
        patch("sma.model.loader.build_training_set", return_value=(X, y, asof)),
        patch("sma.autoresearch.__main__.writer_lock"),
        patch("sma.model.__main__.load_model_config", return_value=ModelConfig(ensemble_seeds=1)),
        patch(
            "sma.autoresearch.config_search.select_and_gate",
            return_value=_outcome(promote=False),
        ) as gate,
        patch("sma.sentinels.write_sentinel"),
    ):
        result = CliRunner().invoke(
            cli,
            ["search", "--dry-run", "--asof", "2026-06-10", "--n-configs", "3",
             "--seed", "1", "--no-require-retrain", "--models-dir", str(tmp_path)],
        )
    assert result.exit_code == 0, result.output
    assert gate.call_args.kwargs["ensemble_seeds"] == 1


def test_search_cli_final_fit_and_cv_rmse_honor_config_ensemble_seeds(tmp_path):
    """On promote, the FINAL fit (train_xgb) and the walk_forward_cv_rmse call
    that produces the persisted cv_rmse must both receive ensemble_seeds from
    config.yaml -- the actual gap this session closes. Wraps the REAL trainer
    functions (not a mock returning a canned model) so the saved artifact is a
    real EnsembleModel."""
    import sma.model.trainer as trainer_mod
    from sma.config import ModelConfig

    X, y, asof = _synthetic()
    calls = {"train_xgb": [], "cv_rmse": []}
    real_train_xgb = trainer_mod.train_xgb
    real_cv_rmse = trainer_mod.walk_forward_cv_rmse

    def spy_train(*a, **k):
        calls["train_xgb"].append(k.get("ensemble_seeds", 1))
        return real_train_xgb(*a, **k)

    def spy_rmse(*a, **k):
        calls["cv_rmse"].append(k.get("ensemble_seeds", 1))
        return real_cv_rmse(*a, **k)

    with (
        patch("sma.ingest.universe.load_universe", return_value=["AAA", "BBB"]),
        patch("sma.model.__main__._load_prices_for_range", return_value=pd.DataFrame()),
        patch("sma.model.__main__._load_politician_trades", return_value=pd.DataFrame()),
        patch("sma.model.__main__._load_earnings_calendar", return_value=pd.DataFrame()),
        patch("sma.model.__main__._load_news_for_count_feature", return_value=pd.DataFrame()),
        patch("sma.model.loader.build_training_set", return_value=(X, y, asof)),
        patch("sma.autoresearch.__main__.writer_lock"),
        patch("sma.model.__main__.load_model_config", return_value=ModelConfig(ensemble_seeds=3)),
        patch(
            "sma.autoresearch.config_search.select_and_gate", return_value=_outcome(promote=True)
        ),
        patch.object(trainer_mod, "train_xgb", side_effect=spy_train),
        patch.object(trainer_mod, "walk_forward_cv_rmse", side_effect=spy_rmse),
        patch("sma.sentinels.write_sentinel"),
    ):
        result = CliRunner().invoke(
            cli,
            ["search", "--asof", "2026-06-10", "--n-configs", "3", "--seed", "1",
             "--no-require-retrain", "--models-dir", str(tmp_path)],
        )
    assert result.exit_code == 0, result.output
    assert calls["train_xgb"] == [3]
    assert calls["cv_rmse"] == [3]

    from sma.model.ensemble import EnsembleModel
    from sma.model.persistence import load_model

    pkls = list(tmp_path.glob("xgb_ret_30d_forward_*.pkl"))
    assert len(pkls) == 1, pkls
    saved = load_model(pkls[0])
    assert isinstance(saved, EnsembleModel)
    assert len(saved) == 3


def test_search_cli_saves_ensemble_consistent_cv_ic(tmp_path):
    """The saved artifact's cv_ic must be the ensemble-consistent gate number
    (SearchOutcome.promoted_cv_ic), not the single-seed search-stage score --
    or a future incumbent re-measurement compares against a stale number."""
    best = ConfigResult(params={"max_depth": 4, "learning_rate": 0.05}, cv_ic=0.05, rank=0)
    outcome = SearchOutcome(
        best=best,
        decision=PromotionDecision(promote=True, reason="ensemble stub"),
        incumbent_cv_ic=None, n_configs_evaluated=1, results=[best],
        promoted_cv_ic=0.071,  # deliberately different from best.cv_ic
    )
    X, y, asof = _synthetic()
    with (
        patch("sma.ingest.universe.load_universe", return_value=["AAA", "BBB"]),
        patch("sma.model.__main__._load_prices_for_range", return_value=pd.DataFrame()),
        patch("sma.model.__main__._load_politician_trades", return_value=pd.DataFrame()),
        patch("sma.model.__main__._load_earnings_calendar", return_value=pd.DataFrame()),
        patch("sma.model.__main__._load_news_for_count_feature", return_value=pd.DataFrame()),
        patch("sma.model.loader.build_training_set", return_value=(X, y, asof)),
        patch("sma.autoresearch.__main__.writer_lock"),
        patch("sma.autoresearch.config_search.select_and_gate", return_value=outcome),
        patch(
            "sma.model.persistence.save_model",
            return_value=(tmp_path / "xgb_x.pkl", tmp_path / "xgb_x.json"),
        ) as save,
        patch("sma.sentinels.write_sentinel"),
    ):
        result = CliRunner().invoke(
            cli,
            ["search", "--asof", "2026-06-10", "--n-configs", "3", "--seed", "1",
             "--no-require-retrain", "--models-dir", str(tmp_path)],
        )
    assert result.exit_code == 0, result.output
    assert save.call_args.kwargs["cv_ic"] == 0.071


def test_search_cli_saved_cv_ic_falls_back_to_search_score_when_gate_ic_absent(tmp_path):
    """Backward-compat fallback: a SearchOutcome without promoted_cv_ic (e.g.
    ensemble_seeds=1, or an older test double) must save best.cv_ic exactly as
    before this wiring existed."""
    result, save, sentinel = _run(tmp_path)
    assert result.exit_code == 0, result.output
    assert save.call_args.kwargs["cv_ic"] == 0.05  # best.cv_ic from _outcome()


def test_search_cli_runs_when_retrain_sentinel_present(tmp_path):
    """With require-retrain on (the default), a present retrain sentinel lets the
    search proceed normally — the gate never blocks a completed retrain."""
    X, y, asof = _synthetic()
    with (
        patch("sma.ingest.universe.load_universe", return_value=["AAA", "BBB"]),
        patch("sma.model.__main__._load_prices_for_range", return_value=pd.DataFrame()),
        patch("sma.model.__main__._load_politician_trades", return_value=pd.DataFrame()),
        patch("sma.model.__main__._load_earnings_calendar", return_value=pd.DataFrame()),
        patch("sma.model.__main__._load_news_for_count_feature", return_value=pd.DataFrame()),
        patch("sma.model.loader.build_training_set", return_value=(X, y, asof)),
        patch("sma.autoresearch.__main__.writer_lock"),
        # Retrain sentinel present for the as-of date.
        patch("sma.sentinels.read_sentinel", return_value={"asof": "2026-06-10"}),
        patch(
            "sma.autoresearch.config_search.select_and_gate", return_value=_outcome()
        ) as gate,
        patch(
            "sma.model.persistence.save_model",
            return_value=(tmp_path / "xgb_x.pkl", tmp_path / "xgb_x.json"),
        ),
        patch("sma.sentinels.write_sentinel") as sentinel,
    ):
        result = CliRunner().invoke(
            cli,
            ["search", "--asof", "2026-06-10", "--n-configs", "3", "--seed", "1",
             "--models-dir", str(tmp_path)],  # require-retrain is the default
        )
    assert result.exit_code == 0, result.output
    assert gate.called  # retrain done -> search proceeds
    assert sentinel.called
