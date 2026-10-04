

def test_train_default_start_env_override(monkeypatch):
    """SMA_TRAIN_START overrides the 2023 default (multi-regime experiment)."""
    from datetime import date

    from sma.model.__main__ import _train_default_start

    monkeypatch.delenv("SMA_TRAIN_START", raising=False)
    assert _train_default_start() == date(2018, 1, 1)
    monkeypatch.setenv("SMA_TRAIN_START", "2018-01-01")
    assert _train_default_start() == date(2018, 1, 1)


# ---------------------------------------------------------------------------
# PIT-universe wiring (activated 2026-08-17). universe.yaml's `added` dates are
# operational April-2026 file dates, so a retrain that doesn't pass a
# membership map trains as if today's universe had always existed — the
# survivorship the breadth study measured at +0.76pp of 30d top-15 excess
# overall and +2.73pp on 2023+. These cover the WIRING, not the filter maths
# (that's tests/unit/model/test_loader.py).
# ---------------------------------------------------------------------------

def _run_train(tmp_path, extra_args=()):
    """Invoke `train` far enough to capture the membership kwarg.

    build_training_set is stubbed to an empty frame so the command aborts right
    after the call — no DB, no CV, no model written.
    """
    from contextlib import nullcontext
    from unittest.mock import patch

    import pandas as pd
    from click.testing import CliRunner

    from sma.model.__main__ import cli

    captured = {}

    def _fake_build(**kwargs):
        captured.update(kwargs)
        return pd.DataFrame(), pd.Series(dtype=float), pd.Series(dtype="object")

    with (
        patch("sma.model.__main__._load_prices_for_range", return_value=pd.DataFrame()),
        patch("sma.model.__main__._load_politician_trades", return_value=pd.DataFrame()),
        patch("sma.model.__main__._load_earnings_calendar", return_value=pd.DataFrame()),
        patch("sma.model.__main__._load_news_for_count_feature", return_value=pd.DataFrame()),
        patch("sma.model.__main__.heavy_job_lock", lambda *a, **k: nullcontext()),
        patch("sma.model.__main__.build_training_set", side_effect=_fake_build),
    ):
        result = CliRunner().invoke(
            cli,
            ["train", "--asof", "2026-06-10", "--no-cv",
             "--models-dir", str(tmp_path), *extra_args],
        )
    return result, captured


def test_train_passes_pit_membership_by_default(tmp_path):
    """The shipped universe_history.yaml must actually reach the loader — the
    whole point-in-time path was inert while nothing passed `membership`."""
    from datetime import date

    _result, captured = _run_train(tmp_path)
    membership = captured["membership"]
    assert membership, "train must pass the PIT membership map by default"
    # A 2023+ addition is restricted; a long-standing member is not.
    assert membership["UBER"] == (date(2023, 12, 31), None)
    assert membership["AAPL"] == (None, None)


def test_train_no_pit_universe_restores_old_behavior(tmp_path):
    _result, captured = _run_train(tmp_path, ["--no-pit-universe"])
    assert captured["membership"] == {}


def test_train_missing_history_file_is_backward_compatible(tmp_path):
    """A checkout without universe_history.yaml trains exactly as it did
    before the file existed: no membership restriction, no error."""
    _result, captured = _run_train(
        tmp_path, ["--universe-history-path", str(tmp_path / "nope.yaml")],
    )
    assert captured["membership"] == {}


# ---------------------------------------------------------------------------
# Seed ensemble (2026-08-24): the training set is built ONCE and shared.
#
# THE performance invariant of this feature. Building the training set is the
# retrain's dominant phase (65-80 min in production); an XGBoost fit on it is
# seconds. The ensemble must therefore repeat ONLY the fit. If a refactor ever
# moved the seed loop outward — around build_training_set instead of around
# the fit — the retrain would go from ~80 min to ~13 HOURS and still look
# correct in every other test. These pin the call counts so that regression
# cannot land silently.
# ---------------------------------------------------------------------------

# Populated by the last _run_full_train call: the kwargs `train` actually
# handed build_training_set. Lets a test assert on what was passed without
# changing _run_full_train's return shape.
_LAST_BUILD_KWARGS: dict = {}


def _run_full_train(tmp_path, extra_args=()):
    """Invoke `train` ALL the way through fitting and saving, on tiny
    synthetic data, counting training-set builds and individual XGBoost fits.

    Returns (result, n_builds, n_fits, models_dir).
    """
    from contextlib import nullcontext
    from unittest.mock import patch

    import numpy as np
    import pandas as pd
    from click.testing import CliRunner

    import sma.model.trainer as trainer_mod
    from sma.model.__main__ import cli

    rng = np.random.default_rng(0)
    n_rows, n_dates = 60, 10
    from sma.features.builder import FEATURE_NAMES
    X = pd.DataFrame(
        rng.standard_normal((n_rows, len(FEATURE_NAMES))), columns=FEATURE_NAMES,
    )
    y = pd.Series(rng.standard_normal(n_rows))
    from datetime import date as _date
    from datetime import timedelta as _td
    asof_dates = pd.Series(
        np.repeat(
            [_date(2026, 1, 1) + _td(days=i) for i in range(n_dates)],
            n_rows // n_dates,
        )
    )

    counts = {"builds": 0, "fits": 0}
    real_fit_one = trainer_mod._fit_one

    def _fake_build(**_kwargs):
        counts["builds"] += 1
        _LAST_BUILD_KWARGS.clear()
        _LAST_BUILD_KWARGS.update(_kwargs)
        return X, y, asof_dates

    def _counting_fit_one(*args, **kwargs):
        counts["fits"] += 1
        return real_fit_one(*args, **kwargs)

    with (
        patch("sma.model.__main__._load_prices_for_range", return_value=pd.DataFrame()),
        patch("sma.model.__main__._load_politician_trades", return_value=pd.DataFrame()),
        patch("sma.model.__main__._load_earnings_calendar", return_value=pd.DataFrame()),
        patch("sma.model.__main__._load_news_for_count_feature", return_value=pd.DataFrame()),
        patch("sma.model.__main__.heavy_job_lock", lambda *a, **k: nullcontext()),
        patch("sma.model.__main__.writer_lock", lambda *a, **k: nullcontext()),
        patch("sma.model.__main__.write_sentinel"),
        patch("sma.model.__main__._current_git_sha", return_value="testsha1"),
        patch("sma.model.__main__.build_training_set", side_effect=_fake_build),
        patch.object(trainer_mod, "_fit_one", side_effect=_counting_fit_one),
    ):
        result = CliRunner().invoke(
            cli,
            ["train", "--asof", "2026-06-10", "--no-cv",
             "--models-dir", str(tmp_path), *extra_args],
        )
    return result, counts["builds"], counts["fits"], tmp_path


def test_train_builds_the_training_set_once_for_ten_seeds(tmp_path):
    """THE performance guard: 10 seeds, ONE training-set build, TEN fits."""
    result, n_builds, n_fits, _ = _run_full_train(tmp_path, ["--ensemble-seeds", "10"])

    assert result.exit_code == 0, result.output
    assert n_builds == 1, f"training set built {n_builds}x for a 10-seed ensemble"
    assert n_fits == 10, f"expected 10 XGBoost fits, got {n_fits}"


def test_train_builds_the_training_set_once_for_one_seed(tmp_path):
    result, n_builds, n_fits, _ = _run_full_train(tmp_path, ["--ensemble-seeds", "1"])

    assert result.exit_code == 0, result.output
    assert n_builds == 1
    assert n_fits == 1


# ---------------------------------------------------------------------------
# feature_workers wiring. The parallel feature build fans out over asof dates
# INSIDE build_training_set, so it must not disturb the invariant above: the
# ensemble still builds the training set exactly ONCE, whatever the worker
# count. If parallelism ever got wired around the build instead of inside it,
# the 10-seed retrain would build the set ten times in parallel — faster per
# build and ~10x the total work, which no other test would notice.
# ---------------------------------------------------------------------------

def test_train_builds_the_training_set_once_with_parallel_feature_workers(tmp_path):
    result, n_builds, n_fits, _ = _run_full_train(
        tmp_path, ["--ensemble-seeds", "10", "--feature-workers", "4"],
    )
    assert result.exit_code == 0, result.output
    assert n_builds == 1, (
        f"training set built {n_builds}x for a 10-seed ensemble at 4 feature "
        "workers; the fan-out must be INSIDE the build, not around it"
    )
    assert n_fits == 10


def test_train_passes_feature_workers_flag_to_the_build(tmp_path):
    result, _builds, _fits, _ = _run_full_train(tmp_path, ["--feature-workers", "3"])
    assert result.exit_code == 0, result.output
    assert _LAST_BUILD_KWARGS["feature_workers"] == 3


def test_train_reads_feature_workers_from_config_when_flag_omitted(tmp_path):
    """No flag: the scheduled 04:00 plist passes none, so the value must come
    from config.yaml's model.feature_workers."""
    from unittest.mock import patch

    from sma.config import ModelConfig

    with patch(
        "sma.model.__main__.load_model_config",
        return_value=ModelConfig(feature_workers=2),
    ):
        result, _builds, _fits, _ = _run_full_train(tmp_path)
    assert result.exit_code == 0, result.output
    assert _LAST_BUILD_KWARGS["feature_workers"] == 2


def test_train_feature_workers_one_is_the_serial_path(tmp_path):
    result, _builds, _fits, _ = _run_full_train(tmp_path, ["--feature-workers", "1"])
    assert result.exit_code == 0, result.output
    assert _LAST_BUILD_KWARGS["feature_workers"] == 1


def test_train_rejects_a_nonsense_feature_worker_count(tmp_path):
    result, _builds, _fits, _ = _run_full_train(tmp_path, ["--feature-workers", "0"])
    assert result.exit_code != 0, "0 workers must fail loudly, not silently mean 1"


def test_train_saves_an_ensemble_artifact(tmp_path):
    """The CLI's --ensemble-seeds must reach the artifact, not just the fit."""
    import json

    from sma.model.ensemble import EnsembleModel
    from sma.model.persistence import load_model

    result, _builds, _fits, models_dir = _run_full_train(
        tmp_path, ["--ensemble-seeds", "3"],
    )
    assert result.exit_code == 0, result.output

    pkls = list(models_dir.glob("xgb_ret_30d_forward_*.pkl"))
    assert len(pkls) == 1, pkls
    model = load_model(pkls[0])
    assert isinstance(model, EnsembleModel)
    assert len(model) == 3

    meta = json.loads(pkls[0].with_suffix(".json").read_text())
    assert meta["ensemble_seeds"] == 3
    assert meta["ensemble_random_states"] == [42, 43, 44]


def test_train_without_the_flag_uses_the_config_default(tmp_path):
    """The scheduled plist passes no --ensemble-seeds, so config.yaml decides.
    Guards the wiring that makes the shipped default actually take effect."""
    from sma.model.persistence import load_model

    result, n_builds, _fits, models_dir = _run_full_train(tmp_path)
    assert result.exit_code == 0, result.output
    assert n_builds == 1

    from sma.config import load_model_config
    expected = load_model_config("config.yaml").ensemble_seeds

    pkls = list(models_dir.glob("xgb_ret_30d_forward_*.pkl"))
    model = load_model(pkls[0])
    from sma.model.ensemble import ensemble_size
    assert ensemble_size(model) == expected
