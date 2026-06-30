"""Integration tests for sma.model CLI: train, predict, backfill-predictions."""

from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest
from click.testing import CliRunner


def _make_tiny_features(n: int = 20) -> pd.DataFrame:
    import numpy as np
    rng = np.random.default_rng(0)
    return pd.DataFrame(rng.random((n, 5)), columns=[f"f{i}" for i in range(5)])


def _make_tiny_y(n: int = 20) -> pd.Series:
    import numpy as np
    rng = np.random.default_rng(1)
    return pd.Series(rng.random(n))


def _make_asof_dates(n: int = 20) -> pd.Series:
    return pd.Series([date(2024, 1, i % 28 + 1) for i in range(n)])


# ---------------------------------------------------------------------------
# Test 1: train smoke (--no-cv)
# ---------------------------------------------------------------------------


def test_train_command_smoke(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """train --no-cv: builds set, trains, saves. Exit 0 and 'Saved' in output."""
    from sma.model import __main__ as mod

    X = _make_tiny_features()  # noqa: N806
    y = _make_tiny_y()
    asof_dates = _make_asof_dates()

    monkeypatch.setattr(mod, "build_training_set", lambda **kw: (X, y, asof_dates))
    monkeypatch.setattr(mod, "train_xgb", lambda X, y, hyperparams=None, **kw: MagicMock())
    monkeypatch.setattr(
        mod, "save_model",
        lambda model, **kw: (tmp_path / "out.pkl", tmp_path / "out.json"),
    )
    monkeypatch.setattr(mod, "_load_prices_for_range", lambda *args, **kw: pd.DataFrame())

    from sma.model.__main__ import cli

    runner = CliRunner()
    result = runner.invoke(cli, [
        "train",
        "--asof", "2025-06-30",
        "--no-cv",
        "--db-path", str(tmp_path / "sma.duckdb"),
        "--models-dir", str(tmp_path / "models"),
        "--universe-path", "src/sma/universe.yaml",
    ])
    assert result.exit_code == 0, result.output
    assert "Saved" in result.output


# ---------------------------------------------------------------------------
# Test 2: train with CV (select_hyperparams called)
# ---------------------------------------------------------------------------


def test_train_command_with_cv(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """train without --no-cv: select_hyperparams is called."""
    from sma.model import __main__ as mod

    X = _make_tiny_features()  # noqa: N806
    y = _make_tiny_y()
    asof_dates = _make_asof_dates()

    select_called = {"n": 0}

    def fake_keep(X, y, asof_dates, **kw):  # noqa: N803
        select_called["n"] += 1
        return {"max_depth": 3, "learning_rate": 0.05}, 0.02, 0.012  # params, cv_ic, cv_rmse

    monkeypatch.setattr(mod, "build_training_set", lambda **kw: (X, y, asof_dates))
    monkeypatch.setattr(mod, "select_hyperparams_keep_better_ic", fake_keep)
    monkeypatch.setattr(mod, "train_xgb", lambda X, y, hyperparams=None, **kw: MagicMock())
    monkeypatch.setattr(
        mod, "save_model",
        lambda model, **kw: (tmp_path / "out.pkl", tmp_path / "out.json"),
    )
    monkeypatch.setattr(mod, "_load_prices_for_range", lambda *args, **kw: pd.DataFrame())

    from sma.model.__main__ import cli

    runner = CliRunner()
    result = runner.invoke(cli, [
        "train",
        "--asof", "2025-06-30",
        "--db-path", str(tmp_path / "sma.duckdb"),
        "--models-dir", str(tmp_path / "models"),
        "--universe-path", "src/sma/universe.yaml",
    ])
    assert result.exit_code == 0, result.output
    assert select_called["n"] == 1
    assert "Saved" in result.output


# ---------------------------------------------------------------------------
# Test 3: train aborts on empty training data
# ---------------------------------------------------------------------------


def test_train_command_aborts_on_empty_data(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """train aborts with exit code 1 when build_training_set returns empty X."""
    from sma.model import __main__ as mod

    monkeypatch.setattr(
        mod, "build_training_set",
        lambda **kw: (pd.DataFrame(), pd.Series([], dtype=float), pd.Series([], dtype=object)),
    )
    monkeypatch.setattr(mod, "_load_prices_for_range", lambda **kw: pd.DataFrame())

    from sma.model.__main__ import cli

    runner = CliRunner()
    result = runner.invoke(cli, [
        "train",
        "--asof", "2025-06-30",
        "--no-cv",
        "--db-path", str(tmp_path / "sma.duckdb"),
        "--models-dir", str(tmp_path / "models"),
        "--universe-path", "src/sma/universe.yaml",
    ])
    assert result.exit_code == 1


# ---------------------------------------------------------------------------
# Test 4: predict writes predictions
# ---------------------------------------------------------------------------


def test_predict_command_writes_predictions(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """predict: calls write_predictions and echoes 'Wrote'."""
    from sma.model import __main__ as mod

    monkeypatch.setattr(
        "sma.model.predictor.Predictor.predict_for_with_model_id",
        lambda self, asof_date, universe, **kw: ({"AAA": 0.05}, "test_model_v1"),
    )
    monkeypatch.setattr(mod, "write_predictions", lambda **kw: 1)

    from sma.model.__main__ import cli

    runner = CliRunner()
    result = runner.invoke(cli, [
        "predict",
        "--no-preflight",  # historical date; no ingest sentinel exists
        "--asof", "2025-06-30",
        "--db-path", str(tmp_path / "sma.duckdb"),
        "--models-dir", str(tmp_path / "models"),
        "--universe-path", "src/sma/universe.yaml",
    ])
    assert result.exit_code == 0, result.output
    assert "Wrote" in result.output


# ---------------------------------------------------------------------------
# Test 5: predict aborts when no model available
# ---------------------------------------------------------------------------


def test_predict_command_aborts_when_no_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """predict exits 1 when predict_for_with_model_id raises FileNotFoundError."""
    monkeypatch.setattr(
        "sma.model.predictor.Predictor.predict_for_with_model_id",
        lambda self, asof_date, universe, **kw: (_ for _ in ()).throw(
            FileNotFoundError("no model")
        ),
    )

    from sma.model.__main__ import cli

    runner = CliRunner()
    result = runner.invoke(cli, [
        "predict",
        "--no-preflight",  # historical date; no ingest sentinel exists
        "--asof", "2025-06-30",
        "--db-path", str(tmp_path / "sma.duckdb"),
        "--models-dir", str(tmp_path / "models"),
        "--universe-path", "src/sma/universe.yaml",
    ])
    assert result.exit_code == 1


# ---------------------------------------------------------------------------
# Test 6: backfill-predictions walks Fridays and calls train + predict
# ---------------------------------------------------------------------------


def test_backfill_predictions_walks_fridays(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """backfill-predictions exits 0 and invokes train/predict for the right dates.

    Window: 2025-07-04 (Friday) to 2025-07-18 (Friday).
    Fridays that anchor model windows: 2025-07-04, 2025-07-11, 2025-07-18.
    """
    from sma.model import __main__ as mod

    X = _make_tiny_features()  # noqa: N806
    y = _make_tiny_y()
    asof_dates = _make_asof_dates()

    # Patch the heavy ops so ctx.invoke(train, ...) and ctx.invoke(predict, ...) complete fast.
    monkeypatch.setattr(mod, "build_training_set", lambda **kw: (X, y, asof_dates))
    monkeypatch.setattr(
        mod, "select_hyperparams_keep_better_ic",
        lambda X, y, asof_dates, **kw: ({}, 0.02, 0.01),  # params, cv_ic, cv_rmse
    )
    monkeypatch.setattr(mod, "train_xgb", lambda X, y, hyperparams=None, **kw: MagicMock())
    monkeypatch.setattr(
        mod, "save_model",
        lambda model, **kw: (tmp_path / "out.pkl", tmp_path / "out.json"),
    )
    monkeypatch.setattr(mod, "_load_prices_for_range", lambda *args, **kw: pd.DataFrame())
    # Make latest_model_for_date always raise so backfill always invokes train
    monkeypatch.setattr(
        mod, "latest_model_for_date",
        lambda models_dir, d, target: (_ for _ in ()).throw(FileNotFoundError("no model")),
    )
    monkeypatch.setattr(
        "sma.model.predictor.Predictor.predict_for_with_model_id",
        lambda self, asof_date, universe, **kw: ({"AAA": 0.05}, "mid"),
    )
    monkeypatch.setattr(mod, "write_predictions", lambda **kw: 1)

    from sma.model.__main__ import cli

    runner = CliRunner()
    result = runner.invoke(cli, [
        "backfill-predictions",
        "--start", "2025-07-04",
        "--end", "2025-07-18",
        "--db-path", str(tmp_path / "sma.duckdb"),
        "--models-dir", str(tmp_path / "models"),
        "--universe-path", "src/sma/universe.yaml",
    ])
    assert result.exit_code == 0, result.output
    assert "Backfill done" in result.output
    # 3 Fridays in [2025-07-04, 2025-07-18] so 3 trains
    assert "3 models trained" in result.output


# ---------------------------------------------------------------------------
# Test 7: backfill-predictions aborts when start > end
# ---------------------------------------------------------------------------


def test_backfill_predictions_aborts_on_bad_dates(tmp_path: Path):
    """backfill-predictions exits 1 when --start is after --end."""
    from sma.model.__main__ import cli

    runner = CliRunner()
    result = runner.invoke(cli, [
        "backfill-predictions",
        "--start", "2025-07-20",
        "--end", "2025-07-10",
        "--db-path", str(tmp_path / "sma.duckdb"),
        "--models-dir", str(tmp_path / "models"),
        "--universe-path", "src/sma/universe.yaml",
    ])
    assert result.exit_code == 1


def test_train_quarantines_worse_than_chance_model(monkeypatch, tmp_path):
    """End-to-end: a model with clearly-negative held-out CV-IC is NOT
    promoted (quarantined to models/rejected), even though its RMSE passes —
    the IC floor is the top deploy guard (2026-06-15)."""

    import pandas as pd
    from click.testing import CliRunner

    from sma.model import __main__ as mod
    from sma.model.__main__ import cli

    X = _make_tiny_features()
    y = _make_tiny_y()
    asof_dates = _make_asof_dates()
    monkeypatch.setattr(mod, "build_training_set", lambda **kw: (X, y, asof_dates))
    # force a clearly-worse-than-chance held-out IC (params, cv_ic, cv_rmse)
    monkeypatch.setattr(
        mod, "select_hyperparams_keep_better_ic",
        lambda *a, **k: ({"max_depth": 3}, -0.08, 0.10),
    )
    monkeypatch.setattr(mod, "train_xgb", lambda X, y, hyperparams=None, **kw: MagicMock())
    monkeypatch.setattr(mod, "_load_prices_for_range", lambda *a, **k: pd.DataFrame())

    saved = {}
    def fake_save(model, **kw):
        saved.update(kw)
        return (tmp_path / "m.pkl", tmp_path / "m.json")
    monkeypatch.setattr(mod, "save_model", fake_save)

    runner = CliRunner()
    result = runner.invoke(cli, [
        "train", "--asof", "2025-06-30",
        "--db-path", str(tmp_path / "x.duckdb"),
        "--models-dir", str(tmp_path / "models"),
        "--universe-path", "src/sma/universe.yaml",
    ])
    assert result.exit_code == 0, result.output
    assert saved["promoted"] is False, "worse-than-chance IC must be quarantined"
    assert str(saved["output_dir"]).endswith("rejected")


def test_train_fails_closed_on_unusable_cv_ic(monkeypatch, tmp_path):
    """Codex review 2026-06-15: CV ran but produced NO usable IC (nan) without
    --no-cv → that's missing evidence, must quarantine (not silently bypass)."""
    import pandas as pd
    from click.testing import CliRunner

    from sma.model import __main__ as mod
    from sma.model.__main__ import cli

    X = _make_tiny_features(); y = _make_tiny_y(); asof_dates = _make_asof_dates()
    monkeypatch.setattr(mod, "build_training_set", lambda **kw: (X, y, asof_dates))
    monkeypatch.setattr(
        mod, "select_hyperparams_keep_better_ic",
        lambda *a, **k: ({"max_depth": 3}, float("nan"), 0.10),  # cv_ic=nan -> fail closed
    )
    monkeypatch.setattr(mod, "train_xgb", lambda X, y, hyperparams=None, **kw: MagicMock())
    monkeypatch.setattr(mod, "_load_prices_for_range", lambda *a, **k: pd.DataFrame())
    saved = {}
    def _save(model, **kw):
        saved.update(kw)
        return (tmp_path / "m.pkl", tmp_path / "m.json")
    monkeypatch.setattr(mod, "save_model", _save)
    # NOT --no-cv: CV ran, nan = unusable -> fail closed
    result = CliRunner().invoke(cli, [
        "train", "--asof", "2025-06-30", "--db-path", str(tmp_path/"x.duckdb"),
        "--models-dir", str(tmp_path/"models"), "--universe-path", "src/sma/universe.yaml",
    ])
    assert result.exit_code == 0, result.output
    assert saved["promoted"] is False, "unusable CV-IC must quarantine, not bypass"


def test_train_no_cv_bypasses_ic_gate(monkeypatch, tmp_path):
    """--no-cv (operator choice) has no IC; the gate must NOT block on that."""
    import pandas as pd
    from click.testing import CliRunner

    from sma.model import __main__ as mod
    from sma.model.__main__ import cli

    X = _make_tiny_features(); y = _make_tiny_y(); asof_dates = _make_asof_dates()
    monkeypatch.setattr(mod, "build_training_set", lambda **kw: (X, y, asof_dates))
    monkeypatch.setattr(mod, "train_xgb", lambda X, y, hyperparams=None, **kw: MagicMock())
    monkeypatch.setattr(mod, "_load_prices_for_range", lambda *a, **k: pd.DataFrame())
    saved = {}
    def _save(model, **kw):
        saved.update(kw)
        return (tmp_path / "m.pkl", tmp_path / "m.json")
    monkeypatch.setattr(mod, "save_model", _save)
    result = CliRunner().invoke(cli, [
        "train", "--no-cv", "--asof", "2025-06-30", "--db-path", str(tmp_path/"x.duckdb"),
        "--models-dir", str(tmp_path/"models"), "--universe-path", "src/sma/universe.yaml",
    ])
    assert result.exit_code == 0, result.output
    assert saved["promoted"] is True, "--no-cv must bypass the IC gate (operator choice)"


def test_train_window_change_promotes_on_ic_floor_despite_worse_rmse(monkeypatch, tmp_path):
    """2026-06-16: a training-WINDOW change (multi-regime) raises RMSE even as
    OOS ranking improves; the gate must promote on the IC floor, not block on
    the RMSE ratio. Incumbent: train_start 2023, cv_rmse 0.124. New: train_start
    2018, cv_rmse 0.138 (>1.10x, would fail RMSE) but cv_ic above floor."""
    import json

    import pandas as pd
    from click.testing import CliRunner

    from sma.model import __main__ as mod
    from sma.model.__main__ import cli

    models = tmp_path / "models"; models.mkdir()
    # seed an incumbent: 2023 train_start, good RMSE
    (models / "xgb_ret_30d_forward_2025-06-30_inc.json").write_text(json.dumps({
        "model_id": "xgb_ret_30d_forward_2025-06-30_inc", "target": "ret_30d_forward",
        "cv_rmse": 0.124, "label_type": "demean", "train_start": "2023-01-01",
        "promoted": True, "train_end_date": "2025-06-30",
    }))
    (models / "xgb_ret_30d_forward_2025-06-30_inc.pkl").write_bytes(b"x")

    X = _make_tiny_features(); y = _make_tiny_y(); asof_dates = _make_asof_dates()
    monkeypatch.setattr(mod, "build_training_set", lambda **kw: (X, y, asof_dates))
    monkeypatch.setattr(
        mod, "select_hyperparams_keep_better_ic",
        lambda *a, **k: ({"max_depth": 3}, 0.015, 0.138),  # cv_ic above floor, worse cv_rmse
    )
    monkeypatch.setattr(mod, "train_xgb", lambda X, y, hyperparams=None, **kw: MagicMock())
    monkeypatch.setattr(mod, "_load_prices_for_range", lambda *a, **k: pd.DataFrame())
    monkeypatch.setenv("SMA_TRAIN_START", "2018-01-01")  # window change
    saved = {}
    def _save(model, **kw):
        saved.update(kw); return (tmp_path / "m.pkl", tmp_path / "m.json")
    monkeypatch.setattr(mod, "save_model", _save)

    result = CliRunner().invoke(cli, [
        "train", "--demean-labels", "--asof", "2025-06-30",
        "--db-path", str(tmp_path / "x.duckdb"), "--models-dir", str(models),
        "--universe-path", "src/sma/universe.yaml",
    ])
    assert result.exit_code == 0, result.output
    assert saved["promoted"] is True, "window change must promote on IC floor despite worse RMSE"
