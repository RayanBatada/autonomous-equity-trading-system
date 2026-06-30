"""incumbent_hyperparams: the deployed model's hyperparameters, so the
autoresearch search can re-measure the incumbent's OWN config on current data
(fair comparison) and the retrain can keep a good config across weeks."""
import json
from datetime import date

from sma.model.persistence import incumbent_hyperparams

TARGET = "ret_30d_forward"


def _write(models_dir, train_end, sha, **meta):
    base = f"xgb_{TARGET}_{train_end}_{sha}"
    (models_dir / f"{base}.pkl").write_bytes(b"x")
    (models_dir / f"{base}.json").write_text(
        json.dumps({"train_end_date": train_end, **meta})
    )


def test_returns_deployed_models_hyperparams(tmp_path):
    _write(tmp_path, "2026-06-22", "aaaaaaaa",
           hyperparams={"max_depth": 5, "learning_rate": 0.05})
    assert incumbent_hyperparams(tmp_path, date(2026, 6, 23), TARGET) == {
        "max_depth": 5, "learning_rate": 0.05,
    }


def test_none_when_hyperparams_missing_or_empty(tmp_path):
    _write(tmp_path, "2026-06-22", "aaaaaaaa")  # no hyperparams key
    assert incumbent_hyperparams(tmp_path, date(2026, 6, 23), TARGET) is None
    _write(tmp_path, "2026-06-21", "bbbbbbbb", hyperparams={})  # empty
    # the newest (6/22, no hyperparams) still governs -> None
    assert incumbent_hyperparams(tmp_path, date(2026, 6, 23), TARGET) is None


def test_none_when_no_model(tmp_path):
    assert incumbent_hyperparams(tmp_path, date(2026, 6, 23), TARGET) is None
