"""Tests for incumbent_cv_ic: the deployed model's out-of-sample rank-IC, used
by the autoresearch search gate to require a real IC lift before promoting."""
import json
from datetime import date

from sma.model.persistence import incumbent_cv_ic

TARGET = "ret_30d_forward"


def _write_model(models_dir, train_end: str, cv_ic, sha="abcd1234"):
    """Write the minimal artifact pair latest_model_for_date + load_metadata read
    (the .pkl is never unpickled by these lookups, so its bytes are irrelevant)."""
    base = f"xgb_{TARGET}_{train_end}_{sha}"
    (models_dir / f"{base}.pkl").write_bytes(b"x")
    (models_dir / f"{base}.json").write_text(
        json.dumps({"cv_ic": cv_ic, "train_end_date": train_end})
    )


def test_returns_cv_ic_of_latest_eligible_model(tmp_path):
    _write_model(tmp_path, "2026-06-01", 0.021)
    _write_model(tmp_path, "2026-06-08", 0.037)  # newer, eligible
    assert incumbent_cv_ic(tmp_path, date(2026, 6, 10), TARGET) == 0.037


def test_returns_none_when_cv_ic_missing(tmp_path):
    _write_model(tmp_path, "2026-06-08", None)
    assert incumbent_cv_ic(tmp_path, date(2026, 6, 10), TARGET) is None


def test_returns_none_when_cv_ic_nan(tmp_path):
    _write_model(tmp_path, "2026-06-08", float("nan"))
    assert incumbent_cv_ic(tmp_path, date(2026, 6, 10), TARGET) is None


def test_returns_none_when_no_model(tmp_path):
    assert incumbent_cv_ic(tmp_path, date(2026, 6, 10), TARGET) is None


def test_ignores_models_newer_than_asof(tmp_path):
    _write_model(tmp_path, "2026-06-20", 0.05)  # train_end after asof
    assert incumbent_cv_ic(tmp_path, date(2026, 6, 10), TARGET) is None
