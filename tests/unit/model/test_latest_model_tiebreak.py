"""latest_model_for_date must break same-date ties by created_at (latest wins).

On Mondays the 04:00 retrain and the 07:00 autoresearch both produce a model
with the same train_end_date; if both promote, the LATER-created (autoresearch)
model must win, or its IC-gated deploy silently doesn't serve.
"""
import json
import os
import time
from datetime import date
from pathlib import Path

from sma.model.persistence import latest_model_for_date

TARGET = "ret_30d_forward"


def _write(models_dir: Path, train_end: str, sha: str, created_at=None, mtime=None):
    base = f"xgb_{TARGET}_{train_end}_{sha}"
    pkl = models_dir / f"{base}.pkl"
    pkl.write_bytes(b"x")
    meta = {"train_end_date": train_end}
    if created_at is not None:
        meta["created_at"] = created_at
    (models_dir / f"{base}.json").write_text(json.dumps(meta))
    if mtime is not None:
        os.utime(pkl, (mtime, mtime))
    return pkl


def test_same_date_tie_broken_by_created_at_latest_wins(tmp_path):
    # create the earlier (retrain) model FIRST, the later (autoresearch) SECOND
    _write(tmp_path, "2026-06-22", "aaaaaaaa", created_at="2026-06-22T04:00:00.000000Z")
    later = _write(tmp_path, "2026-06-22", "bbbbbbbb", created_at="2026-06-22T07:00:00.000000Z")
    assert latest_model_for_date(tmp_path, date(2026, 6, 22), TARGET) == later


def test_single_model_returned_unchanged(tmp_path):
    only = _write(tmp_path, "2026-06-16", "cccccccc", created_at="2026-06-16T04:00:00Z")
    assert latest_model_for_date(tmp_path, date(2026, 6, 20), TARGET) == only


def test_latest_date_still_wins_across_dates(tmp_path):
    _write(tmp_path, "2026-06-15", "dddddddd", created_at="2026-06-15T04:00:00Z")
    newer = _write(tmp_path, "2026-06-22", "eeeeeeee", created_at="2026-06-22T04:00:00Z")
    assert latest_model_for_date(tmp_path, date(2026, 6, 22), TARGET) == newer


def test_tie_falls_back_to_mtime_when_created_at_missing(tmp_path):
    _write(tmp_path, "2026-06-22", "ffffffff", created_at=None, mtime=time.time() - 100)
    new = _write(tmp_path, "2026-06-22", "99999999", created_at=None, mtime=time.time())
    assert latest_model_for_date(tmp_path, date(2026, 6, 22), TARGET) == new


def test_naive_created_at_does_not_crash_and_is_treated_as_utc(tmp_path):
    # Codex HIGH #1: a created_at WITHOUT a tz (legacy/hand-edited) parses naive;
    # comparing it against the aware mtime/aware candidates must not raise.
    _write(tmp_path, "2026-06-22", "aaaaaaaa", created_at="2026-06-22T04:00:00")  # naive
    later = _write(tmp_path, "2026-06-22", "bbbbbbbb", created_at="2026-06-22T07:00:00Z")
    assert latest_model_for_date(tmp_path, date(2026, 6, 22), TARGET) == later


def test_corrupt_sidecar_does_not_crash(tmp_path):
    # Codex HIGH #2: a malformed .json must fall back to mtime, not crash selection
    a = _write(tmp_path, "2026-06-22", "ffffffff", created_at="2026-06-22T07:00:00Z")
    b = _write(tmp_path, "2026-06-22", "99999999", created_at=None, mtime=time.time())
    (tmp_path / "xgb_ret_30d_forward_2026-06-22_99999999.json").write_text("{not valid json")
    result = latest_model_for_date(tmp_path, date(2026, 6, 22), TARGET)
    assert result in (a, b)  # the point is it returned without raising


def test_identical_created_at_is_deterministic(tmp_path):
    # Codex LOW #3: identical created_at must resolve deterministically (by name)
    ca = "2026-06-22T07:00:00.000000Z"
    _write(tmp_path, "2026-06-22", "aaaaaaaa", created_at=ca)
    b = _write(tmp_path, "2026-06-22", "bbbbbbbb", created_at=ca)
    r1 = latest_model_for_date(tmp_path, date(2026, 6, 22), TARGET)
    r2 = latest_model_for_date(tmp_path, date(2026, 6, 22), TARGET)
    assert r1 == r2 == b  # deterministic, and the higher name wins the tiebreak
