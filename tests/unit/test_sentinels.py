import json
from datetime import date

from sma.sentinels import ingest_succeeded_today, read_sentinel, sentinel_path, write_sentinel


def test_write_then_read_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path))
    payload = {
        "label": "com.sma.ingest.daily",
        "asof": "2026-04-30",
        "run_id": 12345,
        "quality": {"passed": True},
    }
    write_sentinel(label="com.sma.ingest.daily", asof=date(2026, 4, 30), payload=payload)
    got = read_sentinel(label="com.sma.ingest.daily", asof=date(2026, 4, 30))
    assert got == payload


def test_read_returns_none_when_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path))
    assert read_sentinel(label="com.sma.ingest.daily", asof=date(2026, 4, 30)) is None


def test_atomic_overwrite_does_not_corrupt(tmp_path, monkeypatch):
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path))
    p = sentinel_path(label="x", asof=date(2026, 4, 30))
    write_sentinel(label="x", asof=date(2026, 4, 30), payload={"v": 1})
    write_sentinel(label="x", asof=date(2026, 4, 30), payload={"v": 2})
    assert json.loads(p.read_text())["v"] == 2


def test_path_format(tmp_path, monkeypatch):
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path))
    p = sentinel_path(label="com.sma.ingest.daily", asof=date(2026, 4, 30))
    assert p.name == "com.sma.ingest.daily-2026-04-30.json"


def test_ingest_succeeded_today_true_when_sentinel_exists_with_passed_quality(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path))
    write_sentinel(
        label="com.sma.ingest.daily",
        asof=date(2026, 4, 30),
        payload={
            "label": "com.sma.ingest.daily",
            "asof": "2026-04-30",
            "quality": {"passed": True, "blocking_failures": []},
        },
    )
    assert ingest_succeeded_today(asof=date(2026, 4, 30)) is True


def test_ingest_succeeded_today_false_when_no_sentinel(tmp_path, monkeypatch):
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path))
    assert ingest_succeeded_today(asof=date(2026, 4, 30)) is False


def test_ingest_succeeded_today_false_when_quality_failed(tmp_path, monkeypatch):
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path))
    write_sentinel(
        label="com.sma.ingest.daily",
        asof=date(2026, 4, 30),
        payload={
            "label": "com.sma.ingest.daily",
            "asof": "2026-04-30",
            "quality": {"passed": False, "blocking_failures": ["all_tickers_have_price"]},
        },
    )
    assert ingest_succeeded_today(asof=date(2026, 4, 30)) is False


def test_atomic_write_does_not_leak_tempfile(tmp_path, monkeypatch):
    """tempfile + os.replace should leave only the final file, no .tmp leftovers."""
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path))
    write_sentinel(label="x", asof=date(2026, 4, 30), payload={"v": 1})
    files = list(tmp_path.iterdir())
    assert len(files) == 1
    assert files[0].name == "x-2026-04-30.json"


def test_monotonicity_newer_run_id_replaces_older(tmp_path, monkeypatch):
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path))
    asof = date(2026, 4, 30)
    p1 = write_sentinel(label="x", asof=asof, payload={"run_id": 100, "v": "first"})
    p2 = write_sentinel(label="x", asof=asof, payload={"run_id": 200, "v": "second"})
    assert p1 is not None and p2 is not None
    got = read_sentinel(label="x", asof=asof)
    assert got["run_id"] == 200
    assert got["v"] == "second"


def test_monotonicity_older_run_id_is_suppressed(tmp_path, monkeypatch):
    """A late-arriving older-run write must NOT overwrite a newer sentinel."""
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path))
    asof = date(2026, 4, 30)
    write_sentinel(label="x", asof=asof, payload={"run_id": 200, "v": "newer"})
    result = write_sentinel(label="x", asof=asof, payload={"run_id": 100, "v": "older"})
    assert result is None  # write was suppressed
    got = read_sentinel(label="x", asof=asof)
    assert got["run_id"] == 200
    assert got["v"] == "newer"


def test_monotonicity_falls_back_to_completed_at_when_no_run_id(tmp_path, monkeypatch):
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path))
    asof = date(2026, 4, 30)
    write_sentinel(
        label="x", asof=asof, payload={"completed_at": "2026-04-30T20:05:00Z", "v": "newer"}
    )
    result = write_sentinel(
        label="x",
        asof=asof,
        payload={"completed_at": "2026-04-30T20:00:00Z", "v": "older"},
    )
    assert result is None
    got = read_sentinel(label="x", asof=asof)
    assert got["v"] == "newer"


def test_monotonicity_equal_run_id_is_suppressed(tmp_path, monkeypatch):
    """Strictly-newer means strictly greater; equal run_id should be suppressed."""
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path))
    asof = date(2026, 4, 30)
    write_sentinel(label="x", asof=asof, payload={"run_id": 100, "v": "first"})
    result = write_sentinel(label="x", asof=asof, payload={"run_id": 100, "v": "second"})
    assert result is None
    got = read_sentinel(label="x", asof=asof)
    assert got["v"] == "first"


def test_concurrent_writes_for_same_label_no_corruption(tmp_path, monkeypatch):
    """Sequential writes of the same (label, asof) with monotonic run_ids must
    leave the sentinel reflecting the highest run_id, regardless of order."""
    import random

    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path))
    asof = date(2026, 4, 30)
    run_ids = list(range(1, 21))
    random.shuffle(run_ids)
    for rid in run_ids:
        write_sentinel(label="x", asof=asof, payload={"run_id": rid, "v": f"run_{rid}"})
    got = read_sentinel(label="x", asof=asof)
    assert got["run_id"] == 20
    assert got["v"] == "run_20"


def test_runless_sentinel_does_not_clobber_real_verdict(tmp_path, monkeypatch):
    """A run-less sentinel (holiday-skip, run_id=None) must NOT overwrite a real
    ingest verdict (run_id set) even with a later completed_at. Otherwise a
    holiday-skip (quality.passed=True) could replace a real quality.passed=False
    verdict and decide would trade on the wrong upstream state."""
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path))
    asof = date(2026, 5, 25)
    write_sentinel(label="com.sma.ingest.daily", asof=asof, payload={
        "run_id": 42, "completed_at": "2026-05-25T13:00:00Z",
        "quality": {"passed": False},
    })
    # A later run-less holiday-skip must be SUPPRESSED.
    result = write_sentinel(label="com.sma.ingest.daily", asof=asof, payload={
        "run_id": None, "completed_at": "2026-05-25T18:00:00Z",
        "quality": {"passed": True}, "holiday_skipped": True,
    })
    assert result is None, "run-less holiday-skip must not overwrite a real verdict"
    got = read_sentinel(label="com.sma.ingest.daily", asof=asof)
    assert got["run_id"] == 42 and got["quality"]["passed"] is False


# --- SMA_SENTINEL_DIR env isolation (2026-06-09: the integration suite wrote
# --- REAL sentinels into data/sentinels/ because SENTINEL_DIR is a CWD-relative
# --- constant; env-first resolution lets conftest isolate every test) ---


def test_env_var_overrides_sentinel_dir(tmp_path, monkeypatch):
    env_dir = tmp_path / "env_sentinels"
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(env_dir))
    p = write_sentinel(label="x", asof=date(2026, 1, 5), payload={"run_id": 1})
    assert p is not None and p.parent == env_dir
    assert read_sentinel(label="x", asof=date(2026, 1, 5)) == {"run_id": 1}
    assert sentinel_path(label="x", asof=date(2026, 1, 5)).parent == env_dir


def test_module_default_used_when_env_unset(tmp_path, monkeypatch):
    monkeypatch.delenv("SMA_SENTINEL_DIR", raising=False)
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path / "legacy"))
    p = write_sentinel(label="x", asof=date(2026, 1, 5), payload={"run_id": 1})
    assert p is not None and p.parent == tmp_path / "legacy"


def test_conftest_isolates_sentinel_dir_for_every_test():
    """The autouse conftest fixture must point SMA_SENTINEL_DIR at a temp dir
    so no test (unit OR integration/CLI) can write production sentinels."""
    import os
    from pathlib import Path

    val = os.environ.get("SMA_SENTINEL_DIR")
    assert val, "conftest must set SMA_SENTINEL_DIR for every test"
    assert Path(val).is_absolute()
    assert "data/sentinels" not in val


def test_nan_and_inf_are_written_as_null(tmp_path, monkeypatch):
    """The weekly-retrain sentinel hit cv_rmse=NaN on tiny folds; json.dump's
    default emits bare NaN, which is invalid strict JSON (jq and other readers
    choke). NaN/inf must serialize as null at any nesting depth."""
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path / "s"))
    write_sentinel(
        label="x",
        asof=date(2026, 1, 6),
        payload={
            "run_id": 1,
            "cv_rmse": float("nan"),
            "nested": {"vals": [float("inf"), 1.0, float("-inf")]},
        },
    )
    raw = sentinel_path(label="x", asof=date(2026, 1, 6)).read_text()
    assert "NaN" not in raw and "Infinity" not in raw
    got = read_sentinel(label="x", asof=date(2026, 1, 6))
    assert got["cv_rmse"] is None
    assert got["nested"]["vals"] == [None, 1.0, None]


def test_monotonicity_compares_timestamps_as_datetimes_not_strings(tmp_path, monkeypatch):
    """Codex module review (2026-06-11 LOW): live sentinels (no run_id) compared
    completed_at LEXICOGRAPHICALLY — '2026-06-10T20:00:00Z' vs
    '2026-06-10T20:00:00.123456Z' mixes fractional/non-fractional forms and
    can suppress a genuinely newer write. Compare as parsed datetimes."""
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path / "s"))
    asof = date(2026, 6, 10)
    write_sentinel(
        label="x", asof=asof,
        payload={"completed_at": "2026-06-10T20:00:01Z", "v": "older"},
    )
    # 0.9s LATER, but the fractional form is LEXICOGRAPHICALLY smaller
    # ('.': 46 < 'Z': 90 at the same position) → string compare suppresses it
    result = write_sentinel(
        label="x", asof=asof,
        payload={"completed_at": "2026-06-10T20:00:01.900000Z", "v": "newer"},
    )
    assert result is not None, "chronologically newer write must not be suppressed"
    assert read_sentinel(label="x", asof=asof)["v"] == "newer"
