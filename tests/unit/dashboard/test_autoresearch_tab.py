"""Tests for the Autoresearch tab's live weekly search-record section
(dashboard/tabs/autoresearch.py), added 2026-09-02.

The tab used to query ONLY the retired autoresearch_experiments DuckDB
table (the LLM tilt()-proposal loop, dead since the 2026-06-19 migration to
the deterministic CV-IC config search -- commits 391c146/5a68736), making it
a permanently-stale gauge. This rewrite makes the live weekly search record
(parsed from data/sentinels/com.sma.autoresearch.nightly-*.json via
sma.autoresearch.history) the primary section and demotes the old table to a
collapsed archive. `_promotion_mask` is exercised by its own regression test
in test_dashboard_reliability_regressions.py and is unchanged here.
"""

from __future__ import annotations

import json

import pytest

from dashboard.tabs import autoresearch
from sma.ingest.store import Store


@pytest.fixture(autouse=True)
def _clear_search_runs_cache():
    autoresearch._search_runs.clear()
    yield
    autoresearch._search_runs.clear()


def _write_sentinel(
    d, date_str, *, promoted=False, dry_run=False, best=0.0153, inc=0.0157,
    n_configs=6, reason="held", completed_at=None, configs=None,
):
    if configs is None:
        configs = [{"rank": 0, "cv_ic": best, "params": {"max_depth": 4}}]
    payload = {
        "kind": "config_search", "asof": date_str, "promoted": promoted,
        "dry_run": dry_run, "best_cv_ic": best, "incumbent_cv_ic": inc,
        "n_configs": n_configs, "configs": configs, "reason": reason,
    }
    if completed_at is not None:
        payload["completed_at"] = completed_at
    (d / f"com.sma.autoresearch.nightly-{date_str}.json").write_text(json.dumps(payload))


# ── _search_history_df: table built from load_search_runs' fixture output ──


def test_history_df_columns_and_newest_first_order_preserved():
    runs = [
        {"asof": "2026-08-31", "best_cv_ic": -0.0024, "incumbent_cv_ic": -0.0007,
         "n_configs": 6, "promoted": False, "reason": "does not beat incumbent"},
        {"asof": "2026-08-24", "best_cv_ic": 0.01, "incumbent_cv_ic": 0.002,
         "n_configs": 6, "promoted": True, "reason": "cleared margin"},
    ]
    df = autoresearch._search_history_df(runs)
    assert list(df.columns) == autoresearch._HISTORY_COLUMNS
    # load_search_runs already returns newest-first; the table must not re-sort.
    assert df["Asof"].tolist() == ["2026-08-31", "2026-08-24"]
    assert df["Decision"].tolist() == ["HOLD", "PROMOTE"]
    assert df.iloc[0]["Best CV-IC"] == "-0.0024"
    assert df.iloc[1]["Reason"] == "cleared margin"


def test_history_df_dry_run_labeled_separately_from_hold():
    runs = [{"asof": "2026-08-01", "dry_run": True, "promoted": False,
             "best_cv_ic": 0.01, "incumbent_cv_ic": 0.002}]
    df = autoresearch._search_history_df(runs)
    assert df.iloc[0]["Decision"] == "DRY-RUN"


def test_history_df_null_safe_on_degraded_run_dict():
    # A minimal/degraded sentinel -- must render every column, never raise.
    df = autoresearch._search_history_df([{"asof": "2026-08-01"}])
    row = df.iloc[0]
    assert row["Best CV-IC"] == "n/a"
    assert row["Incumbent CV-IC"] == "n/a"
    assert row["Reason"] == "n/a"
    assert row["Duration (fire→complete)"] == "n/a"
    assert row["Decision"] == "HOLD"  # promoted/dry_run both absent -> falsy


def test_history_df_empty_list_returns_empty_df_with_columns():
    df = autoresearch._search_history_df([])
    assert df.empty
    assert list(df.columns) == autoresearch._HISTORY_COLUMNS


# ── _search_duration: fire-time-to-completion, computed off real sentinels ─


def test_duration_minutes_only():
    run = {"asof": "2026-08-31", "completed_at": "2026-08-31T11:16:45.683561Z"}
    assert autoresearch._search_duration(run) == "16m"


def test_duration_hours_and_minutes():
    run = {"asof": "2026-08-03", "completed_at": "2026-08-03T17:57:36.144149Z"}
    assert autoresearch._search_duration(run) == "6h57m"


def test_duration_missing_completed_at_is_na():
    assert autoresearch._search_duration({"asof": "2026-08-31"}) == "n/a"


def test_duration_missing_asof_is_na():
    assert autoresearch._search_duration({"completed_at": "2026-08-31T11:16:45Z"}) == "n/a"


def test_duration_garbage_timestamp_is_na_not_a_crash():
    run = {"asof": "2026-08-31", "completed_at": "not-a-timestamp"}
    assert autoresearch._search_duration(run) == "n/a"


def test_duration_garbage_asof_is_na_not_a_crash():
    run = {"asof": "not-a-date", "completed_at": "2026-08-31T11:16:45Z"}
    assert autoresearch._search_duration(run) == "n/a"


# ── empty / missing sentinel dir: _search_runs must never raise ────────────


def test_search_runs_on_missing_sentinel_dir_returns_empty(monkeypatch, tmp_path):
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path / "does-not-exist"))
    assert autoresearch._search_runs(limit=20) == []


def test_search_runs_on_empty_existing_sentinel_dir_returns_empty(monkeypatch, tmp_path):
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path))
    assert autoresearch._search_runs(limit=20) == []


def test_search_runs_reads_fixture_sentinels_newest_first(monkeypatch, tmp_path):
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path))
    _write_sentinel(tmp_path, "2026-08-17")
    _write_sentinel(tmp_path, "2026-08-24")
    runs = autoresearch._search_runs(limit=20)
    assert [r["asof"] for r in runs] == ["2026-08-24", "2026-08-17"]


# ── render() end-to-end: friendly note, never a crash ──────────────────────


def test_render_does_not_crash_with_missing_sentinel_dir_and_empty_db(
    monkeypatch, tmp_path,
):
    db_path = tmp_path / "t.duckdb"
    Store(path=str(db_path)).connect().close()
    monkeypatch.setattr(autoresearch, "DB_PATH", db_path)
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path / "no-such-sentinel-dir"))
    autoresearch._run_summary.clear()
    autoresearch._top_experiments.clear()
    autoresearch._recent_experiments.clear()

    autoresearch.render()  # must not raise


def test_render_does_not_crash_with_populated_sentinels_and_empty_db(
    monkeypatch, tmp_path,
):
    db_path = tmp_path / "t.duckdb"
    Store(path=str(db_path)).connect().close()
    monkeypatch.setattr(autoresearch, "DB_PATH", db_path)
    monkeypatch.setenv("SMA_SENTINEL_DIR", str(tmp_path))
    _write_sentinel(tmp_path, "2026-08-24", promoted=True,
                     completed_at="2026-08-24T11:56:23Z")
    _write_sentinel(tmp_path, "2026-08-31", promoted=False,
                     completed_at="2026-08-31T11:16:45Z")
    autoresearch._run_summary.clear()
    autoresearch._top_experiments.clear()
    autoresearch._recent_experiments.clear()

    autoresearch.render()  # must not raise


# ── archive header: corrected, retirement facts stated ─────────────────────


def test_archive_header_states_retirement_facts():
    assert "Archive" in autoresearch.ARCHIVE_HEADER
    assert "0/52 promoted" in autoresearch.ARCHIVE_HEADER
    assert "retired 6/19" in autoresearch.ARCHIVE_HEADER
