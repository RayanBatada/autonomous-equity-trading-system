"""Tests for the autoresearch search-history review (counterfactual record)."""
import json

from sma.autoresearch.history import format_history, load_search_runs


def _sentinel(d, date_str, *, kind="config_search", promoted=False, best=0.0153,
              inc=0.0157, dry_run=False, configs=None):
    if configs is None:
        configs = [{"rank": 0, "cv_ic": best,
                    "params": {"max_depth": 4, "learning_rate": 0.05}}]
    (d / f"com.sma.autoresearch.nightly-{date_str}.json").write_text(json.dumps({
        "kind": kind, "asof": date_str, "promoted": promoted, "dry_run": dry_run,
        "best_cv_ic": best, "incumbent_cv_ic": inc, "n_configs": len(configs),
        "configs": configs,
    }))


def test_load_search_runs_newest_first_and_limit(tmp_path):
    for ds in ("2026-06-22", "2026-06-29", "2026-07-06"):
        _sentinel(tmp_path, ds)
    runs = load_search_runs(tmp_path, limit=2)
    assert [r["asof"] for r in runs] == ["2026-07-06", "2026-06-29"]


def test_load_skips_non_config_search_and_unreadable(tmp_path):
    _sentinel(tmp_path, "2026-06-22", kind="something_else")
    (tmp_path / "com.sma.autoresearch.nightly-2026-06-23.json").write_text("{bad json")
    assert load_search_runs(tmp_path) == []


def test_format_shows_hold_promote_configs_and_summary(tmp_path):
    _sentinel(tmp_path, "2026-06-22", promoted=False, best=0.0153, inc=0.0157)  # near-miss hold
    _sentinel(tmp_path, "2026-06-29", promoted=True, best=0.0220, inc=0.0157)   # promoted
    out = format_history(load_search_runs(tmp_path))
    assert "PROMOTED" in out and "HELD" in out
    assert "1 promoted, 1 held" in out
    assert "Closest hold" in out      # near-miss surfaced for improvement
    assert "CV-IC" in out             # per-config counterfactual detail shown


def test_format_empty_is_friendly(tmp_path):
    assert "no autoresearch" in format_history([]).lower()
