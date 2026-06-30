"""Tests for the autoresearch loop infrastructure."""

from pathlib import Path

import pytest

from sma.autoresearch import experiment_log
from sma.autoresearch.agent import _parse_proposal, build_prompt
from sma.autoresearch.loop import (
    ACTIVE_PY_PATH,
    _compute_monotonicity,
    _SwappedActivePy,
    _verify_proposal_safe,
)
from sma.ingest.store import Store

# ---- agent --------------------------------------------------------------

def test_parse_proposal_extracts_code_and_summary():
    text = """\
Here's my proposed change:

```python
def tilt(*, asof_date, decisions, ctx):
    return decisions[:5]
```

I keep only the top-5 to concentrate the portfolio."""
    code, summary = _parse_proposal(text)
    assert code.startswith("def tilt")
    assert "top-5" in summary


def test_parse_proposal_raises_without_python_block():
    with pytest.raises(RuntimeError, match="missing"):
        _parse_proposal("just words, no code block")


def test_build_prompt_handles_empty_history():
    prompt = build_prompt(
        current_active_py="def tilt(*, asof_date, decisions, ctx): return decisions",
        recent_experiments=[],
    )
    assert "no prior experiments" in prompt


def test_build_prompt_summarizes_recent_log():
    prompt = build_prompt(
        current_active_py="x",
        recent_experiments=[
            {"iter_index": 0, "monotonicity_score": 3, "sharpe_overall": 0.42,
             "proposal_summary": "concentrate to top-5"},
            {"iter_index": 1, "monotonicity_score": 1, "sharpe_overall": 0.31,
             "proposal_summary": "boost top score"},
        ],
    )
    assert "concentrate to top-5" in prompt
    assert "boost top score" in prompt


# ---- loop static checks -------------------------------------------------

def test_verify_proposal_safe_accepts_well_formed_tilt():
    good = """
def tilt(*, asof_date, decisions, ctx):
    return decisions
"""
    # Should not raise.
    _verify_proposal_safe(good)


def test_verify_proposal_safe_rejects_signature_drift():
    bad = """
def tilt(asof_date, decisions, ctx):
    return decisions
"""
    with pytest.raises(RuntimeError, match="positional"):
        _verify_proposal_safe(bad)


def test_verify_proposal_safe_rejects_missing_tilt():
    bad = "x = 1\n"
    with pytest.raises(RuntimeError, match="does not define tilt"):
        _verify_proposal_safe(bad)


def test_verify_proposal_safe_rejects_syntax_error():
    bad = "def tilt(*, asof_date,\n    decisions,\n    ctx:\n    return\n"
    with pytest.raises(RuntimeError, match="does not parse"):
        _verify_proposal_safe(bad)


# ---- security allowlist (2026-06-05 audit: arbitrary code-exec on import) ----


def test_verify_proposal_safe_accepts_real_active_py():
    """Regression: the actual baseline active.py must pass the allowlist."""
    from sma.autoresearch.loop import ACTIVE_PY_PATH

    _verify_proposal_safe(ACTIVE_PY_PATH.read_text())  # must not raise


def test_verify_proposal_safe_rejects_dangerous_import():
    bad = "import os\ndef tilt(*, asof_date, decisions, ctx):\n    return decisions\n"
    with pytest.raises(RuntimeError, match="disallowed module"):
        _verify_proposal_safe(bad)


def test_verify_proposal_safe_rejects_import_inside_tilt():
    bad = (
        "def tilt(*, asof_date, decisions, ctx):\n"
        "    import subprocess\n"
        "    return decisions\n"
    )
    with pytest.raises(RuntimeError, match="disallowed module"):
        _verify_proposal_safe(bad)


def test_verify_proposal_safe_rejects_dunder_import_call():
    bad = (
        "def tilt(*, asof_date, decisions, ctx):\n"
        "    __import__('os').system('echo hi')\n"
        "    return decisions\n"
    )
    with pytest.raises(RuntimeError, match="banned builtin"):
        _verify_proposal_safe(bad)


def test_verify_proposal_safe_rejects_top_level_statement():
    bad = (
        "print('side effect on import')\n"
        "def tilt(*, asof_date, decisions, ctx):\n"
        "    return decisions\n"
    )
    with pytest.raises(RuntimeError, match="disallowed top-level"):
        _verify_proposal_safe(bad)


def test_verify_proposal_safe_rejects_dunder_attr_escape():
    bad = (
        "def tilt(*, asof_date, decisions, ctx):\n"
        "    decisions.__class__.__bases__[0].__subclasses__()\n"
        "    return decisions\n"
    )
    with pytest.raises(RuntimeError, match="banned attribute"):
        _verify_proposal_safe(bad)


# ---- monotonicity score -------------------------------------------------

def test_compute_monotonicity_counts_windows_above_baseline_plus_threshold():
    sharpes = {"w1": 0.50, "w2": 0.30, "w3": 0.45, "w4": 0.10, "w5": 0.42}
    baseline = [0.40, 0.40, 0.40, 0.40, 0.40]
    # W1 = 0.50 >= 0.40+0.05  ✓
    # W2 = 0.30 <  0.45        ✗
    # W3 = 0.45 == 0.45        ✓
    # W4 = 0.10 <  0.45        ✗
    # W5 = 0.42 <  0.45        ✗
    assert _compute_monotonicity(sharpes, baseline) == 2


def test_compute_monotonicity_skips_none_sharpes():
    sharpes = {"w1": 0.50, "w2": None, "w3": 0.45, "w4": None, "w5": None}
    baseline = [0.40, 0.40, 0.40, 0.40, 0.40]
    assert _compute_monotonicity(sharpes, baseline) == 2


# ---- swap context manager (restores active.py on exit) -----------------

def test_swapped_active_py_restores_original_on_normal_exit():
    original = ACTIVE_PY_PATH.read_text()
    with _SwappedActivePy("# swapped\n"):
        assert ACTIVE_PY_PATH.read_text() == "# swapped\n"
    assert ACTIVE_PY_PATH.read_text() == original


def test_swapped_active_py_restores_original_on_exception():
    original = ACTIVE_PY_PATH.read_text()
    with pytest.raises(RuntimeError, match="boom"), _SwappedActivePy("# swapped\n"):
        assert ACTIVE_PY_PATH.read_text() == "# swapped\n"
        raise RuntimeError("boom")
    assert ACTIVE_PY_PATH.read_text() == original
    # Backup file is cleaned up.
    assert not (ACTIVE_PY_PATH.with_suffix(".py.autoresearch_bak")).exists()


# ---- experiment log -----------------------------------------------------

def test_experiment_log_insert_and_list(tmp_path: Path):
    store = Store(path=str(tmp_path / "ar.duckdb")).connect()
    try:
        rid = store.allocate_run_id()
        eid = experiment_log.insert_experiment(
            store=store, run_id=rid, iter_index=0,
            proposal_sha="abc123", proposal_summary="concentrate to top-5",
            active_py_text="def tilt(*, asof_date, decisions, ctx): return decisions[:5]",
            baseline_overall=0.20,
            sharpes={"w1": 0.30, "w2": 0.25, "w3": 0.35,
                     "w4": 0.28, "w5": 0.31, "overall": 0.298},
            monotonicity_score=4,
            status="ok", error=None,
            agent_cost_usd=0.05, duration_seconds=12.3,
        )
        assert eid

        recent = experiment_log.list_recent(store=store, limit=10)
        assert len(recent) == 1
        assert recent[0]["proposal_summary"] == "concentrate to top-5"
        assert recent[0]["monotonicity_score"] == 4

        top = experiment_log.list_top(store=store, k=5)
        assert len(top) == 1
        assert top[0]["sharpe_w1"] == 0.30
    finally:
        store.conn.close()
