"""Tests for `python -m sma.autoresearch diff/promote` CLI subcommands.

The autoresearch loop NEVER auto-modifies the live `active.py`; the agent's
job is to propose, the human's job is to merge winners. These CLIs are the
bridge: `diff` shows a candidate, `promote` writes it (after sanity gates).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from sma.autoresearch import experiment_log
from sma.autoresearch.__main__ import (
    ACTIVE_PY_PATH,
    cli,
)
from sma.ingest.store import Store


def _make_store(tmp_path: Path) -> tuple[Store, Path]:
    db_path = tmp_path / "test.duckdb"
    store = Store(path=str(db_path)).connect()
    return store, db_path


def _seed_experiment(
    store: Store,
    *,
    proposed_text: str = "def tilt(*, asof_date, decisions, ctx):\n    return decisions[:5]\n",
    status: str = "ok",
    mono: int = 4,
    overall: float = 1.2,
    baseline: float = 1.0,
) -> str:
    rid = store.allocate_run_id()
    return experiment_log.insert_experiment(
        store=store,
        run_id=rid,
        iter_index=0,
        proposal_sha="abc12345",
        proposal_summary="top-5 concentration",
        active_py_text=proposed_text,
        baseline_overall=baseline,
        sharpes={"w1": 1.1, "w2": 1.2, "w3": 1.3, "w4": 1.1, "w5": 1.4, "overall": overall},
        monotonicity_score=mono,
        status=status,
        error=None,
        agent_cost_usd=0.05,
        duration_seconds=12.3,
    )


def test_diff_prints_unified_diff_for_changed_proposal(tmp_path):
    store, db_path = _make_store(tmp_path)
    exp_id = _seed_experiment(store)
    store.close()

    runner = CliRunner()
    result = runner.invoke(cli, ["diff", exp_id[:8], "--db", str(db_path)])
    assert result.exit_code == 0, result.output
    assert "decisions[:5]" in result.output
    assert "top-5 concentration" in result.output


def test_diff_reports_no_change_when_proposal_matches_live(tmp_path):
    store, db_path = _make_store(tmp_path)
    current = ACTIVE_PY_PATH.read_text()
    exp_id = _seed_experiment(store, proposed_text=current)
    store.close()

    runner = CliRunner()
    result = runner.invoke(cli, ["diff", exp_id[:8], "--db", str(db_path)])
    assert result.exit_code == 0
    assert "identical to live" in result.output


def test_promote_writes_proposed_text_to_active_py(tmp_path, monkeypatch):
    """Sanity gates pass → promote replaces active.py.

    monkeypatch ACTIVE_PY_PATH so the test doesn't actually mutate the
    live module file. (The CLI uses the import-time value, so monkeypatch
    has to target both the CLI module and the loop module that imports it.)
    """
    fake_active = tmp_path / "active.py"
    original_text = "# original content\ndef tilt(): pass\n"
    fake_active.write_text(original_text)

    import sma.autoresearch.__main__ as _cli_mod
    monkeypatch.setattr(_cli_mod, "ACTIVE_PY_PATH", fake_active)

    store, db_path = _make_store(tmp_path)
    proposed = "# proposed content\ndef tilt(): return []\n"
    exp_id = _seed_experiment(store, proposed_text=proposed)
    store.close()

    runner = CliRunner()
    result = runner.invoke(cli, ["promote", exp_id[:8], "--db", str(db_path)])
    assert result.exit_code == 0, result.output
    assert "promoted experiment" in result.output
    assert fake_active.read_text() == proposed


def test_promote_dry_run_does_not_write(tmp_path, monkeypatch):
    fake_active = tmp_path / "active.py"
    original_text = "# original\n"
    fake_active.write_text(original_text)

    import sma.autoresearch.__main__ as _cli_mod
    monkeypatch.setattr(_cli_mod, "ACTIVE_PY_PATH", fake_active)

    store, db_path = _make_store(tmp_path)
    exp_id = _seed_experiment(store, proposed_text="# proposed\n")
    store.close()

    runner = CliRunner()
    result = runner.invoke(
        cli, ["promote", exp_id[:8], "--dry-run", "--db", str(db_path)],
    )
    assert result.exit_code == 0
    assert "DRY RUN" in result.output
    assert fake_active.read_text() == original_text


@pytest.mark.parametrize("kwargs", [
    {"status": "eval_error"},
    {"mono": 1},
    {"overall": 1.05, "baseline": 1.0},  # gap=0.05 < 0.10 threshold
])
def test_promote_rejects_proposals_failing_sanity_gates(tmp_path, monkeypatch, kwargs):
    """Each of status!=ok, mono<3, gap<0.10 must individually block promote."""
    fake_active = tmp_path / "active.py"
    fake_active.write_text("# original\n")

    import sma.autoresearch.__main__ as _cli_mod
    monkeypatch.setattr(_cli_mod, "ACTIVE_PY_PATH", fake_active)

    store, db_path = _make_store(tmp_path)
    exp_id = _seed_experiment(store, **kwargs)
    store.close()

    runner = CliRunner()
    result = runner.invoke(cli, ["promote", exp_id[:8], "--db", str(db_path)])
    assert result.exit_code != 0
    assert fake_active.read_text() == "# original\n"


def test_promote_force_overrides_sanity_gates(tmp_path, monkeypatch):
    fake_active = tmp_path / "active.py"
    fake_active.write_text("# original\n")

    import sma.autoresearch.__main__ as _cli_mod
    monkeypatch.setattr(_cli_mod, "ACTIVE_PY_PATH", fake_active)

    store, db_path = _make_store(tmp_path)
    exp_id = _seed_experiment(store, status="eval_error", mono=0, overall=0.5, baseline=1.0)
    store.close()

    runner = CliRunner()
    result = runner.invoke(
        cli, ["promote", exp_id[:8], "--force", "--db", str(db_path)],
    )
    assert result.exit_code == 0, result.output
    assert "promoted" in result.output


def test_promote_errors_on_unknown_exp_id(tmp_path):
    store, db_path = _make_store(tmp_path)
    store.close()
    runner = CliRunner()
    result = runner.invoke(cli, ["promote", "deadbeef", "--db", str(db_path)])
    assert result.exit_code != 0
    assert "no experiment" in result.output.lower()


def test_promote_errors_on_ambiguous_prefix(tmp_path):
    store, db_path = _make_store(tmp_path)
    # Two experiments with same first 4 chars (extremely unlikely with uuid4,
    # but emulate by inserting two known IDs that share a prefix).
    rid = store.allocate_run_id()
    common_prefix = "aaaaaaaa"
    for tail in ("0000", "1111"):
        store.conn.execute(
            """
            INSERT INTO autoresearch_experiments (
                experiment_id, run_id, iter_index, proposal_sha, proposal_summary,
                active_py_text, baseline_overall,
                sharpe_w1, sharpe_w2, sharpe_w3, sharpe_w4, sharpe_w5, sharpe_overall,
                monotonicity_score, status, error,
                agent_cost_usd, duration_seconds, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                f"{common_prefix}-0000-0000-0000-00000000{tail}",
                rid, 0, "sha", "sum", "txt", 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 5, "ok", None, 0.01, 1.0,
                __import__("datetime").datetime.utcnow(),
            ],
        )
    store.close()
    runner = CliRunner()
    result = runner.invoke(cli, ["promote", common_prefix, "--db", str(db_path)])
    assert result.exit_code != 0
    assert "ambiguous" in result.output.lower()
