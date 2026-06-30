"""Read/write helpers for the `autoresearch_experiments` DuckDB table."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime


def insert_experiment(
    *, store, run_id: int, iter_index: int, proposal_sha: str,
    proposal_summary: str, active_py_text: str,
    baseline_overall: float | None,
    sharpes: dict[str, float | None],   # keys: w1..w5, overall
    monotonicity_score: int | None,
    status: str,                         # 'ok' | 'agent_error' | 'eval_error' | 'parse_error'
    error: str | None,
    agent_cost_usd: float,
    duration_seconds: float,
) -> str:
    """Insert one experiment row. Returns the generated experiment_id."""
    exp_id = str(uuid.uuid4())
    # NOTE: per-window Sharpes use fixed columns sharpe_w1..sharpe_w5, so this
    # schema supports at most 5 CV windows. loop._build_cv_windows() guards
    # n_windows<=5; raising that cap requires migrating this table (and the
    # autoresearch dashboard tab) to add sharpe_w6+ or a JSON column.
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
        [exp_id, run_id, iter_index, proposal_sha, proposal_summary,
         active_py_text, baseline_overall,
         sharpes.get("w1"), sharpes.get("w2"), sharpes.get("w3"),
         sharpes.get("w4"), sharpes.get("w5"), sharpes.get("overall"),
         monotonicity_score, status, error,
         agent_cost_usd, duration_seconds, datetime.now(UTC).replace(tzinfo=None)],
    )
    return exp_id


def list_recent(
    *, store, limit: int = 20,
) -> list[dict]:
    """Return last N experiments as dicts."""
    rows = store.conn.execute(
        """
        SELECT experiment_id, run_id, iter_index, proposal_summary,
               sharpe_overall, monotonicity_score, status, error
        FROM autoresearch_experiments
        ORDER BY created_at DESC LIMIT ?
        """,
        [int(limit)],
    ).fetchall()
    cols = ["experiment_id", "run_id", "iter_index", "proposal_summary",
            "sharpe_overall", "monotonicity_score", "status", "error"]
    return [dict(zip(cols, r, strict=True)) for r in rows]


def list_top(
    *, store, k: int = 10,
) -> list[dict]:
    """Return top-k by (monotonicity_score, sharpe_overall) for promotion review."""
    rows = store.conn.execute(
        """
        SELECT experiment_id, run_id, iter_index, proposal_summary,
               proposal_sha, sharpe_overall, monotonicity_score,
               sharpe_w1, sharpe_w2, sharpe_w3, sharpe_w4, sharpe_w5, status
        FROM autoresearch_experiments
        WHERE status = 'ok'
        ORDER BY monotonicity_score DESC NULLS LAST, sharpe_overall DESC NULLS LAST
        LIMIT ?
        """,
        [int(k)],
    ).fetchall()
    cols = ["experiment_id", "run_id", "iter_index", "proposal_summary",
            "proposal_sha", "sharpe_overall", "monotonicity_score",
            "sharpe_w1", "sharpe_w2", "sharpe_w3", "sharpe_w4", "sharpe_w5", "status"]
    return [dict(zip(cols, r, strict=True)) for r in rows]
