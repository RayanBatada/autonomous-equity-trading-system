"""Regression guard: dashboard DB opens must be lock-tolerant.

A raw `duckdb.connect(..., read_only=True)` crashes with an IOException when a
scheduled writer job (ingest/predict/agents/decide/reconcile) holds DuckDB's
file lock — the roadmap (default) tab crashed this way on 2026-06-05. All
dashboard reads must go through `sma.db_connect.read_only_connect`, which
retries the open with capped backoff.

Uses AST (not text matching) so docstrings/comments that mention the pattern
don't trip it — only real call sites count.
"""

import ast
import pathlib


def _is_raw_readonly_connect(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if not (
        isinstance(func, ast.Attribute)
        and func.attr == "connect"
        and isinstance(func.value, ast.Name)
        and func.value.id == "duckdb"
    ):
        return False
    return any(
        kw.arg == "read_only"
        and isinstance(kw.value, ast.Constant)
        and kw.value.value is True
        for kw in node.keywords
    )


def test_dashboard_has_no_raw_readonly_duckdb_connect():
    dashboard_dir = pathlib.Path(__file__).resolve().parents[3] / "dashboard"
    offenders = []
    for p in dashboard_dir.rglob("*.py"):
        tree = ast.parse(p.read_text())
        if any(_is_raw_readonly_connect(n) for n in ast.walk(tree)):
            offenders.append(str(p.relative_to(dashboard_dir.parent)))
    assert not offenders, (
        "raw read-only duckdb.connect found (use read_only_connect): " + ", ".join(offenders)
    )
