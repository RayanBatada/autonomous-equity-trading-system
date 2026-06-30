"""Autoresearch loop runner.

For each iteration:
  1. Read the current `src/sma/strategy/active.py` (the "baseline" body).
  2. Call the LLM agent → get a proposed new file content + 1-line summary.
  3. Sanity-check the proposal (signature preserved, parses as Python).
  4. Write to a TEMP path (not live `active.py`!).
  5. Run the existing backtest harness against the temp tilt across 5
     walk-forward CV sub-windows.
  6. Compute per-window Sharpe + overall + monotonicity_score.
  7. INSERT experiment row.
  8. NEVER auto-merge — the temp file is discarded; the agent's job is
     to propose candidates, the human's job is to merge winners.

Promotion is via `python -m sma.autoresearch top --k 10` (separate CLI),
which surfaces the highest-scoring iterations for the human to review.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import os
import shutil
import sys
import time
import traceback
from datetime import date, timedelta
from pathlib import Path

from loguru import logger

from sma.autoresearch import agent, experiment_log

ACTIVE_PY_PATH = Path(__file__).parents[1] / "strategy" / "active.py"

# Walk-forward CV sub-windows. Originally hardcoded to H2-2025 (locked in the
# Phase 6 spec Q3, 2026-04-29), which went stale: by mid-2026 the loop only ever
# backtested proposals on a ~1-year-old regime. They are now built dynamically
# from the latest available data (see _build_cv_windows / _resolve_cv_windows).
_CV_N_WINDOWS = 5
_CV_WINDOW_DAYS = 42  # ~6 weeks per sub-window (matches the original spans)


def _build_cv_windows(
    latest_date: date,
    n_windows: int = _CV_N_WINDOWS,
    window_days: int = _CV_WINDOW_DAYS,
) -> list[tuple[date, date]]:
    """The most recent ``n_windows`` contiguous ``window_days``-long sub-windows
    ending at ``latest_date``, oldest first.

    Replaces the hardcoded H2-2025 windows so the autoresearch eval tracks the
    current regime and auto-extends as data accumulates. Within one run, the
    baseline and every proposal use the same resolved windows, so in-run
    selection stays apples-to-apples; only comparisons ACROSS runs months apart
    shift — which is desirable, since you want recent eval.
    """
    # experiment_log persists per-window Sharpes in fixed columns
    # (sharpe_w1..sharpe_w5), and the autoresearch dashboard tab reads the same
    # five. More than five windows would be silently dropped at insert time, so
    # fail loudly until that schema is migrated.
    if n_windows > 5:
        raise ValueError(
            f"n_windows={n_windows} but experiment_log only persists w1..w5; "
            "migrate the schema (and dashboard/tabs/autoresearch.py) first."
        )
    windows: list[tuple[date, date]] = []
    end = latest_date
    for _ in range(n_windows):
        start = end - timedelta(days=window_days - 1)
        windows.append((start, end))
        end = start - timedelta(days=1)
    windows.reverse()  # oldest first, matching the original ordering
    return windows


def _resolve_cv_windows(conn) -> list[tuple[date, date]]:
    """Build the CV windows anchored to the latest price date in the DB.

    Anchoring on MAX(prices.date) assumes EOD-final bars, which holds in
    production: ingest fires post-close (18:30 ET) and autoresearch runs the
    next morning, so the anchor is always a completed session, never a partial
    intraday bar. Falls back to today's date when the DB/connection can't be
    read, so the windows are never silently stale (the point of the dynamic
    build) — though a DB that can't be read will fail the backtest anyway."""
    latest: date | None = None
    if conn is not None:
        try:
            row = conn.execute("SELECT MAX(date) FROM prices").fetchone()
            if row and row[0] is not None:
                latest = row[0]
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("autoresearch: could not read latest price date: {}", e)
    if latest is None:
        latest = date.today()
    return _build_cv_windows(latest)


# Defense-in-depth allowlist for the LLM-proposed active.py. The proposer is our
# own model (not adversarial input), but a hallucinated proposal must not be able
# to run arbitrary code when active.py is imported/reloaded or tilt() is eval'd.
# This is NOT a perfect sandbox (AST sandboxing never is) — it blocks the obvious
# import/exec/escape vectors while permitting the analytical building blocks the
# baseline active.py and a reasonable tilt() need.
_ALLOWED_IMPORT_PREFIXES = (
    "__future__", "dataclasses", "datetime", "typing", "collections",
    "math", "statistics", "numpy", "pandas", "sma",
)
_BANNED_NAMES = frozenset({
    "__import__", "eval", "exec", "compile", "open", "globals", "locals",
    "vars", "input", "breakpoint",
})
_BANNED_ATTRS = frozenset({
    "__globals__", "__builtins__", "__subclasses__", "__bases__", "__mro__",
    "__class__", "__code__", "__closure__", "__dict__", "__getattribute__",
})


def _import_allowed(module: str | None) -> bool:
    if not module:  # relative import (`from . import x`) — disallowed
        return False
    return any(module == p or module.startswith(p + ".") for p in _ALLOWED_IMPORT_PREFIXES)


def _verify_proposal_safe(proposed_py: str) -> None:
    """Static checks before we trust the proposal to run.

    Raises RuntimeError on signature drift, syntax errors, OR any code the
    allowlist disallows (top-level statements other than imports/funcs/classes/
    docstring, imports outside the allowlist, or banned builtins/dunder escapes).
    """
    try:
        tree = ast.parse(proposed_py)
    except SyntaxError as e:
        raise RuntimeError(f"proposal does not parse: {e}") from e

    found_tilt = False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "tilt":
            found_tilt = True
            # Signature must be (*, asof_date, decisions, ctx).
            args = node.args
            if args.args:
                raise RuntimeError("tilt() must have no positional args")
            kwonly = [a.arg for a in args.kwonlyargs]
            if kwonly != ["asof_date", "decisions", "ctx"]:
                raise RuntimeError(
                    f"tilt() kw-only args drifted; expected "
                    f"['asof_date', 'decisions', 'ctx'], got {kwonly}"
                )
    if not found_tilt:
        raise RuntimeError("proposal does not define tilt()")

    # Top-level may contain ONLY a docstring, imports, functions, and classes —
    # nothing that executes arbitrary code at import time.
    for node in tree.body:
        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            continue  # module docstring
        if isinstance(
            node,
            (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
        ):
            continue
        raise RuntimeError(
            f"proposal has a disallowed top-level statement: {type(node).__name__} "
            "(only a docstring, imports, functions, and classes are allowed)"
        )

    # Anywhere (incl. inside tilt): imports must be allowlisted; no banned
    # builtins or dunder-attribute escapes.
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if not _import_allowed(alias.name):
                    raise RuntimeError(f"proposal imports a disallowed module: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if not _import_allowed(node.module):
                raise RuntimeError(f"proposal imports from a disallowed module: {node.module}")
        elif isinstance(node, ast.Name) and node.id in _BANNED_NAMES:
            raise RuntimeError(f"proposal references a banned builtin: {node.id}")
        elif isinstance(node, ast.Attribute) and node.attr in _BANNED_ATTRS:
            raise RuntimeError(f"proposal accesses a banned attribute: {node.attr}")


def _evaluate_proposal(
    proposed_py: str,
    *,
    cv_windows: list[tuple[date, date]],
    conn=None,
) -> tuple[dict, str | None]:
    """Apply the proposed `active.py` to a sandbox path, reimport, run the
    walk-forward CV over `cv_windows`. Returns (sharpes_dict, error_msg). On
    error, sharpes is empty.

    Sharpes dict: {"w1": x, ..., "wN": x, "overall": x}.

    NOTE: This swaps the live `active.py` for the duration of the eval.
    The caller must restore the original on completion. Done via the
    `with _SwappedActivePy(...)` ctx manager in the run() flow.

    `conn` reuses the caller's writable DuckDB connection to avoid the
    "different configuration than existing connections" error when the
    autoresearch loop holds a writer_lock with its own connection
    (regression caught 2026-05-18 first live fire).
    """
    # Reimport strategy.active so the new tilt body is picked up. Also
    # reload xgb_top_k so its `from sma.strategy.active import tilt` re-resolves.
    if "sma.strategy.active" in sys.modules:
        importlib.reload(sys.modules["sma.strategy.active"])
    else:
        importlib.import_module("sma.strategy.active")
    if "sma.backtest.strategies.xgb_top_k" in sys.modules:
        importlib.reload(sys.modules["sma.backtest.strategies.xgb_top_k"])

    sharpes: dict[str, float | None] = {f"w{i+1}": None for i in range(len(cv_windows))}
    sharpes["overall"] = None

    try:
        per_window: list[float] = []
        for win_start, win_end in cv_windows:
            sharpe = _run_window_sharpe(win_start, win_end, conn=conn)
            per_window.append(sharpe)
        for i, s in enumerate(per_window):
            sharpes[f"w{i+1}"] = s
        sharpes["overall"] = sum(per_window) / len(per_window) if per_window else None
    except Exception as e:
        return sharpes, f"eval failed: {e}\n{traceback.format_exc()[:500]}"
    return sharpes, None


def _run_window_sharpe(
    win_start: date, win_end: date, *, conn=None,
) -> float:
    """Run the real backtest harness over [win_start, win_end] with the
    currently-imported `active.py`'s tilt() applied. Returns the
    BacktestResult's Sharpe scalar.

    Loads prices, builds a fresh XGBoostTopKStrategy (which picks up the
    currently-imported `sma.strategy.active.tilt` because xgb_top_k.decide()
    imports tilt at call time), runs simulate(), and returns the Sharpe.
    Raises on any error so the caller's outer try/except records an
    eval_error experiment row.

    `conn` should be passed by the autoresearch loop (which holds a writable
    Store conn under writer_lock). DuckDB rejects opening additional
    connections with different read/write configs on the same file, so the
    sub-queries must reuse the existing handle.
    """
    from pathlib import Path

    from sma.backtest.simulator import simulate
    from sma.backtest.strategies.xgb_top_k import XGBoostTopKStrategy
    from sma.eval.evaluate_strategy import _load_prices_for_window
    from sma.ingest.universe import load_universe
    from sma.model.predictor import Predictor
    from sma.sectors import sector_map_for

    universe = load_universe(Path("src/sma/universe.yaml"))
    db_path = Path("data/sma.duckdb")
    prices = _load_prices_for_window(
        db_path, universe, win_start, win_end, conn=conn,
    )
    predictor = Predictor(
        models_dir=Path("models_artifacts"), db_path=db_path, conn=conn,
    )
    strategy = XGBoostTopKStrategy(
        predictor=predictor,
        universe=universe,
        # Eval must NOT swallow a crashing proposed tilt() — surface it so the
        # experiment is recorded as eval_error instead of the untilted baseline.
        tilt_strict=True,
    )
    from sma.backtest.risk import eval_rails_for

    result = simulate(
        strategy=strategy,
        universe=universe,
        prices=prices,
        sector_map=sector_map_for(universe),
        window_name="val",
        start_date=win_start,
        end_date=win_end,
        # Research eval over historical windows: current-universe membership
        # accepted EXPLICITLY (same stance as the backtest CLI) until a true
        # point-in-time membership map exists — simulate(membership=None)
        # means no filtering, i.e. survivorship-biased like every other
        # research eval here (Codex strategy review 2026-06-11 #8).
        membership=None,
        # Default rails (5% cap) REJECT every 10%-target decision — without
        # this every autoresearch experiment scores a zero-trade portfolio
        # (2026-06-11 Codex finding, inverse-selection class).
        rails=eval_rails_for(strategy),
    )
    sharpe = result.sharpe
    return float(sharpe) if sharpe is not None else 0.0


def _compute_monotonicity(
    sharpes: dict[str, float | None], baseline_per_window: list[float],
    threshold: float = 0.05,
) -> int:
    """Count windows where sharpe[w] >= baseline[w] + threshold."""
    score = 0
    for i, baseline in enumerate(baseline_per_window):
        cur = sharpes.get(f"w{i+1}")
        if cur is None:
            continue
        if cur >= baseline + threshold:
            score += 1
    return score


def recover_stale_active_py() -> bool:
    """Startup recovery: a SIGKILL/power-loss between _SwappedActivePy's
    __enter__ and __exit__ leaves the UNEVALUATED LLM proposal swapped into
    the LIVE src/sma/strategy/active.py, with the original stranded in
    .py.autoresearch_bak — live decide would import the proposal (Codex
    module review 2026-06-11 HIGH). Called at autoresearch startup; the
    watchdog separately pages if the backup file is ever seen. Returns True
    when a restore happened."""
    bak = ACTIVE_PY_PATH.with_suffix(".py.autoresearch_bak")
    if not bak.exists():
        return False
    logger.warning(
        "recovering stranded active.py from {} (a prior eval died mid-swap)",
        bak,
    )
    os.replace(bak, ACTIVE_PY_PATH)
    ACTIVE_PY_PATH.with_suffix(".py.autoresearch_tmp").unlink(missing_ok=True)
    if "sma.strategy.active" in sys.modules:
        importlib.reload(sys.modules["sma.strategy.active"])
    return True


class _SwappedActivePy:
    """Context manager that swaps active.py for the proposed body during
    eval, then restores the original on exit (even on exception)."""

    def __init__(self, proposed_text: str):
        self.proposed_text = proposed_text
        self.backup_path = ACTIVE_PY_PATH.with_suffix(".py.autoresearch_bak")

    def __enter__(self) -> None:
        shutil.copy(ACTIVE_PY_PATH, self.backup_path)
        # Atomic write: write to a temp sibling then os.replace, so a partial
        # write (disk full, crash, iCloud lock) never leaves the LIVE active.py
        # truncated — live trading imports this file.
        tmp = ACTIVE_PY_PATH.with_suffix(".py.autoresearch_tmp")
        tmp.write_text(self.proposed_text)
        os.replace(tmp, ACTIVE_PY_PATH)

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: D401
        # Restoring the live active.py is CRITICAL — a failure here leaves the
        # LLM proposal running in production. Use an atomic rename (can't
        # partially fail) and a try/finally so the reload + cleanup always run.
        try:
            os.replace(self.backup_path, ACTIVE_PY_PATH)
        finally:
            self.backup_path.unlink(missing_ok=True)
            ACTIVE_PY_PATH.with_suffix(".py.autoresearch_tmp").unlink(missing_ok=True)
            # Force reimport so subsequent code sees the restored module.
            if "sma.strategy.active" in sys.modules:
                importlib.reload(sys.modules["sma.strategy.active"])


def run_iterations(
    *, store, iterations: int, run_id: int,
    baseline_per_window: list[float] | None = None,
) -> dict:
    """Run N autoresearch iterations. Returns a summary dict."""
    summary = {
        "iterations": iterations,
        "ok": 0, "agent_errors": 0, "eval_errors": 0, "parse_errors": 0,
        "total_cost_usd": 0.0,
    }

    baseline_text = ACTIVE_PY_PATH.read_text()
    # Reuse the autoresearch store's connection so sub-loads don't open a
    # second connection with mismatched read/write config (2026-05-18 fix).
    eval_conn = getattr(store, "conn", None)
    # Resolve the CV windows ONCE from the latest available data so the whole
    # run (baseline + every proposal) evaluates on the same current-regime
    # windows, instead of the formerly-hardcoded (now stale) H2-2025 set.
    cv_windows = _resolve_cv_windows(eval_conn)
    logger.info(
        "autoresearch CV windows: {} .. {} ({} windows)",
        cv_windows[0][0], cv_windows[-1][1], len(cv_windows),
    )
    if baseline_per_window is None:
        # Real backtest with the live (identity) tilt() body — used as the
        # baseline Sharpe per sub-window. Proposals must beat this by >0.05
        # in a window to score on the monotonicity criterion.
        baseline_per_window = [
            _run_window_sharpe(s, e, conn=eval_conn) for s, e in cv_windows
        ]
    baseline_overall = sum(baseline_per_window) / len(baseline_per_window)

    recent = experiment_log.list_recent(store=store, limit=20)

    # Compute the live perf snapshot ONCE at run start. Sticking the same
    # string into every iteration's prompt keeps the cache prefix stable
    # so iteration 2+ are ~80% cheaper (prompt-cache reads vs full input).
    perf_context = agent.compute_live_perf_context(store)
    logger.info("autoresearch perf context for proposer:\n{}", perf_context)

    for i in range(iterations):
        start = time.monotonic()
        try:
            proposal = agent.propose(
                current_active_py=baseline_text,
                recent_experiments=recent,
                perf_context=perf_context,
            )
        except Exception as e:
            logger.warning("iter {}: agent error: {}", i, e)
            experiment_log.insert_experiment(
                store=store, run_id=run_id, iter_index=i,
                proposal_sha="", proposal_summary="",
                active_py_text="",
                baseline_overall=baseline_overall,
                sharpes={}, monotonicity_score=None,
                status="agent_error", error=str(e)[:500],
                agent_cost_usd=0.0,
                duration_seconds=time.monotonic() - start,
            )
            summary["agent_errors"] += 1
            continue

        sha = hashlib.sha256(proposal.active_py_text.encode()).hexdigest()[:16]
        try:
            _verify_proposal_safe(proposal.active_py_text)
        except Exception as e:
            logger.warning("iter {}: proposal rejected by static check: {}", i, e)
            experiment_log.insert_experiment(
                store=store, run_id=run_id, iter_index=i,
                proposal_sha=sha, proposal_summary=proposal.summary,
                active_py_text=proposal.active_py_text,
                baseline_overall=baseline_overall,
                sharpes={}, monotonicity_score=None,
                status="parse_error", error=str(e)[:500],
                agent_cost_usd=proposal.cost_usd,
                duration_seconds=time.monotonic() - start,
            )
            summary["parse_errors"] += 1
            summary["total_cost_usd"] += proposal.cost_usd
            continue

        # Eval inside the swap context — guaranteed restoration.
        sharpes: dict = {}
        eval_err: str | None = None
        with _SwappedActivePy(proposal.active_py_text):
            sharpes, eval_err = _evaluate_proposal(
                proposal.active_py_text, cv_windows=cv_windows, conn=eval_conn,
            )

        if eval_err:
            experiment_log.insert_experiment(
                store=store, run_id=run_id, iter_index=i,
                proposal_sha=sha, proposal_summary=proposal.summary,
                active_py_text=proposal.active_py_text,
                baseline_overall=baseline_overall,
                sharpes={}, monotonicity_score=None,
                status="eval_error", error=eval_err[:500],
                agent_cost_usd=proposal.cost_usd,
                duration_seconds=time.monotonic() - start,
            )
            summary["eval_errors"] += 1
            summary["total_cost_usd"] += proposal.cost_usd
            continue

        mono = _compute_monotonicity(sharpes, baseline_per_window)
        experiment_log.insert_experiment(
            store=store, run_id=run_id, iter_index=i,
            proposal_sha=sha, proposal_summary=proposal.summary,
            active_py_text=proposal.active_py_text,
            baseline_overall=baseline_overall,
            sharpes=sharpes, monotonicity_score=mono,
            status="ok", error=None,
            agent_cost_usd=proposal.cost_usd,
            duration_seconds=time.monotonic() - start,
        )
        summary["ok"] += 1
        summary["total_cost_usd"] += proposal.cost_usd
        logger.info(
            "iter {}: mono={} overall={:.3f} cost=${:.4f} — {}",
            i, mono, sharpes.get("overall") or 0.0,
            proposal.cost_usd, proposal.summary[:60],
        )

    return summary
