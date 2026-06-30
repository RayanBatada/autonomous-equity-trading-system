"""CLI for the autoresearch loop.

Commands:
  run     — execute N iterations of propose-and-evaluate
  top     — list the highest-scoring iterations for human promotion review
  list    — list recent iterations (debug)
  diff    — show the diff between an experiment's proposed active.py and current
  promote — copy an experiment's active_py_text over the live active.py
"""

from __future__ import annotations

import difflib
import sys
from pathlib import Path

import click

from sma.ingest.store import Store
from sma.locks import heavy_job_lock, writer_lock

DEFAULT_DB = Path("data/sma.duckdb")
ACTIVE_PY_PATH = Path(__file__).parents[1] / "strategy" / "active.py"


@click.group()
def cli() -> None:
    """Phase 6 autoresearch loop CLI."""


@cli.command("run")
@click.option("--iterations", default=5, type=int,
              help="Number of propose-eval iterations to run")
@click.option("--db", default=str(DEFAULT_DB), type=click.Path())
def run_cmd(iterations: int, db: str) -> None:
    """Run N autoresearch iterations. Each calls Anthropic to propose a new
    tilt() body, evaluates it against the walk-forward CV harness, and
    logs the result. Live active.py is NEVER auto-modified — proposals
    are written to a temp swap during eval then restored."""
    from sma.autoresearch.loop import recover_stale_active_py, run_iterations

    # Startup recovery: a prior eval killed mid-swap leaves the unevaluated
    # LLM proposal as LIVE active.py (Codex module review 2026-06-11 HIGH).
    if recover_stale_active_py():
        from sma.ingest.notify import notify_failure
        notify_failure(
            title="SMA autoresearch: stranded active.py RESTORED",
            message=(
                "a prior eval died mid-swap; live active.py was an "
                "unevaluated LLM proposal until this restore"
            ),
        )

    with writer_lock(label="autoresearch"):
        store = Store(path=db).connect()
        try:
            run_id = store.allocate_run_id()
            summary = run_iterations(
                store=store, iterations=iterations, run_id=run_id,
            )
            click.echo(f"summary: {summary}")
        finally:
            store.conn.close()


@cli.command("search")
@click.option("--asof", type=click.DateTime(formats=["%Y-%m-%d"]), default=None,
              help="As-of date; defaults to today (ET) so the plist can omit it.")
@click.option("--n-configs", default=16, type=int,
              help="Number of training configs to evaluate by walk-forward CV-IC.")
@click.option("--seed", default=None, type=int,
              help="Search RNG seed. Default varies by the as-of date so each "
                   "scheduled (weekly) run explores DIFFERENT configs instead of "
                   "re-evaluating the same ones; still reproducible per date.")
@click.option("--db", default=str(DEFAULT_DB), type=click.Path())
@click.option("--models-dir", default="models_artifacts", type=click.Path(path_type=Path))
@click.option("--universe-path", default="src/sma/universe.yaml",
              type=click.Path(path_type=Path))
@click.option("--raw-labels", is_flag=True, default=False,
              help="Train on raw returns. Default is demean, to match the live "
                   "model — CV-IC is only comparable to the incumbent on the same "
                   "label, so a mismatch makes the gate defer.")
@click.option("--objective", type=click.Choice(["reg", "rank"]), default="reg",
              show_default=True)
@click.option("--dry-run", is_flag=True, default=False,
              help="Search + decide but never train/save a model (still writes a "
                   "sentinel). Use to inspect what it would promote.")
def search_cmd(asof, n_configs, seed, db, models_dir, universe_path, raw_labels,
               objective, dry_run):
    """Search the training-config space for the best held-out CV rank-IC and
    auto-promote the winner through the IC gate (same autonomy as the weekly
    retrain). Replaces the old tilt()-rewriting loop, which optimized a
    post-selection reweighter against noisy short-window Sharpe and never
    contributed (see the vault autoresearch-diagnosis note). Deterministic: no
    LLM/API in the path."""
    import time
    from datetime import UTC, datetime
    from zoneinfo import ZoneInfo

    from loguru import logger

    from sma.autoresearch.config_search import select_and_gate
    from sma.ingest.universe import load_universe
    from sma.model.__main__ import (
        DEFAULT_FORWARD_HORIZON,
        DEFAULT_TARGET,
        _current_git_sha,
        _load_earnings_calendar,
        _load_news_for_count_feature,
        _load_politician_trades,
        _load_prices_for_range,
        _train_default_start,
    )
    from sma.model.loader import build_training_set
    from sma.model.persistence import save_model
    from sma.model.trainer import DEFAULT_HYPERPARAMS, train_xgb, walk_forward_cv_rmse
    from sma.sentinels import write_sentinel

    models_dir = Path(models_dir)
    asof_date = (
        asof.date() if asof
        else datetime.now(ZoneInfo("America/New_York")).date()
    )
    label_type = "raw" if raw_labels else "demean"
    # Default the seed to the as-of date's ordinal so a weekly schedule explores
    # a fresh set of configs each run (reproducible given the date) rather than
    # re-checking the same ones forever; an explicit --seed still wins.
    if seed is None:
        seed = asof_date.toordinal()
    universe = load_universe(universe_path)
    train_start = _train_default_start()
    logger.info(
        "autoresearch search: asof={} n_configs={} seed={} label={} dry_run={}",
        asof_date, n_configs, seed, label_type, dry_run,
    )

    prices = _load_prices_for_range(Path(db), universe, train_start, asof_date)
    # Serialize this memory-heavy build against the weekly retrain's build so the
    # two never run concurrently and thrash RAM when their schedules coalesce onto
    # one wake (2026-06-29 incident). heavy_job_lock is separate from writer_lock,
    # so holding it never blocks the evening DB-writing trading jobs.
    with heavy_job_lock(label="autoresearch-build"):
        X, y, asof_dates = build_training_set(  # noqa: N806
            prices=prices,
            universe=universe,
            train_start=train_start,
            train_end=asof_date,
            forward_horizon_days=DEFAULT_FORWARD_HORIZON,
            demean_labels=not raw_labels,
            politician_trades=_load_politician_trades(Path(db)),
            earnings=_load_earnings_calendar(Path(db)),
            news=_load_news_for_count_feature(Path(db)),
        )
    if X.empty:
        click.echo("No training data; aborting.", err=True)
        raise click.exceptions.Exit(1)

    # heavy_job_lock (outer, before writer_lock for deadlock-safe ordering) keeps
    # this CV/search phase from overlapping the retrain's build — so the two
    # jobs never run heavy work concurrently.
    with heavy_job_lock(label="autoresearch-search"), writer_lock(label="autoresearch"):
        outcome = select_and_gate(
            X, y, asof_dates,
            models_dir=models_dir, asof_date=asof_date, train_start=train_start,
            label_type=label_type, target=DEFAULT_TARGET, objective=objective,
            n_configs=n_configs, seed=seed,
        )
        best = outcome.best
        promote = bool(outcome.decision.promote and not dry_run)
        model_id = None
        if promote:
            params = {**DEFAULT_HYPERPARAMS, **best.params}
            t0 = time.perf_counter()
            model = train_xgb(
                X, y, hyperparams=params, asof_dates=asof_dates, objective=objective,
            )
            train_dur = time.perf_counter() - t0
            preds = model.predict(X)
            train_rmse = float((((preds - y.to_numpy()) ** 2).mean()) ** 0.5)
            # Compute a REAL walk-forward CV-RMSE for the winner. The search
            # selects by IC, but persisting cv_rmse=nan would read back as a None
            # incumbent RMSE and silently disable the next retrain's RMSE gate
            # (Codex review 2026-06-17). One extra CV pass for the winner only.
            try:
                cv_rmse = walk_forward_cv_rmse(X, y, asof_dates, params, objective=objective)
            except Exception as e:  # noqa: BLE001 - never let this abort a promote
                logger.warning("search: winner cv_rmse computation failed: {}", e)
                cv_rmse = float("nan")
            pkl_path, _ = save_model(
                model,
                hyperparams=params,
                feature_names=list(X.columns),
                train_end_date=asof_date,
                train_rows=len(X),
                train_rmse=train_rmse,
                cv_rmse=cv_rmse,
                cv_ic=best.cv_ic,
                code_commit=_current_git_sha(),
                training_duration_seconds=train_dur,
                output_dir=models_dir,
                target=DEFAULT_TARGET,
                promoted=True,
                objective=objective,
                label_type=label_type,
                train_start=train_start,
            )
            model_id = pkl_path.stem
        # Sentinel on EVERY run (promote, hold, or dry-run) so the watchdog and
        # dashboard can see autoresearch is alive — the monitoring blind spot
        # that hid the old loop's failures for a month. The label MUST match the
        # launchd job label the watchdog/dashboard already monitor
        # (sma.schedule: com.sma.autoresearch.nightly), so turning this on needs
        # NO watchdog/dashboard change; `kind` distinguishes it from the old loop.
        write_sentinel(
            label="com.sma.autoresearch.nightly",
            asof=asof_date,
            payload={
                "label": "com.sma.autoresearch.nightly",
                "kind": "config_search",
                "asof": asof_date.isoformat(),
                "completed_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                "promoted": promote,
                "reason": outcome.decision.reason,
                "best_cv_ic": best.cv_ic,
                "incumbent_cv_ic": outcome.incumbent_cv_ic,
                "best_params": best.params,
                "n_configs": outcome.n_configs_evaluated,
                # EVERY evaluated config (best-first) — the counterfactual record
                # of what each would have scored, so a HELD run is fully
                # reviewable (`python -m sma.autoresearch history`).
                "configs": [
                    {"rank": r.rank, "cv_ic": r.cv_ic, "params": r.params}
                    for r in outcome.results
                ],
                "model_id": model_id,
                "dry_run": dry_run,
            },
        )

    verb = "PROMOTED" if promote else "held"
    click.echo(
        f"search: {verb} — best CV-IC {best.cv_ic:+.4f} "
        f"(incumbent {outcome.incumbent_cv_ic}) — {outcome.decision.reason}"
    )


@cli.command("history")
@click.option("--limit", default=20, type=int, help="How many recent runs to show.")
def history_cmd(limit: int) -> None:
    """Review recent config-search runs from their sentinels: every config that
    was evaluated (CV-IC) and whether we held or promoted — the counterfactual
    record for improving the search, INCLUDING runs that deployed nothing."""
    from sma.autoresearch.history import format_history, load_search_runs
    from sma.sentinels import sentinel_dir

    runs = load_search_runs(sentinel_dir(), limit=limit)
    click.echo(format_history(runs))


@cli.command("top")
@click.option("--k", default=10, type=int)
@click.option("--db", default=str(DEFAULT_DB), type=click.Path())
def top_cmd(k: int, db: str) -> None:
    """Print top-k experiments ranked by (monotonicity_score, overall Sharpe).
    Promotion criterion: mono >= 3 AND overall > baseline + 0.1."""
    from sma.autoresearch.experiment_log import list_top
    store = Store(path=db).connect(read_only=True)
    try:
        rows = list_top(store=store, k=k)
    finally:
        store.conn.close()
    if not rows:
        click.echo("no experiments yet — run `python -m sma.autoresearch run` first")
        return
    click.echo(f"{'exp_id':<10} {'iter':>4} {'mono':>5} {'overall':>8} summary")
    for r in rows:
        eid = r["experiment_id"][:8]
        mono = r["monotonicity_score"] if r["monotonicity_score"] is not None else "-"
        overall = (
            f"{r['sharpe_overall']:.3f}"
            if r["sharpe_overall"] is not None else "-"
        )
        click.echo(
            f"{eid:<10} {r['iter_index']:>4} {str(mono):>5} {overall:>8} "
            f"{(r['proposal_summary'] or '')[:80]}"
        )


@cli.command("list")
@click.option("--limit", default=20, type=int)
@click.option("--db", default=str(DEFAULT_DB), type=click.Path())
def list_cmd(limit: int, db: str) -> None:
    """Print last N experiments (any status)."""
    from sma.autoresearch.experiment_log import list_recent
    store = Store(path=db).connect(read_only=True)
    try:
        rows = list_recent(store=store, limit=limit)
    finally:
        store.conn.close()
    for r in rows:
        click.echo(
            f"{r['experiment_id'][:8]}  iter={r['iter_index']:<4}  "
            f"status={r['status']:<12}  mono={r['monotonicity_score']}  "
            f"summary={(r['proposal_summary'] or '')[:60]}"
        )


def _fetch_experiment_active_py(
    store, exp_id: str,
) -> tuple[str, str, str | None, float | None, int | None]:
    """Return (active_py_text, proposal_summary, status, sharpe_overall,
    monotonicity_score) for a single experiment. Raises ClickException if not
    found. Accepts a short prefix (first 8+ chars) like `top` displays."""
    rows = store.conn.execute(
        """
        SELECT experiment_id, active_py_text, proposal_summary, status,
               sharpe_overall, monotonicity_score
        FROM autoresearch_experiments
        WHERE CAST(experiment_id AS VARCHAR) LIKE ? || '%'
        ORDER BY created_at DESC
        """,
        [exp_id],
    ).fetchall()
    if not rows:
        raise click.ClickException(f"no experiment found with id prefix '{exp_id}'")
    if len(rows) > 1:
        raise click.ClickException(
            f"ambiguous id prefix '{exp_id}' matched {len(rows)} experiments; "
            "pass more characters"
        )
    _, text, summary, status, overall, mono = rows[0]
    return text, summary, status, overall, mono


@cli.command("diff")
@click.argument("exp_id")
@click.option("--db", default=str(DEFAULT_DB), type=click.Path())
def diff_cmd(exp_id: str, db: str) -> None:
    """Print a unified diff between the experiment's proposed active.py and
    the currently-live active.py. Use this to review before `promote`."""
    store = Store(path=db).connect(read_only=True)
    try:
        proposed, summary, status, overall, mono = _fetch_experiment_active_py(store, exp_id)
    finally:
        store.conn.close()
    current = ACTIVE_PY_PATH.read_text()
    if proposed == current:
        click.echo(f"experiment {exp_id} is identical to live active.py")
        return
    diff = difflib.unified_diff(
        current.splitlines(keepends=True),
        proposed.splitlines(keepends=True),
        fromfile="live active.py",
        tofile=f"experiment {exp_id}",
    )
    click.echo(
        f"# status={status} mono={mono} overall={overall} — {summary}"
    )
    sys.stdout.writelines(diff)


@cli.command("promote")
@click.argument("exp_id")
@click.option("--db", default=str(DEFAULT_DB), type=click.Path())
@click.option("--dry-run", is_flag=True, default=False,
              help="Print the would-be change without writing")
@click.option("--force", is_flag=True, default=False,
              help="Skip the eval-status sanity gates (mono/overall/status)")
def promote_cmd(exp_id: str, db: str, dry_run: bool, force: bool) -> None:
    """Replace the live `src/sma/strategy/active.py` with the experiment's
    proposed body. Sanity-gated unless --force:
      - status must be 'ok' (not eval_error / parse_error / agent_error)
      - monotonicity_score >= 3
      - sharpe_overall must beat baseline by at least 0.10 (the promotion
        criterion declared in the Phase 6 spec)

    This is reversible: git tracks active.py, so `git checkout` restores.
    The autoresearch loop is non-destructive (proposals never auto-merge),
    so this CLI is the gateway from research → production.
    """
    store = Store(path=db).connect(read_only=True)
    try:
        proposed, summary, status, overall, mono = _fetch_experiment_active_py(store, exp_id)
        baseline_row = store.conn.execute(
            "SELECT baseline_overall FROM autoresearch_experiments "
            "WHERE CAST(experiment_id AS VARCHAR) LIKE ? || '%' LIMIT 1",
            [exp_id],
        ).fetchone()
    finally:
        store.conn.close()
    baseline = float(baseline_row[0]) if baseline_row and baseline_row[0] is not None else None

    if not force:
        if status != "ok":
            raise click.ClickException(
                f"experiment status is {status!r}, not 'ok'. Use --force to override."
            )
        if mono is None or mono < 3:
            raise click.ClickException(
                f"monotonicity_score={mono} below promotion threshold (3). "
                "Use --force to override."
            )
        if overall is None or baseline is None or overall - baseline < 0.10:
            gap = (
                f"{overall - baseline:.4f}"
                if (overall is not None and baseline is not None) else "n/a"
            )
            raise click.ClickException(
                f"overall sharpe {overall} vs baseline {baseline}: gap "
                f"{gap} below promotion threshold (+0.10). Use --force to override."
            )

    if dry_run:
        click.echo(f"DRY RUN — would replace active.py with experiment {exp_id}")
        click.echo(f"  summary: {summary}")
        click.echo(f"  mono={mono} overall={overall} baseline={baseline}")
        click.echo(f"  bytes: {len(proposed)} (current: {len(ACTIVE_PY_PATH.read_text())})")
        return

    ACTIVE_PY_PATH.write_text(proposed)
    click.echo(
        f"promoted experiment {exp_id} to {ACTIVE_PY_PATH}\n"
        f"  summary: {summary}\n"
        f"  mono={mono} overall={overall} baseline={baseline}\n"
        "Review with `git diff src/sma/strategy/active.py` then commit."
    )


if __name__ == "__main__":
    cli()
