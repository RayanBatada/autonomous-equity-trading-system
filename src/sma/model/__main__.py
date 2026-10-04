"""CLI for the XGBoost quant model: train, predict, backfill-predictions."""

import time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import click
import pandas as pd
from loguru import logger

from sma.config import load_model_config
from sma.db_connect import read_only_connect
from sma.ingest.notify import notify_failure
from sma.ingest.universe import load_training_membership, load_universe
from sma.locks import heavy_job_lock, writer_lock
from sma.model.ensemble import (
    ensemble_random_states,
    ensemble_size,
    seeds_are_inert,
)
from sma.model.loader import build_training_set
from sma.model.persistence import (
    GATE_MAX_CV_RMSE_RATIO,
    incumbent_cv_rmse,
    incumbent_label_type,
    incumbent_train_start,
    latest_model_for_date,
    passes_ic_floor,
    prune_old_artifacts,
    save_model,
    should_promote,
    write_predictions,
)
from sma.model.predictor import Predictor
from sma.model.trainer import (
    DEFAULT_HYPERPARAMS,
    select_hyperparams_keep_better_ic,
    train_xgb,
)
from sma.sentinels import read_sentinel, write_sentinel

DEFAULT_DB_PATH = Path("data/sma.duckdb")
DEFAULT_MODELS_DIR = Path("models_artifacts")
DEFAULT_UNIVERSE_PATH = Path("src/sma/universe.yaml")
# Point-in-time membership map (survivorship fix). Sits beside universe.yaml;
# absent file → {} → training behaves exactly as it did before it existed.
DEFAULT_UNIVERSE_HISTORY_PATH = Path("src/sma/universe_history.yaml")
DEFAULT_TARGET = "ret_30d_forward"
DEFAULT_FORWARD_HORIZON = 30
def _train_default_start() -> date:
    """Training-history start: $SMA_TRAIN_START (ISO) if set, else 2023-01-01.
    The env override lets the multi-regime experiment (2026-06-15) train across
    pre-2023 crashes without editing code; production uses the default."""
    import os
    v = os.environ.get("SMA_TRAIN_START")
    return date.fromisoformat(v) if v else date(2018, 1, 1)


def _resolve_ensemble_seeds(cli_value, *, config_path="config.yaml") -> int:
    """How many seeds this retrain trains: the --ensemble-seeds flag when
    given, otherwise model.ensemble_seeds from config.yaml (default 10).

    The scheduled plist passes no flag and runs with WorkingDirectory set to
    the repo root, so the relative config path resolves for the 04:00 job.
    """
    n = cli_value if cli_value is not None else load_model_config(config_path).ensemble_seeds
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError(f"ensemble_seeds must be an integer >= 1; got {n!r}")
    return n


def _resolve_feature_workers(cli_value, *, config_path="config.yaml") -> int:
    """How many processes the per-asof feature build fans out over: the
    --feature-workers flag when given, otherwise model.feature_workers from
    config.yaml (default min(4, cpu_count - 1)).

    Same shape and same tolerance as _resolve_ensemble_seeds: the scheduled
    plist passes no flag and runs with WorkingDirectory at the repo root, so
    the relative config path resolves for the 04:00 job, and an unreadable
    config falls back to the default rather than killing the retrain.
    """
    n = (
        cli_value if cli_value is not None
        else load_model_config(config_path).feature_workers
    )
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError(f"feature_workers must be an integer >= 1; got {n!r}")
    return n


# 2026-06-16: default moved 2023 -> 2018 after the multi-regime experiment.
# Training across the 2018 selloff / 2020 COVID crash / 2022 bear FLIPPED the
# held-out val IC from -0.027 (t -1.35) to +0.028 (t +1.88) — the model no
# longer inverts in reversal regimes because it has now seen crashes. Requires
# pre-2023 price history (backfilled via scripts/backfill_history_2018.py).
TRAIN_DEFAULT_START = date(2018, 1, 1)  # back-compat module constant (default)


def _load_politician_trades(db_path: Path) -> pd.DataFrame:
    """Load all politician_trades rows for use as the politician_flow_30d
    feature. Returns empty DataFrame if the table doesn't exist (older DB
    without v5 migration) or has no rows."""
    if not db_path.exists():
        return pd.DataFrame()
    con = read_only_connect(db_path)  # 2026-08-05 audit: retry lock overlap w/ ingest
    try:
        df = con.execute(
            "SELECT ticker, filing_date, transaction_date, transaction_type, "
            "amount_min, amount_max FROM politician_trades "
            "WHERE ticker IS NOT NULL "
            # Exclude options/derivatives so they don't pollute the stock-flow
            # feature. MUST match predictor._fetch_politician_flows exactly.
            "AND COALESCE(asset_type, '') NOT IN ('Stock Option', 'OP')"
        ).df()
    except Exception:
        df = pd.DataFrame()
    finally:
        con.close()
    if not df.empty:
        df["transaction_date"] = pd.to_datetime(df["transaction_date"]).dt.date
    return df


def _load_news_for_count_feature(db_path: Path) -> pd.DataFrame:
    """Load just (ticker, published_at) for use as the news_count_7d_log
    feature. Returns empty DataFrame if table missing. Pre-computes a
    `published_at_date` column to avoid per-row date conversion in the
    inner training loop."""
    if not db_path.exists():
        return pd.DataFrame()
    con = read_only_connect(db_path)  # 2026-08-05 audit: retry lock overlap w/ ingest
    try:
        df = con.execute(
            "SELECT ticker, CAST(published_at AS DATE) AS published_at_date "
            "FROM news WHERE ticker IS NOT NULL AND published_at IS NOT NULL"
        ).df()
    except Exception:
        df = pd.DataFrame()
    finally:
        con.close()
    if not df.empty:
        df["published_at_date"] = pd.to_datetime(df["published_at_date"]).dt.date
    return df


def _load_earnings_calendar(db_path: Path) -> pd.DataFrame:
    """Load (ticker, report_date, eps_estimate, eps_actual) rows from
    `earnings` for the days_to_next_earnings AND earnings_surprise_last
    features. Returns empty DataFrame if the table doesn't exist or is empty.

    The eps columns are REQUIRED: loader._compute_latest_surprises defensively
    returns {} when they're absent, so a ticker+date-only SELECT silently
    trained earnings_surprise_last as a constant 0.0 while predict served real
    values — train/serve skew, live since the feature shipped (review
    2026-07-01 HIGH)."""
    if not db_path.exists():
        return pd.DataFrame()
    con = read_only_connect(db_path)  # 2026-08-05 audit: retry lock overlap w/ ingest
    try:
        df = con.execute(
            "SELECT DISTINCT ticker, report_date, eps_estimate, eps_actual "
            "FROM earnings "
            "WHERE ticker IS NOT NULL AND report_date IS NOT NULL "
            "ORDER BY ticker, report_date"
        ).df()
    except Exception:
        df = pd.DataFrame()
    finally:
        con.close()
    if not df.empty:
        df["report_date"] = pd.to_datetime(df["report_date"]).dt.date
    return df


def _load_prices_for_range(
    db_path: Path,
    universe: list[str],
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """Load prices for the universe (plus SPY) over [start_date, end_date]."""
    if not db_path.exists():
        raise FileNotFoundError(f"DuckDB not found at {db_path}.")
    tickers = list(set(list(universe) + ["SPY"]))
    con = read_only_connect(db_path)  # 2026-08-05 audit: retry lock overlap w/ ingest
    try:
        df = con.execute(
            """
            SELECT ticker, date, open, high, low, close, adj_close, volume
            FROM (
                SELECT *,
                       ROW_NUMBER() OVER (
                           PARTITION BY ticker, date
                           ORDER BY CASE source WHEN 'yfinance' THEN 0
                                                 WHEN 'alpaca' THEN 1
                                                 ELSE 2 END
                       ) AS rn
                FROM prices
                WHERE ticker = ANY($tickers)
                  AND date BETWEEN $start_date AND $end_date
                  AND adj_close IS NOT NULL
            ) t
            WHERE rn = 1
            ORDER BY ticker, date
        """,
            {"tickers": tickers, "start_date": start_date, "end_date": end_date},
        ).df()
    finally:
        con.close()
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def _current_git_sha() -> str:
    """Return current commit SHA, or empty string if not a git repo."""
    import subprocess

    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=Path(__file__).resolve().parent.parent.parent,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


@click.group()
def cli():
    """Phase 2 quant model operations."""
    pass


# ============================================================
# Task 15: train
# ============================================================
@cli.command()
@click.option(
    "--asof",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    default=None,
    help="Train on data with asof_date <= ASOF - 30 trading days. "
         "Defaults to today (ET) so the launchd plist can omit it.",
)
@click.option(
    "--no-cv", is_flag=True, default=False, help="Skip CV; use DEFAULT_HYPERPARAMS directly."
)
@click.option("--db-path", type=click.Path(path_type=Path), default=DEFAULT_DB_PATH)
@click.option("--models-dir", type=click.Path(path_type=Path), default=DEFAULT_MODELS_DIR)
@click.option("--universe-path", type=click.Path(path_type=Path), default=DEFAULT_UNIVERSE_PATH)
@click.option(
    "--demean-labels/--raw-labels", "demean_labels", default=True,
    help="Train on cross-sectionally demeaned (relative/alpha) 30d returns "
         "(the production standard since 2026-06-13) vs raw returns, which let "
         "the model learn beta. DEFAULT IS DEMEAN: a scheduler entry that "
         "omits the flag must train the production label type — the 2026-06-29 "
         "incident (installed plist lost --demean-labels, silently deploying a "
         "raw model via the label-transition promote rule) is why. Raw now "
         "requires an explicit --raw-labels.",
)
@click.option(
    "--label-stride",
    type=int,
    default=1,
    help="Keep every Nth asof_date in the training set to thin overlapping "
         "labels (effective N ~ N/horizon at stride=1). 1 = all dates.",
)
@click.option(
    "--objective",
    type=click.Choice(["reg", "rank"]),
    default="reg",
    show_default=True,
    help="Model objective: regression returns or per-asof-date pairwise ranking.",
)
@click.option(
    "--universe-history-path",
    type=click.Path(path_type=Path),
    default=DEFAULT_UNIVERSE_HISTORY_PATH,
    help="Point-in-time membership map. Absent file = no restriction.",
)
@click.option(
    "--ensemble-seeds",
    type=int,
    default=None,
    help="Train N models differing only in random_state and score their MEAN "
         "(the 2026-08-21 ensemble-rank study's adopted B_ens arm, taken for "
         "seed-variance reduction rather than a mean-IC gain). Omit to use "
         "model.ensemble_seeds from config.yaml (10). 1 = the single-model "
         "behaviour that trained every artifact before 2026-08-24. The "
         "training set is built ONCE and shared across the N fits.",
)
@click.option(
    "--feature-workers",
    type=int,
    default=None,
    help="Processes to fan the per-asof feature build out over. The feature "
         "build is 65-80 min of this job and the asof axis is embarrassingly "
         "parallel; output is BIT-IDENTICAL at any worker count. Omit to use "
         "model.feature_workers from config.yaml (min(4, cpu_count - 1)). "
         "1 = the exact serial path, no pool.",
)
@click.option(
    "--pit-universe/--no-pit-universe", "pit_universe", default=True,
    help="Restrict each ticker's training rows to its point-in-time membership "
         "window from universe_history.yaml. DEFAULT IS ON: without it the "
         "training set treats today's April-2026 universe as having always "
         "existed, which the breadth study measured at +0.76pp of 30d top-15 "
         "excess overall and +2.73pp on 2023+. --no-pit-universe reproduces "
         "the pre-2026-08-17 (survivorship-biased) training set for A/B.",
)
def train(
    asof,
    no_cv,
    db_path,
    models_dir,
    universe_path,
    demean_labels,
    label_stride,
    objective,
    ensemble_seeds,
    universe_history_path,
    feature_workers,
    pit_universe,
):
    """Train an XGBoost model up to ASOF and save it.

    The writer_lock is held for the full training duration (potentially hours
    for CV). This is intentional: retrain runs Saturday 02:00 ET when no other
    scheduled jobs are running (ingest/predict/decide are weekday-only).
    Holding the lock prevents a racing manual run or backfill from interleaving
    writes during training.
    """
    if asof is None:
        from zoneinfo import ZoneInfo
        asof_date = datetime.now(ZoneInfo("America/New_York")).date()
    else:
        asof_date = asof.date()
    universe = load_universe(universe_path)

    n_ensemble = _resolve_ensemble_seeds(ensemble_seeds)
    logger.info(
        f"Seed ensemble: {n_ensemble} model(s) per fit"
        + (" — CV gates score the ensemble MEAN (study B_ens, 2026-08-21)."
           if n_ensemble > 1 else " — single-model (pre-2026-08-24 behaviour).")
    )
    n_feature_workers = _resolve_feature_workers(feature_workers)
    logger.info(
        f"Feature build: {n_feature_workers} worker process(es)"
        + (" — per-asof fan-out, bit-identical to serial."
           if n_feature_workers > 1 else " — serial (no pool).")
    )

    # Point-in-time membership (survivorship fix, activated 2026-08-17). Each
    # ticker contributes rows only inside [added, removed). Empty map — file
    # absent, or --no-pit-universe — restores the previous training set exactly.
    membership: dict[str, tuple[date | None, date | None]] = {}
    if pit_universe:
        membership = load_training_membership(universe_history_path)
        if membership:
            restricted = sum(
                1 for t in universe
                if membership.get(t, (None, None)) != (None, None)
            )
            logger.info(
                f"PIT universe ON: {len(membership)} membership intervals from "
                f"{universe_history_path}; {restricted}/{len(universe)} "
                f"universe tickers carry a date bound."
            )
        else:
            logger.warning(
                f"PIT universe requested but {universe_history_path} is absent "
                "or empty — training on the full (survivorship-biased) universe."
            )
    else:
        logger.warning(
            "PIT universe OFF (--no-pit-universe): training treats today's "
            "universe as having always existed."
        )

    logger.info(f"Loading prices for {len(universe)} tickers up to {asof_date}")
    _train_start = _train_default_start()
    prices = _load_prices_for_range(db_path, universe, _train_start, asof_date)
    logger.info(f"Loaded {len(prices)} price rows.")

    # 2026-05-10: load politician trades so the new politician_flow_30d
    # feature gets populated in the training set. Empty DataFrame is OK
    # (means feature defaults to 0 for every sample, which is expected
    # for asof_dates that predate the PTR backfill).
    politician_trades_df = _load_politician_trades(db_path)
    logger.info(f"Loaded {len(politician_trades_df)} politician trade rows.")

    # 2026-05-24: load earnings calendar for days_to_next_earnings feature.
    # Empty DataFrame → feature defaults to 60d (cap, ≈ "no earnings soon")
    # for every sample. Coverage is ~641 rows spanning 2000-2026 currently.
    earnings_df = _load_earnings_calendar(db_path)
    logger.info(f"Loaded {len(earnings_df)} earnings calendar rows.")

    # 2026-05-26: load news rows for news_count_7d_log feature. ~318k rows
    # across finnhub + alpaca:benzinga + newsapi sources. Heavy-tailed
    # coverage — top names like GOOGL/NVDA see 500-700 articles/week, median
    # ~50, long tail of small names with <10. Empty DataFrame → feature
    # defaults to 0 (log1p(0)) which the model treats as "no attention".
    news_df = _load_news_for_count_feature(db_path)
    logger.info(f"Loaded {len(news_df)} news rows.")

    logger.info("Building training set (this may take a minute)...")
    # Serialize this memory-heavy build against the autoresearch search so the
    # two never run concurrently and thrash RAM when their schedules coalesce
    # onto a single wake (2026-06-29 incident). heavy_job_lock is separate from
    # writer_lock, so it never blocks the evening trading jobs.
    with heavy_job_lock(label="retrain-build"):
        X, y, asof_dates = build_training_set(  # noqa: N806
            prices=prices,
            universe=universe,
            train_start=_train_start,
            train_end=asof_date,
            forward_horizon_days=DEFAULT_FORWARD_HORIZON,
            label_stride=label_stride,
            demean_labels=demean_labels,
            politician_trades=politician_trades_df,
            earnings=earnings_df,
            news=news_df,
            membership=membership,
            feature_workers=n_feature_workers,
        )
    logger.info(f"Training set: {len(X)} rows.")
    if X.empty:
        click.echo("No training data; aborting.", err=True)
        raise click.exceptions.Exit(1)

    # heavy_job_lock (outer, acquired before writer_lock for deadlock-safe
    # ordering) keeps the CV/train phase from overlapping the autoresearch
    # build — so no heavy phase of one job runs while the other is heavy.
    with heavy_job_lock(label="retrain-train"), writer_lock(label="retrain"):
        cv_ic = float("nan")
        if no_cv:
            hyperparams = DEFAULT_HYPERPARAMS.copy()
            cv_rmse = float("nan")
            logger.info("Skipping CV (--no-cv); deploy gate will not apply.")
        else:
            logger.info("Running walk-forward CV hyperparameter search...")
            cv_start = time.perf_counter()
            # Pick the RMSE-grid winner, but KEEP the deployed model's OWN config
            # if it ranks better on held-out CV-IC here — so a good config (from a
            # past retrain or an autoresearch win) persists across weeks instead of
            # being re-rolled by RMSE every time (2026-06-23). cv_ic is the metric
            # that actually tracks ranking edge (RMSE is point-error, not order).
            hyperparams, cv_ic, cv_rmse = select_hyperparams_keep_better_ic(
                X, y, asof_dates,
                models_dir=models_dir, asof_date=asof_date,
                target=DEFAULT_TARGET, objective=objective,
                ensemble_seeds=n_ensemble,
            )
            logger.info(
                f"CV done in {time.perf_counter() - cv_start:.1f}s. "
                f"Chosen hyperparams: {hyperparams}, CV RMSE: {cv_rmse:.5f}, "
                f"CV rank-IC: {cv_ic:+.4f}"
            )

        if n_ensemble > 1 and seeds_are_inert(hyperparams):
            # An XGBoost fit is only seed-dependent through its stochastic
            # parts. At subsample/colsample >= 1.0 the N members would be
            # bit-identical: N times the cost, zero variance reduction, and
            # silent about it.
            logger.warning(
                "ensemble_seeds={} but the chosen hyperparameters have no "
                "stochastic component (subsample/colsample >= 1.0) — every "
                "member will be an IDENTICAL model at {}x the fit cost.",
                n_ensemble, n_ensemble,
            )

        logger.info("Training production model...")
        train_start = time.perf_counter()
        # X/y were built ONCE above (the 65-80 min phase). The N seed fits
        # share that one matrix — only the XGBoost fit repeats, seconds each.
        model = train_xgb(
            X,
            y,
            hyperparams=hyperparams,
            asof_dates=asof_dates,
            objective=objective,
            ensemble_seeds=n_ensemble,
        )
        train_duration = time.perf_counter() - train_start
        if ensemble_size(model) > 1:
            logger.info(
                f"Ensemble trained on random_states {ensemble_random_states(model)}"
            )
        # True IN-SAMPLE RMSE (fit error). The gap vs cv_rmse is the overfit
        # signal; this is what `train_rmse` should mean (historically it was
        # mistakenly fed the CV value).
        _preds = model.predict(X)
        train_rmse = float((((_preds - y.to_numpy()) ** 2).mean()) ** 0.5)
        logger.info(
            f"Training complete in {train_duration:.1f}s. "
            f"in-sample RMSE {train_rmse:.5f}, CV RMSE {cv_rmse:.5f}"
        )

        # ---- Deploy gate: don't auto-promote a model materially worse OOS than
        # the incumbent. predict() picks the newest artifact in models_dir, so a
        # rejected model is quarantined to models_dir/"rejected" (invisible to
        # latest_model_for_date's non-recursive glob) and the incumbent keeps
        # serving. Look up the incumbent at asof_date (latest_model_for_date
        # serves date <= asof, and the new model isn't saved yet, so this returns
        # the true currently-deployed model — including a same-date one on a
        # re-run — and never the model we're about to write).
        incumbent = incumbent_cv_rmse(models_dir, asof_date, DEFAULT_TARGET)
        _new_label_type = "demean" if demean_labels else "raw"
        _inc_label_type = incumbent_label_type(models_dir, asof_date, DEFAULT_TARGET)
        if not passes_ic_floor(cv_ic, cv_ran=not no_cv):
            # Top-level safety: a model that ranks WORSE than chance
            # out-of-sample (held-out CV-IC below the floor) must never deploy,
            # whatever its RMSE or a label transition says — RMSE can't see
            # ranking inversion, the 2026-06-13 regime-luck trap. Quarantine it;
            # the incumbent keeps serving.
            logger.warning(
                "RETRAIN NOT PROMOTED: CV rank-IC {:+.4f} below floor — model "
                "ranks worse than chance OOS; quarantining.", cv_ic,
            )
            promote = False
        elif _inc_label_type is not None and _inc_label_type != _new_label_type:
            # Clean label-type transition: RMSE scales aren't comparable across
            # raw vs demeaned targets, so promote the new type (2026-06-13).
            # A transition is ~always a deliberate one-time event, so PAGE it:
            # the 2026-06-29 incident rode this exact branch to silently deploy
            # an accidental demean->raw regression (plist drift dropped the
            # flag). Loud beats silent for anything that changes what the live
            # model learns.
            logger.warning(
                "label_type transition {} -> {}; promoting (RMSE incomparable)",
                _inc_label_type, _new_label_type,
            )
            notify_failure(
                title="SMA retrain: label-type TRANSITION deployed",
                message=(
                    f"live model label_type changed {_inc_label_type} -> "
                    f"{_new_label_type}. If this wasn't deliberate, the "
                    "scheduler args regressed (see 2026-06-29 incident)."
                ),
            )
            promote = True
        elif (
            incumbent is not None
            and incumbent_train_start(models_dir, asof_date, DEFAULT_TARGET)
            != _train_start.isoformat()
        ):
            # Training-WINDOW change (e.g. 2023-only -> 2018 multi-regime): RMSE
            # isn't comparable across different training distributions (more/
            # harder regimes raise RMSE even as OOS ranking IMPROVES — verified
            # 2026-06-16: multi-regime beat 2023-only on 6/6 reversal dates).
            # Gate on the IC floor (already passed above), not the RMSE ratio.
            logger.warning(
                "train_start change {} -> {}; promoting on IC floor "
                "(RMSE incomparable across training windows)",
                incumbent_train_start(models_dir, asof_date, DEFAULT_TARGET),
                _train_start.isoformat(),
            )
            promote = True
        else:
            promote = should_promote(cv_rmse, incumbent)
        out_dir = models_dir if promote else (models_dir / "rejected")

        pkl_path, json_path = save_model(
            model,
            hyperparams=hyperparams,
            feature_names=list(X.columns),
            train_end_date=asof_date,
            train_rows=len(X),
            train_rmse=train_rmse,
            cv_rmse=cv_rmse,
            cv_ic=cv_ic,
            code_commit=_current_git_sha(),
            training_duration_seconds=train_duration,
            output_dir=out_dir,
            target=DEFAULT_TARGET,
            promoted=promote,
            objective=objective,
            label_type=_new_label_type,
            train_start=_train_start,
        )
        model_id = pkl_path.stem
        # The IC-floor and transition cases log their own reason above; only
        # log the RMSE-comparison reason when there's an incumbent to compare.
        if not promote and incumbent is not None and passes_ic_floor(cv_ic, cv_ran=not no_cv):
            logger.warning(
                "RETRAIN NOT PROMOTED: new CV RMSE {:.5f} exceeds incumbent "
                "{:.5f} * {:.2f}. Quarantined to {}; live keeps the incumbent.",
                cv_rmse, incumbent, GATE_MAX_CV_RMSE_RATIO, out_dir,
            )
        # Sentinel write: INSIDE writer_lock (after save_model) so the ordering
        # contract is satisfied: sentinel writes are serialized by the lock.
        write_sentinel(
            label="com.sma.model.retrain.weekly",
            asof=asof_date,
            payload={
                "label": "com.sma.model.retrain.weekly",
                "asof": asof_date.isoformat(),
                "completed_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                "model_id": model_id,
                "train_end_date": asof_date.isoformat(),
                "cv_rmse": cv_rmse,
                "cv_ic": cv_ic,
                "train_rmse": train_rmse,
                "incumbent_cv_rmse": incumbent,
                "promoted": promote,
                "training_rows": len(X),
                "objective": objective,
                # How many boosters the promoted artifact holds, so the
                # sentinel trail shows when the ensemble started serving.
                "ensemble_seeds": ensemble_size(model),
            },
        )

        # Retention (2026-08-24 audit): models_artifacts/ had no pruning and
        # grew unboundedly. Best-effort — never fails the retrain job over a
        # housekeeping error, mirroring sma.backup.runner's pattern for its
        # own end-of-job cleanup steps.
        try:
            pruned = prune_old_artifacts(models_dir, asof_date, DEFAULT_TARGET)
            if pruned:
                logger.info("retrain: pruned {} old artifact(s): {}", len(pruned), pruned)
        except Exception as exc:
            logger.exception("retrain: artifact pruning FAILED: {}", exc)

    verb = "Saved+deployed" if promote else "Saved but QUARANTINED (worse than incumbent)"
    click.echo(
        f"{verb} {pkl_path.name} ({len(X)} rows, CV RMSE {cv_rmse:.5f}, "
        f"in-sample {train_rmse:.5f})"
    )


# ============================================================
# Task 16: predict
# ============================================================
@cli.command()
@click.option(
    "--asof",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    default=None,
    help="Predict for ASOF (ISO date). Defaults to today (ET) when omitted, "
         "so the launchd plist can invoke this command without args.",
)
@click.option("--db-path", type=click.Path(path_type=Path), default=DEFAULT_DB_PATH)
@click.option("--models-dir", type=click.Path(path_type=Path), default=DEFAULT_MODELS_DIR)
@click.option("--universe-path", type=click.Path(path_type=Path), default=DEFAULT_UNIVERSE_PATH)
@click.option(
    "--preflight/--no-preflight",
    "preflight",
    default=True,
    help="Gate on the ingest sentinel for ASOF (the scheduled path). Use "
         "--no-preflight for backfills/eval over historical dates, which have "
         "no ingest sentinels.",
)
def predict(asof, db_path, models_dir, universe_path, preflight):
    """Predict for every ticker in the universe at ASOF using the latest eligible model."""
    if asof is None:
        from zoneinfo import ZoneInfo
        asof_date = datetime.now(ZoneInfo("America/New_York")).date()
    else:
        asof_date = asof.date()

    # Ingest gate + lineage (2026-06-09): without this, a failed ingest let
    # predict silently write predictions off STALE features with a healthy
    # sentinel — and once ingest healed, decide would trade on them. On
    # refusal we write NO sentinel, so the watchdog re-kicks predict after
    # ingest heals; on success we stamp the consumed ingest run_id for
    # decide's lineage check.
    ingest_run_id = None
    if preflight:
        try:
            ingest_run_id = _wait_for_ingest_ready(asof_date)
        except IngestHolidaySkippedError as e:
            click.echo(f"predict: skipped — {e} (market holiday)")
            return

    universe = load_universe(universe_path)

    predictor = Predictor(models_dir=models_dir, db_path=db_path, target=DEFAULT_TARGET)
    try:
        scores, model_id = predictor.predict_for_with_model_id(asof_date, universe)
    except FileNotFoundError as exc:
        click.echo(f"No model available for {asof_date}: {exc}", err=True)
        raise click.exceptions.Exit(1) from exc

    if not scores:
        click.echo(f"No predictions produced for {asof_date} (no eligible tickers).")
        return

    with writer_lock(label="predict", timeout_s=600.0):  # queue behind slow ingest (2026-06-12)
        n_written = write_predictions(
            db_path=db_path,
            asof_date=asof_date,
            target=DEFAULT_TARGET,
            model_id=model_id,
            predictions=scores,
        )
        write_sentinel(
            label="com.sma.model.predict.daily",
            asof=asof_date,
            payload={
                "label": "com.sma.model.predict.daily",
                "asof": asof_date.isoformat(),
                "completed_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                "model_id": model_id,
                "rows_written": n_written,
                "tickers": len(scores),
                "ingest_run_id": ingest_run_id,
            },
        )
    click.echo(f"Wrote {n_written} predictions for {asof_date} (model_id={model_id})")


class IngestHolidaySkippedError(Exception):
    """Raised by _wait_for_ingest_ready when ingest marked the day a market
    holiday — predict must skip quietly (no predictions, no sentinel, no page)."""


def _wait_for_ingest_ready(
    asof_date: date,
    *,
    poll_interval_s: float = 30.0,
    now_fn=None,
    sleep_fn=None,
) -> int | None:
    """Block until the ingest sentinel for `asof_date` is READY; return its run_id.

    WAIT (sentinel not yet written) polls until ingest's deadline + 15 min —
    the natural 19:30 fire racing a slow ingest. FAIL (unwaived blocking
    quality failures) or a deadline that has already passed (historical asof
    with the gate left on) refuses immediately: notify + exit 1 with NO
    sentinel written, so the watchdog re-kicks predict once ingest heals.
    """
    from sma import schedule as sched
    from sma.readiness import ReadinessState, live_readiness

    _now = now_fn or (lambda: datetime.now(sched.NY_TZ))
    _sleep = sleep_fn or time.sleep
    label = "com.sma.ingest.daily"
    waivers = sched.get("com.sma.model.predict.daily").waivers
    try:
        wait_until = sched.deadline(label, asof=asof_date) + timedelta(minutes=15)
    except ValueError:
        # asof is not a scheduled ingest day (weekend/holiday backfill with the
        # gate left on): evaluate once, never poll.
        wait_until = _now()

    while True:
        sentinel = read_sentinel(label=label, asof=asof_date)
        r = live_readiness(label=label, sentinel=sentinel, waivers=waivers)
        if sentinel is not None and (
            sentinel.get("holiday_skipped")
            or (r.state == ReadinessState.READY and sentinel.get("run_id") is None)
        ):
            # Holiday-skip sentinel: passed=True, run_id=None. There is no new
            # data and no session — stale-feature predictions would only feed
            # a decide that must itself skip today.
            raise IngestHolidaySkippedError(
                f"ingest marked {asof_date.isoformat()} as a market holiday"
            )
        if r.state == ReadinessState.READY:
            return (sentinel or {}).get("run_id")
        if r.state == ReadinessState.FAIL or _now() >= wait_until:
            notify_failure(
                title="SMA predict BLOCKED — no fresh predictions",
                message=(
                    f"{asof_date.isoformat()}: ingest not ready ({r.explanation}). "
                    "Predictions NOT written; stale-feature predictions would "
                    "poison decide."
                ),
            )
            raise click.exceptions.Exit(1)
        _sleep(poll_interval_s)


# ============================================================
# Task 17: backfill-predictions
# ============================================================
@cli.command(name="backfill-predictions")
@click.option("--start", type=click.DateTime(formats=["%Y-%m-%d"]), required=True)
@click.option("--end", type=click.DateTime(formats=["%Y-%m-%d"]), required=True)
@click.option("--db-path", type=click.Path(path_type=Path), default=DEFAULT_DB_PATH)
@click.option("--models-dir", type=click.Path(path_type=Path), default=DEFAULT_MODELS_DIR)
@click.option("--universe-path", type=click.Path(path_type=Path), default=DEFAULT_UNIVERSE_PATH)
@click.pass_context
def backfill_predictions(ctx, start, end, db_path, models_dir, universe_path):
    """Walk every Friday in [START, END]: train (or load) a model with data up to that
    Friday, then for each weekday from that Friday through the next Friday, predict
    the universe and write to predictions table.
    """
    start_date = start.date()
    end_date = end.date()
    if start_date > end_date:
        click.echo("--start must be before --end", err=True)
        raise click.exceptions.Exit(1)

    # Walk Fridays in range. Find the first Friday on or before start_date.
    fridays: list[date] = []
    d = start_date
    while d.weekday() != 4:  # Monday=0 ... Friday=4
        d -= timedelta(days=1)
        if d < start_date - timedelta(days=7):
            break  # safety
    if d.weekday() == 4:
        fridays.append(d)
    while fridays and fridays[-1] + timedelta(days=7) <= end_date:
        fridays.append(fridays[-1] + timedelta(days=7))

    logger.info(
        f"Backfill: {len(fridays)} Friday-rooted model windows from {start_date} to {end_date}"
    )

    n_models_trained = 0
    n_models_loaded = 0
    n_predictions = 0

    for friday in fridays:
        # Train (or skip if a model for this Friday already exists).
        try:
            existing = latest_model_for_date(models_dir, friday, DEFAULT_TARGET)
            existing_train_end_str = existing.stem.split("_")[-2]
            if existing_train_end_str == friday.isoformat():
                logger.info(f"Model for {friday} already exists ({existing.name}); skipping train.")
                n_models_loaded += 1
            else:
                ctx.invoke(
                    train,
                    asof=_to_datetime(friday),
                    no_cv=True,
                    db_path=db_path,
                    models_dir=models_dir,
                    universe_path=universe_path,
                )
                n_models_trained += 1
        except FileNotFoundError:
            ctx.invoke(
                train,
                asof=_to_datetime(friday),
                no_cv=True,
                db_path=db_path,
                models_dir=models_dir,
                universe_path=universe_path,
            )
            n_models_trained += 1

        # Predict for each weekday from this Friday until the day before the next Friday
        # (or until end_date, whichever is sooner).
        next_friday = friday + timedelta(days=7)
        d = friday
        while d < next_friday and d <= end_date:
            if d.weekday() < 5 and d >= start_date:
                ctx.invoke(
                    predict,
                    asof=_to_datetime(d),
                    db_path=db_path,
                    models_dir=models_dir,
                    universe_path=universe_path,
                    # Historical walk: no ingest sentinels exist for the past.
                    preflight=False,
                )
                n_predictions += 1
            d += timedelta(days=1)

    click.echo(
        f"Backfill done: {n_models_trained} models trained, "
        f"{n_models_loaded} models reused, {n_predictions} prediction days written."
    )


def _to_datetime(d: date):
    """Click DateTime expects a datetime. Convert."""
    return datetime(d.year, d.month, d.day)


if __name__ == "__main__":
    cli()
