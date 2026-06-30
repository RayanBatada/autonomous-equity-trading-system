"""CLI entrypoint for Phase 4 LLM research agents."""

from datetime import UTC, datetime, timedelta
from datetime import date as date_cls
from pathlib import Path

import click
from anthropic import Anthropic
from loguru import logger

from sma.agents.analyst import Analyst
from sma.agents.base import FilingRow, NewsRow, PriceContext
from sma.agents.cost_tracker import CostTracker
from sma.agents.pipeline import CachedThesisFallback, ThesisContext, ThesisPipeline
from sma.agents.researcher import Researcher
from sma.agents.strategist import Strategist
from sma.agents.triggers import TriggerConfig, tickers_needing_refresh
from sma.config import load_settings
from sma.ingest.store import Store
from sma.ingest.universe import load_universe
from sma.locks import writer_lock
from sma.sectors import sector_for
from sma.sentinels import write_sentinel


def _now_et() -> "datetime":
    from zoneinfo import ZoneInfo

    return datetime.now(ZoneInfo("America/New_York"))


def _build_pipeline(settings, store: Store) -> ThesisPipeline:
    api_key = settings.secrets.anthropic_api_key
    if not api_key:
        raise click.ClickException("ANTHROPIC_API_KEY not set in .env; Phase 4 requires it")
    # max_retries=3 + timeout=90s: enough headroom for one Anthropic
    # Retry-After (typically 30-60s) without letting a single stalled
    # ticker-day eat 30+ minutes (max_retries=6 × 60s + 600s default
    # timeout did exactly that during the 2026-04-27 val run). If a
    # request can't recover in 4 attempts, fail fast — the prefill is
    # idempotent + resumable, so re-running picks up the failures cheaply.
    client = Anthropic(api_key=api_key, max_retries=3, timeout=90.0)
    ct = CostTracker(
        client=client,
        store=store,
        daily_budget_usd=settings.agents.daily_budget_usd,
        warn_threshold_pct=settings.agents.warn_threshold_pct,
    )
    model = settings.agents.haiku_model_id
    return ThesisPipeline(
        researcher=Researcher(cost_tracker=ct, model=model),
        analyst=Analyst(cost_tracker=ct, model=model),
        strategist=Strategist(cost_tracker=ct, model=model),
        store=store,
    )


def _build_context(store: Store, ticker: str, asof: date_cls) -> ThesisContext:
    """Assemble ThesisContext from DuckDB queries (news, filings, prices, earnings, predictions).

    Both news and filings queries are bounded above by `asof + 1 day` (exclusive)
    so historical prefills cannot see articles or filings published after the
    simulated decision date. Without the upper bound, asof=2025-07-04 would
    pull news through whatever the table currently contains (potentially
    months in the future), causing look-ahead leakage in val/test theses.
    """
    news_rows = [
        NewsRow(published_at=r[0], source=r[1], headline=r[2], body_excerpt=r[3])
        for r in store.conn.execute(
            "SELECT published_at, source, headline, body_excerpt FROM news "
            "WHERE ticker = ? "
            "  AND published_at >= ? - INTERVAL '7 days' "
            "  AND published_at <  ? + INTERVAL '1 day' "
            "ORDER BY published_at DESC LIMIT 30",
            [ticker, asof, asof],
        ).fetchall()
    ]
    filing_rows = [
        FilingRow(filed_at=r[0], filing_type=r[1], url=r[2])
        for r in store.conn.execute(
            "SELECT filed_at, filing_type, url FROM filings "
            "WHERE ticker = ? "
            "  AND filed_at >= ? - INTERVAL '90 days' "
            "  AND filed_at <  ? + INTERVAL '1 day' "
            "ORDER BY filed_at DESC LIMIT 10",
            [ticker, asof, asof],
        ).fetchall()
    ]

    pc = PriceContext()
    px = store.conn.execute(
        "SELECT date, adj_close FROM prices "
        "WHERE ticker = ? AND date <= ? AND date >= ? - INTERVAL '30 days' "
        "AND adj_close IS NOT NULL ORDER BY date",
        [ticker, asof, asof],
    ).fetchall()
    if len(px) >= 2:
        first, last = px[0][1], px[-1][1]
        pc.total_return_30d = (last - first) / first if first else None
        peak = first
        max_dd = 0.0
        for _, c in px:
            peak = max(peak, c)
            dd = (c - peak) / peak if peak else 0.0
            max_dd = min(max_dd, dd)
        pc.max_drawdown_30d = max_dd

    next_earn = store.conn.execute(
        "SELECT MIN(report_date) FROM earnings "
        "WHERE ticker = ? AND report_date BETWEEN ? AND ? + INTERVAL '14 days'",
        [ticker, asof, asof],
    ).fetchone()
    next_earnings_date = next_earn[0] if (next_earn and next_earn[0]) else None

    pred = store.conn.execute(
        "SELECT predicted_value FROM predictions "
        "WHERE ticker = ? AND asof_date = ? "
        "ORDER BY model_id DESC LIMIT 1",
        [ticker, asof],
    ).fetchone()
    quant_pred = pred[0] if pred else None

    return ThesisContext(
        ticker=ticker,
        asof_date=asof,
        news_rows=news_rows,
        filing_rows=filing_rows,
        price_ctx=pc,
        sector=sector_for(ticker),
        next_earnings_date=next_earnings_date,
        held_shares=0,  # Phase 5 will populate from positions table
        quant_predicted_return=quant_pred,
        quant_universe_rank=None,
        portfolio_sector_exposure_pct=0.0,
    )


def _thesis_exists(store: Store, ticker: str, asof: date_cls) -> bool:
    """True if a thesis already exists for (ticker, asof) — used to skip a
    same-day rerun so we don't re-spend budget or insert a duplicate thesis."""
    row = store.conn.execute(
        "SELECT 1 FROM theses WHERE ticker = ? AND asof_date = ? LIMIT 1",
        [ticker, asof],
    ).fetchone()
    return row is not None


@click.group()
def cli():
    """Phase 4 LLM research agents."""


@cli.command()
@click.option("--asof-date", default=None, help="ISO date (default: today)")
@click.option("--config", default="config.yaml", type=click.Path(exists=True))
@click.option("--universe", default="src/sma/universe.yaml", type=click.Path(exists=True))
@click.option("--db", default="data/sma.duckdb")
@click.option(
    "--force-full", is_flag=True, default=False, help="Run full universe regardless of weekday"
)
def run(asof_date, config, universe, db, force_full):
    """Nightly thesis pipeline: full universe on Friday, triggered tickers on Mon-Thu."""
    settings = load_settings(config_path=Path(config))
    universe_list = load_universe(Path(universe))
    asof = date_cls.fromisoformat(asof_date) if asof_date else date_cls.today()

    Path(db).parent.mkdir(parents=True, exist_ok=True)

    tickers_processed = 0
    tickers_skipped_no_trigger = 0
    tickers_skipped_budget = 0
    tickers_skipped_existing = 0
    tickers_cached_fallback = 0
    tickers_failed = 0

    with writer_lock(label="agents", timeout_s=600.0):  # queue behind slow predict (2026-06-12)
        store = Store(path=db).connect()
        try:
            pipeline = _build_pipeline(settings, store)
            run_id = store.allocate_run_id()

            is_friday = asof.weekday() == 4
            if is_friday or force_full:
                tickers = sorted(set(universe_list))
                click.echo(f"Friday full refresh: {len(tickers)} tickers")
            else:
                cfg = TriggerConfig(
                    price_move_pct=settings.agents.triggers.price_move_pct,
                    material_filing_types=settings.agents.triggers.material_filing_types,
                )
                tickers = tickers_needing_refresh(store, asof, universe_list, cfg)
                tickers_skipped_no_trigger = len(universe_list) - len(tickers)
                click.echo(f"Mon-Thu triggered refresh: {len(tickers)} tickers ({tickers})")

            # Deadline budget (2026-06-12): the first 267-name night had 81
            # new tickers with no theses; agents ground ~2.5 min/ticker for
            # HOURS holding the writer lock — decide died at 20:00 on lock
            # timeout. Theses are ADVISORY: past the cutoff (decide fires at
            # 20:00; leave 2 min to release the lock) stop STARTING tickers,
            # count the rest as deadline-skipped, write the sentinel, exit 0.
            _cutoff_reached = False
            tickers_skipped_deadline = 0
            for t in tickers:
                now_et = _now_et()
                # Budget applies only to the SCHEDULED same-day run —
                # historical/manual asofs are unbudgeted (ingest doctrine).
                if _cutoff_reached or (
                    now_et.date() == asof
                    and (now_et.hour, now_et.minute) >= (19, 58)
                ):
                    if not _cutoff_reached:
                        logger.warning(
                            "agents deadline budget: cutoff reached at {}; "
                            "skipping remaining tickers", now_et.strftime("%H:%M")
                        )
                    _cutoff_reached = True
                    tickers_skipped_deadline += 1
                    continue
                try:
                    if _thesis_exists(store, t, asof):
                        # Same-day rerun: a thesis already exists. Skip — don't
                        # re-spend LLM budget or insert a duplicate thesis row.
                        tickers_skipped_existing += 1
                        logger.info("{}: thesis already exists for {}; skipping", t, asof)
                        continue
                    ctx = _build_context(store, t, asof)
                    out = pipeline.run(ctx, run_id=run_id)
                    if out is None:
                        tickers_skipped_budget += 1
                    elif isinstance(out, CachedThesisFallback):
                        tickers_cached_fallback += 1
                    else:
                        tickers_processed += 1
                        logger.info("{}: conviction={} score={:.2f}", t, out.conviction, out.score)
                except Exception as e:
                    tickers_failed += 1
                    logger.exception("pipeline failed for {}: {}", t, e)

            total_cost = (
                store.conn.execute(
                    "SELECT COALESCE(SUM(est_cost_usd), 0.0) FROM agent_calls WHERE run_id = ?",
                    [run_id],
                ).fetchone()[0]
                or 0.0
            )
            click.echo(f"Run {run_id} complete. Total cost: ${total_cost:.4f}")
        finally:
            store.close()

        # Sentinel write: inside writer_lock (after store closed) so that the
        # ordering contract is satisfied per the sentinels module docstring.
        # quality.passed is required so live_readiness() can gate decide on
        # this sentinel. budget_exhausted is surfaced as a blocking_failure
        # so the budget_exhausted waiver in decide's JobSchedule is live.
        _budget_exhausted = (
            tickers_skipped_budget + tickers_cached_fallback > 0
            and tickers_processed == 0
        )
        # A run that had work to do but produced ZERO theses because every
        # ticker errored is a systemic failure (LLM down, code bug) and must NOT
        # green-light decide via live_readiness(). Partial failures degrade
        # gracefully (decide falls back to model predictions + prior theses), so
        # they're recorded in tickers_failed but don't block.
        _all_failed = tickers_failed > 0 and tickers_processed == 0
        # A run where MOST attempted tickers errored is a systemic problem
        # (LLM flaking, quota, code bug) even when a few succeeded; it must
        # not read as a healthy advisory. Agents stays an ADVISORY dep, so
        # this never blocks trading — it surfaces honestly in readiness +
        # the dashboard instead of masquerading as passed=True (audit MED).
        _majority_failed = (
            not _all_failed and tickers_failed > tickers_processed
        )
        _blocking: list[str] = []
        if _budget_exhausted:
            _blocking.append("budget_exhausted")
        if _all_failed:
            _blocking.append("all_tickers_failed")
        if _majority_failed:
            _blocking.append("majority_tickers_failed")
        write_sentinel(
            label="com.sma.agents.daily",
            asof=asof,
            payload={
                "label": "com.sma.agents.daily",
                "asof": asof.isoformat(),
                "completed_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                "run_id": run_id,
                "tickers_processed": tickers_processed,
                "tickers_skipped_no_trigger": tickers_skipped_no_trigger,
                "tickers_skipped_budget": tickers_skipped_budget,
                "tickers_skipped_existing": tickers_skipped_existing,
                "tickers_skipped_deadline": tickers_skipped_deadline,
                "tickers_cached_fallback": tickers_cached_fallback,
                "tickers_failed": tickers_failed,
                "budget_spent_usd": total_cost,
                "force_full": force_full,
                "quality": {
                    "passed": not _blocking,
                    "blocking_failures": _blocking,
                },
            },
        )


@cli.command()
@click.option("--ticker", required=True)
@click.option("--asof", required=True, help="ISO date")
@click.option("--config", default="config.yaml", type=click.Path(exists=True))
@click.option("--db", default="data/sma.duckdb")
def thesis(ticker, asof, config, db):
    """Run the full pipeline for one ticker and print every intermediate output."""
    settings = load_settings(config_path=Path(config))
    asof_d = date_cls.fromisoformat(asof)

    Path(db).parent.mkdir(parents=True, exist_ok=True)
    with writer_lock(label="agents-thesis"):
        store = Store(path=db).connect()
        try:
            pipeline = _build_pipeline(settings, store)
            run_id = store.allocate_run_id()
            ctx = _build_context(store, ticker, asof_d)
            click.echo(
                f"=== Researcher input: {len(ctx.news_rows)} news rows, "
                f"{len(ctx.filing_rows)} filings ==="
            )
            out = pipeline.run(ctx, run_id=run_id)
            if out is None:
                click.echo("No thesis produced (budget exhausted, no cached fallback).")
                return

            row = store.conn.execute(
                """
                SELECT news_summary, key_developments, notable_filings,
                       bull_case, bear_case, asymmetric_risks, catalyst_window,
                       conviction, score, flags, action_hint, reasoning
                FROM theses
                WHERE ticker = ? AND asof_date = ? AND run_id = ?
                """,
                [ticker, asof_d, run_id],
            ).fetchone()
            if row is None:
                click.echo(
                    "Pipeline returned but thesis row not found "
                    "(likely cached fallback; nothing persisted this run)."
                )
                return

            click.echo("=== Researcher ===")
            click.echo(f"  news_summary: {row[0]}")
            click.echo(f"  key_developments: {row[1]}")
            click.echo(f"  notable_filings: {row[2]}")
            click.echo("=== Analyst ===")
            click.echo(f"  bull_case: {row[3]}")
            click.echo(f"  bear_case: {row[4]}")
            click.echo(f"  asymmetric_risks: {row[5]}")
            click.echo(f"  catalyst_window: {row[6]}")
            click.echo("=== Strategist ===")
            click.echo(f"  conviction: {row[7]}  score: {row[8]:.2f}")
            click.echo(f"  flags: {row[9]}")
            click.echo(f"  action_hint: {row[10]}")
            click.echo(f"  reasoning: {row[11]}")
        finally:
            store.close()


# Per-ticker-day cost estimate (Haiku 4.5, all 3 agents, NO batch discount).
# Calibrated against Smoke A (2026-04-27): 39 successful 3-stage theses for
# $0.43 → ~$0.011/thesis. Slight padding lets dry-run estimates lean
# conservative against the --max-cost-usd guardrail.
PREFILL_USD_PER_TICKER_DAY = 0.012


def _enqueue_prefill_tickerdays(
    store: Store,
    universe: list[str],
    start: date_cls,
    end: date_cls,
    trigger_cfg: TriggerConfig,
) -> list[tuple[str, date_cls]]:
    """Walk weekdays from start..end, picking tickers per cadence rules.

    - Friday: full universe.
    - Mon-Thu: tickers_needing_refresh result.
    - Weekends: skipped.
    Returns list of (ticker, asof_date) pairs in chronological order.
    """
    queue: list[tuple[str, date_cls]] = []
    cur = start
    while cur <= end:
        wd = cur.weekday()
        if wd <= 3:  # Mon-Thu
            tickers = tickers_needing_refresh(store, cur, universe, trigger_cfg)
        elif wd == 4:  # Friday
            tickers = sorted(set(universe))
        else:  # weekend
            tickers = []
        for t in tickers:
            queue.append((t, cur))
        cur = cur + timedelta(days=1)
    return queue


def _filter_already_persisted(store, queue):
    """Drop (ticker, asof) pairs that already have a thesis in the table."""
    if not queue:
        return queue
    earliest = min(q[1] for q in queue)
    latest = max(q[1] for q in queue)
    rows = store.conn.execute(
        "SELECT DISTINCT ticker, asof_date FROM theses WHERE asof_date BETWEEN ? AND ?",
        [earliest, latest],
    ).fetchall()
    persisted = {(r[0], r[1]) for r in rows}
    return [(t, d) for (t, d) in queue if (t, d) not in persisted]


@cli.command()
@click.option("--start", required=True, help="ISO date for window start")
@click.option("--end", required=True, help="ISO date for window end")
@click.option(
    "--dry-run", is_flag=True, default=False, help="Print estimate without calling Anthropic"
)
@click.option(
    "--max-cost-usd",
    default=30.0,
    type=float,
    help="Pre-flight gate: refuse to start if dry-run estimate "
    "exceeds this (default $30). Separate from the running "
    "cost ceiling; that is --daily-budget-usd / config.",
)
@click.option(
    "--daily-budget-usd",
    default=None,
    type=float,
    help="Override settings.agents.daily_budget_usd for this run "
    "only. Use to raise the running ceiling above the "
    "config default (typically $0.50) without editing "
    "config.yaml.",
)
@click.option(
    "--max-workers",
    default=1,
    type=int,
    help="Number of parallel ticker-day workers (default: 1, "
    "sequential). Each worker runs the full 3-stage pipeline "
    "on one ticker-day; 5 is a good balance for Haiku "
    "rate limits and DuckDB write contention.",
)
@click.option("--config", default="config.yaml", type=click.Path(exists=True))
@click.option("--universe", default="src/sma/universe.yaml", type=click.Path(exists=True))
@click.option("--db", default="data/sma.duckdb")
def prefill(start, end, dry_run, max_cost_usd, daily_budget_usd, max_workers, config, universe, db):
    """Pre-fill the theses table for the val+test backtest windows.

    Locking strategy: prefill acquires writer_lock per ticker-day (option b).
    Each per-ticker thesis acquisition cycle is short (one 3-stage LLM pipeline),
    so the lock is held briefly and released before the next ticker. This avoids
    blocking the rest of the system (ingest, predict, decide) for the entire
    multi-hour prefill run. Trade-off: higher lock churn vs. blocking ingest for
    hours (option a: one lock for the entire run). Option b is preferred for
    prefill since it runs rarely (once per val window) and the per-ticker duration
    is bounded by the Anthropic API timeout (90s max).

    For multi-worker runs (--max-workers > 1): each worker acquires its own
    writer_lock for each ticker-day. Writers serialize through the flock even
    under concurrent execution, so DuckDB write contention is avoided without
    any additional synchronization.
    """
    settings = load_settings(config_path=Path(config))
    if daily_budget_usd is not None:
        settings.agents.daily_budget_usd = daily_budget_usd
    universe_list = load_universe(Path(universe))
    start_d = date_cls.fromisoformat(start)
    end_d = date_cls.fromisoformat(end)
    if start_d > end_d:
        raise click.ClickException("--start must be <= --end")

    Path(db).parent.mkdir(parents=True, exist_ok=True)

    # Setup phase: brief writer_lock to ensure schema exists and allocate run_id.
    # Also runs planning queries (trigger detection + already-persisted filter)
    # using the same writable connection, then releases the lock. The planning
    # queries are reads-only in semantics, but DuckDB requires schema migrations
    # to happen on the first writable connect, and a non-existent DB cannot be
    # opened read-only (duckdb.IOException). A single brief lock for setup is
    # acceptable since it completes in milliseconds.
    with writer_lock(label="prefill"):
        store = Store(path=db).connect()
        try:
            trigger_cfg = TriggerConfig(
                price_move_pct=settings.agents.triggers.price_move_pct,
                material_filing_types=settings.agents.triggers.material_filing_types,
            )
            queue = _enqueue_prefill_tickerdays(store, universe_list, start_d, end_d, trigger_cfg)
            original_count = len(queue)
            queue = _filter_already_persisted(store, queue)
            skipped = original_count - len(queue)
            run_id_holder: list[int] = []
            # Only allocate run_id if we'll actually run (avoid unused run_id rows).
            # We'll do a conditional check after cost gate; just get run_id here
            # if we plan to run, else we close without it.
            # Actually: allocate unconditionally; the dry_run + empty-queue
            # checks happen after we release. If they abort, the run_id row
            # is simply unreferenced (harmless).
            run_id_holder.append(store.allocate_run_id())
        finally:
            store.close()

    run_id = run_id_holder[0]

    est_cost = len(queue) * PREFILL_USD_PER_TICKER_DAY
    click.echo(f"Pre-fill plan: {len(queue)} ticker-days (skipping {skipped} already persisted)")
    click.echo(
        f"Estimated cost: ${est_cost:.2f}  (at ${PREFILL_USD_PER_TICKER_DAY:.4f}/ticker-day)"
    )

    if est_cost > max_cost_usd:
        raise click.ClickException(
            f"Estimated cost ${est_cost:.2f} exceeds --max-cost-usd "
            f"${max_cost_usd:.2f}. Adjust the window or raise the ceiling."
        )

    if dry_run:
        click.echo("(dry-run; no calls made)")
        return

    if not queue:
        click.echo("Nothing to pre-fill.")
        return

    completed = 0

    if max_workers <= 1:
        for ticker, asof in queue:
            try:
                # Each ticker-day: hold writer_lock for the full 3-stage LLM
                # pipeline + DB write (option b: per-ticker lock). Avoids
                # blocking ingest/predict/decide for the entire multi-hour
                # prefill run (option a: one lock). Tradeoff: higher lock churn
                # but each hold is bounded by the Anthropic API timeout (90s max
                # per ticker-day). Prefill runs rarely (once per val window).
                with writer_lock(label="prefill"):
                    store = Store(path=db).connect()
                    try:
                        pipeline = _build_pipeline(settings, store)
                        ctx = _build_context(store, ticker, asof)
                        pipeline.run(ctx, run_id=run_id)
                    finally:
                        store.close()
                completed += 1
                if completed % 25 == 0:
                    click.echo(f"  ... {completed}/{len(queue)} ticker-days complete")
            except Exception as e:
                logger.exception(
                    "prefill failed for {} {}: {}",
                    ticker,
                    asof,
                    e,
                )
    else:
        # DuckDB connections are NOT safe to share across threads. Each worker
        # opens its own Store + Pipeline inside writer_lock per ticker-day
        # (option b). The flock serializes writes; concurrent threads queue at
        # the flock rather than at DuckDB.
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _run_one(ticker_asof):
            ticker, asof = ticker_asof
            try:
                with writer_lock(label="prefill"):
                    s = Store(path=db).connect()
                    try:
                        p = _build_pipeline(settings, s)
                        ctx = _build_context(s, ticker, asof)
                        p.run(ctx, run_id=run_id)
                    finally:
                        s.close()
                return True
            except Exception as e:
                logger.exception(
                    "prefill failed for {} {}: {}",
                    ticker,
                    asof,
                    e,
                )
                return False

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(_run_one, pair) for pair in queue]
            for done_count, fut in enumerate(as_completed(futures), 1):
                if fut.result():
                    completed += 1
                if done_count % 25 == 0:
                    click.echo(
                        f"  ... {done_count}/{len(queue)} ticker-days "
                        f"processed ({completed} succeeded)"
                    )

    # Final cost summary: brief lock to open writable store + read cost.
    with writer_lock(label="prefill"):
        store = Store(path=db).connect()
        try:
            actual = (
                store.conn.execute(
                    "SELECT COALESCE(SUM(est_cost_usd), 0.0) FROM agent_calls WHERE run_id = ?",
                    [run_id],
                ).fetchone()[0]
                or 0.0
            )
        finally:
            store.close()
    click.echo(
        f"Pre-fill complete: {completed}/{len(queue)} ticker-days. Total cost: ${actual:.4f}"
    )


if __name__ == "__main__":
    cli()
