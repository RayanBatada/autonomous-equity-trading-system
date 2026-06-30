"""CLI entry point. Run as `python -m sma.ingest run [...]`."""

from copy import deepcopy
from datetime import date as date_cls
from datetime import timedelta
from pathlib import Path

import click
from loguru import logger

from sma.config import load_settings
from sma.ingest.earnings_backfill import (
    backfill_earnings,
    backfill_earnings_yfinance,
)
from sma.ingest.notify import notify_failure
from sma.ingest.ratelimit import TokenBucket
from sma.ingest.runner import IngestRunner
from sma.ingest.runner import run as ingest_run
from sma.ingest.sources.alpaca_news import AlpacaNewsSource
from sma.ingest.sources.alpaca_prices import AlpacaPricesSource
from sma.ingest.sources.edgar_filings import EdgarFilingsSource
from sma.ingest.sources.finnhub_fundamentals import FinnhubFundamentalsSource
from sma.ingest.sources.finnhub_news import FinnhubNewsSource
from sma.ingest.sources.finnhub_sentiment import FinnhubSentimentSource
from sma.ingest.sources.newsapi import NewsAPISource
from sma.ingest.sources.yfinance_prices import YFinancePricesSource
from sma.ingest.store import Store
from sma.ingest.universe import load_universe
from sma.locks import writer_lock
from sma.schedule import get as _get_job
from sma.sentinels import ingest_succeeded_today

NEWS_SOURCE_NAMES = frozenset({"finnhub_news", "alpaca_news", "newsapi"})


def _ingest_quality_blocking(report) -> list[str]:
    """Blocking quality failures for the ingest job, applying its schedule waivers.

    Extracted + regression-tested: the inline version imported a NON-EXISTENT
    `get_job` from sma.schedule and crashed EVERY ingest run with ImportError at
    the exit-code step (after prices/sentinel were written) — the recurring
    `ingest exit 1`, which also defeated the 'exit non-zero only on blocking'
    intent. The real export is `get`.
    """
    waivers = frozenset(_get_job("com.sma.ingest.daily").waivers)
    return report.blocking_failures(waivers)


def _build_source(
    name: str,
    settings,
    finnhub_limiter: TokenBucket | None = None,
    lookback_days_override: int | None = None,
) -> object:
    s = settings.secrets
    cfg = settings.ingest
    if name == "yfinance":
        return YFinancePricesSource(lookback_days=cfg.default_lookback_days)
    if name == "alpaca":
        return AlpacaPricesSource(
            api_key=s.alpaca_api_key,
            api_secret=s.alpaca_api_secret,
            base_url=s.alpaca_base_url,
            lookback_days=cfg.default_lookback_days,
        )
    if name == "finnhub_news":
        kwargs = {"api_key": s.finnhub_api_key, "rate_limiter": finnhub_limiter}
        if lookback_days_override is not None:
            kwargs["lookback_days"] = lookback_days_override
        return FinnhubNewsSource(**kwargs)
    if name == "alpaca_news":
        kwargs = {
            "api_key": s.alpaca_api_key,
            "api_secret": s.alpaca_api_secret,
        }
        if lookback_days_override is not None:
            kwargs["lookback_days"] = lookback_days_override
        return AlpacaNewsSource(**kwargs)
    if name == "finnhub_sentiment":
        return FinnhubSentimentSource(api_key=s.finnhub_api_key)
    if name == "finnhub_fundamentals":
        return FinnhubFundamentalsSource(api_key=s.finnhub_api_key, rate_limiter=finnhub_limiter)
    if name == "newsapi":
        kwargs = {"api_key": s.newsapi_key}
        if lookback_days_override is not None:
            kwargs["lookback_days"] = lookback_days_override
        return NewsAPISource(**kwargs)
    if name == "edgar":
        return EdgarFilingsSource(user_agent=s.edgar_user_agent)
    raise ValueError(f"Unknown source: {name}")


def _is_first_run(store: Store) -> bool:
    n = store.conn.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
    return int(n) == 0


def _is_us_market_holiday_weekday(asof: date_cls, settings) -> bool:
    """True iff `asof` is a calendar weekday but NOT a US market trading day.

    Skips weekends (Sat/Sun) — those are obvious non-trading days and the
    ingest plist doesn't fire on them anyway. The interesting case is a
    weekday that's a market holiday (Memorial Day, July 4 when Fri, etc.).

    Uses Alpaca's trading calendar. On calendar-lookup failure, returns
    False (let ingest run; worse to skip a real trading day than to ingest
    on a holiday).
    """
    if asof.weekday() >= 5:  # Sat / Sun
        return False
    try:
        from sma.live.alpaca_client import AlpacaClient
        s = settings.secrets
        if not s.alpaca_api_key or not s.alpaca_api_secret:
            return False
        alpaca = AlpacaClient.paper_from_env(
            api_key=s.alpaca_api_key, secret_key=s.alpaca_api_secret,
        )
        sessions = alpaca.sessions_between(start=asof, end=asof)
        return len(sessions) == 0
    except Exception:
        logger.warning(
            "alpaca calendar lookup failed for asof=%s; assuming trading day",
            asof, exc_info=True,
        )
        return False


def _write_holiday_skipped_sentinel(*, asof: date_cls) -> None:
    """Write an ingest sentinel marking today as a market-holiday no-op.

    Sets quality.passed=True so downstream preflight reads this as a clean
    "nothing to do today" rather than a failure. Decide's preflight will
    still block (no new prices means stale data), but the block is
    correctly attributable to "market closed" not "ingest broke".
    """
    from datetime import UTC, datetime

    from sma.sentinels import write_sentinel
    write_sentinel(
        label="com.sma.ingest.daily",
        asof=asof,
        payload={
            "label": "com.sma.ingest.daily",
            "asof": asof.isoformat(),
            "completed_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "quality": {"passed": True, "blocking_failures": [], "checks": []},
            "holiday_skipped": True,
            "run_id": None,
        },
    )


def _missing_price_dates(
    *,
    last_priced: date_cls | None,
    asof: date_cls,
    sessions: list[date_cls],
    max_back: int = 10,
) -> list[date_cls]:
    """Trading days that should be price-backfilled before ingesting `asof`.

    Self-healing catch-up (per the 6/1-6/3 freeze): when the machine is offline
    at the scheduled ingest time, those trading days' prices are never fetched
    and the pipeline stays frozen until someone intervenes. Whenever ingest next
    runs with network up, it should detect the gap and backfill it.

    Returns the trading `sessions` strictly after `last_priced` and strictly
    before `asof` (asof itself is ingested normally by the caller), oldest-first,
    capped at the most recent `max_back` to bound a cold start / long outage.
    Empty when the DB has no prices yet (don't backfill all of history).
    """
    if last_priced is None:
        return []
    gap = sorted(d for d in sessions if last_priced < d < asof)
    return gap[-max_back:] if max_back > 0 else []


def _catch_up_missing_prices(
    *, db: str, asof: date_cls, source_objs: list, universe_list: list,
    settings, max_back: int = 10,
) -> int:
    """Backfill prices for trading days missed before `asof` (best-effort).

    Self-healing recovery from the 6/1-6/3 freeze: if the machine was offline at
    the scheduled ingest time, those days' prices were never fetched. Before
    today's ingest, detect the gap (trading days after the last priced date) and
    re-ingest each, oldest-first. Each failed day logs and continues so today's
    ingest always proceeds. Returns the number of days backfilled. Caller must
    hold the writer_lock (ingest_run writes under it).
    """
    from sma.db_connect import read_only_connect
    from sma.live.alpaca_client import AlpacaClient

    try:
        con = read_only_connect(db)
        try:
            last = con.execute("SELECT MAX(date) FROM prices").fetchone()[0]
        finally:
            con.close()
    except Exception:
        return 0  # no prices table / cold DB
    if last is None:
        return 0
    try:
        s = settings.secrets
        alpaca = AlpacaClient.paper_from_env(
            api_key=s.alpaca_api_key, secret_key=s.alpaca_api_secret
        )
        sessions = alpaca.sessions_between(start=last, end=asof)
    except Exception as e:
        logger.warning("catch-up: trading-calendar lookup failed ({}); skipping backfill", e)
        return 0
    missing = _missing_price_dates(
        last_priced=last, asof=asof, sessions=sessions, max_back=max_back
    )
    done = 0
    for d in missing:
        try:
            logger.info("catch-up: backfilling missed ingest for {}", d)
            ingest_run(asof=d, db_path=db, sources=source_objs, universe=universe_list)
            done += 1
        except Exception as e:
            logger.warning("catch-up: backfill for {} failed ({}); continuing", d, e)
    if done:
        logger.info("catch-up: backfilled {} missed trading day(s) before {}", done, asof)
    return done


@click.group()
def cli():
    pass


@cli.command()
@click.option(
    "--config", default="config.yaml", type=click.Path(exists=True), help="Path to config.yaml"
)
@click.option(
    "--universe",
    default="src/sma/universe.yaml",
    type=click.Path(exists=True),
    help="Path to universe.yaml",
)
@click.option("--db", default="data/sma.duckdb", help="Path to DuckDB file")
@click.option("--asof-date", default=None, help="ISO date (YYYY-MM-DD); defaults to today")
@click.option(
    "--sources",
    default=None,
    help="Comma-separated subset of sources to run (default: all enabled)",
)
@click.option(
    "--tickers", default=None, help="Comma-separated subset of tickers (default: full universe)"
)
@click.option(
    "--skip-quality",
    is_flag=True,
    default=False,
    help="Skip quality-check evaluation (testing only)",
)
@click.option(
    "--lookback-days",
    default=None,
    type=int,
    help="Override auto-detected lookback (skips first-run check)",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Re-run even if today's ingest is already complete (default: skip duplicates)",
)
def run(config, universe, db, asof_date, sources, tickers, skip_quality, lookback_days, force):
    settings = load_settings(config_path=Path(config))
    universe_list = load_universe(Path(universe))
    if tickers is not None:
        wanted = {t.strip().upper() for t in tickers.split(",") if t.strip()}
        universe_list = sorted(set(universe_list) & wanted)

    enabled = settings.sources_enabled
    if sources is not None:
        wanted = {s.strip() for s in sources.split(",") if s.strip()}
        enabled = [s for s in enabled if s in wanted]

    asof = date_cls.fromisoformat(asof_date) if asof_date else date_cls.today()

    Path(db).parent.mkdir(parents=True, exist_ok=True)

    # Early-exit check: reads the sentinel file only; no store/DB required.
    if not force:
        try:
            already_done = ingest_succeeded_today(asof=asof)
        except Exception:
            already_done = False
        if already_done:
            click.echo(
                f"Ingest for {asof} already complete (all CRITICAL sources ok). "
                f"Skipping. Use --force to re-run."
            )
            return

    # US market holiday short-circuit. Without this, Mon Memorial Day (and
    # every other market-closed weekday) runs the full ingest pipeline,
    # gets no new prices from yfinance/alpaca, fails the
    # all_tickers_have_price quality check, and writes a
    # quality.passed=False sentinel that downstream preflight has to
    # waive or ignore. Cleaner: detect the holiday upfront and write a
    # quality.passed=True "holiday_skipped" sentinel so decide preflight
    # treats today as a non-trading no-op (it'll be blocked anyway since
    # there are no new prices, but for the right reason).
    _holiday = not skip_quality and _is_us_market_holiday_weekday(asof, settings)

    if lookback_days is not None:
        effective_lookback = lookback_days
        mode_msg = f"explicit lookback ({lookback_days}d)"
    else:
        # Need a quick read to determine first-run status; do it before
        # acquiring the writer_lock so we don't hold it during I/O setup.
        try:
            ro_store = Store(path=db).connect(read_only=True)
            try:
                first_run = _is_first_run(ro_store)
            finally:
                ro_store.close()
        except Exception:
            first_run = True
        effective_lookback = settings.ingest.default_lookback_days if first_run else 1
        mode_msg = "first-run (long lookback)" if first_run else "incremental (1d lookback)"

    adjusted_settings = deepcopy(settings)
    adjusted_settings.ingest.default_lookback_days = effective_lookback
    # Shared Finnhub rate limiter across news + fundamentals so we stay
    # under the 60/min free-tier cap. Capacity sized to the configured
    # requests_per_minute, refill at the equivalent per-second rate.
    finnhub_rpm = settings.ingest.rate_limits.finnhub.requests_per_minute
    finnhub_limiter = TokenBucket(
        capacity=finnhub_rpm,
        refill_rate_per_sec=finnhub_rpm / 60.0,
    )
    source_objs = [_build_source(n, adjusted_settings, finnhub_limiter) for n in enabled]
    click.echo(f"Mode: {mode_msg}")

    with writer_lock(label="ingest"):
        if _holiday:
            # Write holiday sentinel INSIDE the writer_lock so the ordering
            # contract is satisfied: sentinel writes must be serialized by the
            # lock (per sma.sentinels module docstring) to prevent a race
            # between a holiday-skip write and a concurrent recovery ingest.
            _write_holiday_skipped_sentinel(asof=asof)
            click.echo(f"Ingest for {asof} skipped: US market holiday (no trading session).")
            return
        if skip_quality:
            # Skip-quality path: use IngestRunner directly, no sentinel.
            store = Store(path=db).connect(read_only=False)
            try:
                runner = IngestRunner(store=store, sources=source_objs, universe=universe_list)
                run_id = runner.run(asof_date=asof)
            finally:
                store.close()
            click.echo(f"Run {run_id} complete. Quality checks skipped.")
            return

        # Self-healing catch-up: backfill any trading days whose prices were
        # missed (machine offline at the scheduled ingest time) before doing
        # today's ingest. Best-effort; never blocks today's run.
        _catch_up_missing_prices(
            db=db, asof=asof, source_objs=source_objs,
            universe_list=universe_list, settings=settings,
        )

        # Normal path: ingest_run opens store, runs sources, runs quality
        # checks, writes sentinel, and closes the store -- all inside the
        # writer_lock that we now hold.
        # Deadline budget: scheduled runs must not blow past the 20:30 ET
        # deadline doing overlay work (6/10 ran 2h08m). Manual/backfill asofs
        # that aren't scheduled days run unbudgeted.
        from datetime import datetime as _dt

        from sma import schedule as _sched
        try:
            _deadline = _sched.deadline("com.sma.ingest.daily", asof=asof)
        except ValueError:
            _deadline = None
        result = ingest_run(
            asof=asof, db_path=db, sources=source_objs, universe=universe_list,
            deadline=_deadline,
            now_fn=lambda: _dt.now(tz=_sched.NY_TZ),
        )

    report = result.quality_report
    log_dir = Path("logs/quality")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{asof.isoformat()}.txt"
    log_path.write_text(report.summary())
    click.echo(report.summary())
    # Exit non-zero only on BLOCKING failures (matching the sentinel + decide
    # gate). Non-blocking overlay failures (news/earnings/theses) are recorded in
    # the summary but must not mark the whole ingest job 'failed' to launchd.
    blocking = _ingest_quality_blocking(report)
    if blocking:
        notify_failure(
            title="SMA Ingest FAILED",
            message=f"blocking quality failures for {asof}: {', '.join(blocking)}",
        )
        raise SystemExit(2)


@cli.command("backfill-news")
@click.option(
    "--config", default="config.yaml", type=click.Path(exists=True), help="Path to config.yaml"
)
@click.option(
    "--universe",
    default="src/sma/universe.yaml",
    type=click.Path(exists=True),
    help="Path to universe.yaml",
)
@click.option("--db", default="data/sma.duckdb", help="Path to DuckDB file")
@click.option("--start", required=True, help="ISO start date (YYYY-MM-DD), inclusive")
@click.option("--end", required=True, help="ISO end date (YYYY-MM-DD), inclusive")
@click.option("--batch-days", default=7, type=int, help="Chunk size in days (default: 7)")
@click.option(
    "--sources",
    default=None,
    help="Comma-separated subset of news sources (default: all enabled news sources)",
)
@click.option(
    "--tickers", default=None, help="Comma-separated subset of tickers (default: full universe)"
)
def backfill_news(config, universe, db, start, end, batch_days, sources, tickers):
    """Backfill historical news by walking a date range in chunks.

    Re-uses the daily news sources by setting their lookback to batch_days
    and calling fetch() once per chunk with the chunk's end date as asof.
    Idempotent: re-running the same range overwrites existing rows via the
    news table's hash-based PK.

    The final chunk may have a smaller lookback than batch_days so the
    backfill never reads earlier than --start.
    """
    if batch_days < 1:
        raise click.BadParameter("--batch-days must be >= 1")

    start_date = date_cls.fromisoformat(start)
    end_date = date_cls.fromisoformat(end)
    if end_date < start_date:
        raise click.BadParameter("--end must be >= --start")

    settings = load_settings(config_path=Path(config))
    universe_list = load_universe(Path(universe))
    if tickers is not None:
        wanted = {t.strip().upper() for t in tickers.split(",") if t.strip()}
        universe_list = sorted(set(universe_list) & wanted)

    enabled_news = [s for s in settings.sources_enabled if s in NEWS_SOURCE_NAMES]
    if sources is not None:
        wanted = {s.strip() for s in sources.split(",") if s.strip()}
        enabled_news = [s for s in enabled_news if s in wanted]

    if not enabled_news:
        click.echo("no news sources enabled; nothing to backfill")
        return

    Path(db).parent.mkdir(parents=True, exist_ok=True)
    with writer_lock(label="backfill-news"):
        store = Store(path=db).connect()
        try:
            finnhub_rpm = settings.ingest.rate_limits.finnhub.requests_per_minute
            finnhub_limiter = TokenBucket(
                capacity=finnhub_rpm,
                refill_rate_per_sec=finnhub_rpm / 60.0,
            )

            # Build list of (chunk_start, chunk_end, lookback_days) tuples.
            # The first chunk's end is min(start + batch_days - 1, end); each
            # later chunk starts the day after the previous chunk's end and
            # ends batch_days later, clamped to `end`. The final chunk's
            # lookback_days = (chunk_end - chunk_start), so we never look
            # earlier than --start.
            chunks: list[tuple[date_cls, date_cls, int]] = []
            cursor = start_date
            while cursor <= end_date:
                chunk_end = min(cursor + timedelta(days=batch_days - 1), end_date)
                chunk_lookback = (chunk_end - cursor).days
                chunks.append((cursor, chunk_end, chunk_lookback))
                cursor = chunk_end + timedelta(days=1)

            adjusted_settings = deepcopy(settings)

            total_rows = 0
            chunks_run = 0
            for i, (_chunk_start, chunk_end, chunk_lookback) in enumerate(chunks, 1):
                source_objs = [
                    _build_source(
                        n,
                        adjusted_settings,
                        finnhub_limiter,
                        lookback_days_override=chunk_lookback,
                    )
                    for n in enabled_news
                ]
                runner = IngestRunner(
                    store=store,
                    sources=source_objs,
                    universe=universe_list,
                )
                run_id = runner.run(asof_date=chunk_end)
                chunks_run += 1
                # Sum rows inserted for this run from the ingest_log.
                for source_name in enabled_news:
                    row = store.conn.execute(
                        "SELECT rows_inserted FROM ingest_log WHERE run_id = ? AND source = ?",
                        [run_id, source_name],
                    ).fetchone()
                    rows = int(row[0]) if row and row[0] is not None else 0
                    total_rows += rows
                    logger.info(
                        "chunk {}/{} asof={} source={} rows={}",
                        i,
                        len(chunks),
                        chunk_end,
                        source_name,
                        rows,
                    )
                    click.echo(
                        f"chunk {i}/{len(chunks)} asof={chunk_end} source={source_name} rows={rows}"
                    )

            distinct_tickers = store.conn.execute(
                "SELECT COUNT(DISTINCT ticker) FROM news"
            ).fetchone()[0]
            click.echo(
                f"backfill complete: {chunks_run} chunks, "
                f"{total_rows} rows inserted, "
                f"{int(distinct_tickers)} distinct tickers in news"
            )
        finally:
            store.close()


@cli.command("backfill-earnings")
@click.option(
    "--quarters", default=8, type=int, help="Number of historical quarters per ticker (default: 8)"
)
@click.option(
    "--provider",
    type=click.Choice(["finnhub", "yfinance", "both"]),
    default="finnhub",
    help="Earnings data source (default: finnhub). 'both' runs "
    "Finnhub first, then yfinance for tickers Finnhub missed.",
)
@click.option(
    "--config", default="config.yaml", type=click.Path(exists=True), help="Path to config.yaml"
)
@click.option(
    "--universe",
    default="src/sma/universe.yaml",
    type=click.Path(exists=True),
    help="Path to universe.yaml",
)
@click.option("--db", default="data/sma.duckdb", help="Path to DuckDB file")
@click.option("--tickers", default=None, help="Comma-separated subset; defaults to full universe")
def backfill_earnings_cmd(quarters, provider, config, universe, db, tickers):
    """Backfill historical earnings (past N quarters per ticker).

    Provider 'finnhub' uses company_earnings (default; needs an active
    Finnhub key, capped by daily quota on the free tier). Provider
    'yfinance' scrapes Yahoo Finance — no quota but flakier per-ticker.
    Provider 'both' runs Finnhub then yfinance with only_missing=True so
    yfinance only fills gaps Finnhub couldn't reach.

    Idempotent on (ticker, report_date) via INSERT OR REPLACE.
    """
    settings = load_settings(config_path=Path(config))
    universe_list = load_universe(Path(universe))
    if tickers is not None:
        wanted = {t.strip().upper() for t in tickers.split(",") if t.strip()}
        universe_list = sorted(set(universe_list) & wanted)

    Path(db).parent.mkdir(parents=True, exist_ok=True)
    with writer_lock(label="backfill-earnings"):
        store = Store(path=db).connect()
        try:
            run_id = store.allocate_run_id()
            log_source = (
                "finnhub_earnings_backfill"
                if provider == "finnhub"
                else "yfinance_earnings_backfill"
                if provider == "yfinance"
                else "earnings_backfill_both"
            )
            store.log_run_start(run_id, source=log_source)
            finnhub_rows = 0
            yf_rows = 0
            if provider in ("finnhub", "both"):
                finnhub_rpm = settings.ingest.rate_limits.finnhub.requests_per_minute
                rate_limiter = TokenBucket(
                    capacity=finnhub_rpm,
                    refill_rate_per_sec=finnhub_rpm / 60.0,
                )
                finnhub_rows = backfill_earnings(
                    api_key=settings.secrets.finnhub_api_key,
                    tickers=universe_list,
                    store=store,
                    run_id=run_id,
                    quarters=quarters,
                    rate_limiter=rate_limiter,
                )
                click.echo(f"finnhub: {finnhub_rows} rows across {len(universe_list)} tickers")
            if provider in ("yfinance", "both"):
                yf_rows = backfill_earnings_yfinance(
                    tickers=universe_list,
                    store=store,
                    run_id=run_id,
                    quarters=quarters,
                    only_missing=(provider == "both"),
                )
                click.echo(
                    f"yfinance: {yf_rows} rows "
                    f"({'gap-fill only' if provider == 'both' else 'all tickers'})"
                )
            total_rows = finnhub_rows + yf_rows
            store.log_run_end(
                run_id,
                source=log_source,
                rows_inserted=total_rows,
                status="ok",
                error=None,
            )
            click.echo(f"backfill complete: {total_rows} rows across {len(universe_list)} tickers")
        finally:
            store.close()


if __name__ == "__main__":
    cli()
