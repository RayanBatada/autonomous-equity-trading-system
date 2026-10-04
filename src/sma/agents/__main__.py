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
from sma.agents.triggers import TriggerConfig, refresh_order, tickers_needing_refresh
from sma.config import load_settings
from sma.ingest.notify import notify_failure
from sma.ingest.store import Store
from sma.ingest.universe import load_universe

# Single source of truth for "what does the ledger say we hold" — shared with the
# 09:25 pre-open guard rather than duplicating the paper_fills netting SQL.
from sma.live.preopen_guard import ledger_net_positions
from sma.locks import writer_lock
from sma.sectors import sector_for
from sma.sentinels import read_sentinel, write_sentinel


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


# Hard floor (2026-08-13): agents must NEVER start a new ticker past this ET
# time-of-day on asof's overnight-continuation morning, regardless of whether
# decide has run. decide's own late-kick window closes 03:00 ET (schedule.py:
# fire 20:00 + deadline_offset 60min + late_kick_max_hours 6h default =
# 21:00 + 6h = 03:00), so 02:30 leaves it 30 minutes of headroom to acquire
# the writer lock and run once agents finally releases it.
_AGENTS_HARD_FLOOR_HOUR_MINUTE_ET = (2, 30)

# Pre-decide cutoff (restored 2026-08-16): agents must have STOPPED starting
# new tickers by this ET time-of-day on asof itself, so the writer lock is free
# before decide fires at 20:00. Two minutes of slack for the in-flight ticker
# to finish and for the sentinel write + lock release.
_AGENTS_PRE_DECIDE_CUTOFF_HOUR_MINUTE_ET = (19, 58)


def _deadline_reached(*, now_et: datetime, asof: date_cls) -> bool:
    """True once agents must stop STARTING new tickers for the SCHEDULED live
    run — see the "Deadline budget" comment at the call site for the history.

    Historical/manual asofs stay unbudgeted: `now_et`'s date must be within
    one calendar day of `asof` — today (the normal evening run) or tomorrow
    morning (the overnight continuation of tonight's chain, since decide's
    own late-kick window runs past midnight) — for any check below to apply
    at all. Matches the deadline budget's original "SCHEDULED same-day run"
    rule (a manual/backtest asof from weeks ago is never near now_et's date,
    so days_after lands outside {0, 1}).

    The rule (2026-08-16): agents may start a same-day ticker only when
      (now_et is before 19:58 ET on asof)  OR  (decide's sentinel exists),
    with the 02:30 hard floor overriding both on the next morning.

    Why NOT the plain decide-gate that 91e0baa shipped (2026-08-13): that
    version let agents keep starting tickers for as long as decide had NOT
    run, which is exactly backwards under the lock topology. agents holds the
    single global writer_lock (sma.locks) across its whole ticker loop, and
    decide needs that same lock before it can write the sentinel the gate
    waits on — the gate waited on a value it prevented. It deadlocked two
    real paths:
      (a) a Friday full refresh from 19:45 held the lock past 20:00; decide
          timed out at 20:15 (timeout_s=900, live/__main__.py) and paged
          "bot did NOT trade";
      (b) a post-outage 01:30 watchdog pass kickstarts agents BEFORE decide
          (SCHEDULE iteration order), agents took the lock, decide died at
          01:45, and no further checkpoint existed before decide's own 03:00
          window closed.
    Yielding instead makes both recover: agents defers, decide gets the lock
    and trades, and agents catches up on a later checkpoint or the next day
    (the stale-thesis fallback already covers the gap — theses are ADVISORY).

    The `OR decide sentinel exists` half is what da322f7's ingest-defer needs:
    once decide HAS run, nothing is waiting on the lock, so a late re-kick
    (e.g. 21:00 after an ingest heal) does real work instead of tripping on
    ticker one and skipping the whole universe (the 2026-08-13 bug). Those
    theses inform tomorrow via the freshness fallback.

    Three ways to trip, cheapest first:
      - hard floor: 02:30 ET on asof's next calendar day. Independent of
        decide's sentinel — a backstop if decide itself never completes or
        never writes one, so agents cannot run forever.
      - overnight, decide still pending: any time on asof's next calendar day
        with no decide sentinel means decide is inside its late-kick window
        and needs the lock. Yield.
      - same day at/after 19:58 ET with no decide sentinel: decide fires at
        20:00 and must find the lock free.
    """
    days_after = (now_et.date() - asof).days
    if days_after not in (0, 1):
        return False  # historical/manual — unbudgeted
    if days_after == 1 and (now_et.hour, now_et.minute) >= _AGENTS_HARD_FLOOR_HOUR_MINUTE_ET:
        return True
    if read_sentinel(label="com.sma.live.decide.daily", asof=asof) is not None:
        # decide has already run for this trading day — nothing is queued
        # behind us on the writer lock, so keep building (tomorrow's overlay).
        return False
    # decide has NOT run yet: get out of its way.
    return days_after == 1 or (
        (now_et.hour, now_et.minute) >= _AGENTS_PRE_DECIDE_CUTOFF_HOUR_MINUTE_ET
    )


def _merge_agents_daily_sentinel(existing: dict | None, this_run: dict) -> dict:
    """Merge this run's ticker counters into an EARLIER agents sentinel for
    the SAME asof (an earlier run today), so a later same-day run can never
    erase an earlier run's real attempt/failure record just by writing with a
    higher run_id.

    Root cause fixed here (2026-09-14): two agents runs hit the same asof ~2h
    apart. run1 attempted the full held/triggered set and failed all 11 (LLM
    host DNS outage). run2 was a PAST-CUTOFF deadline-skip re-kick (0 tickers
    attempted -- see _deadline_reached) that nonetheless wrote a sentinel with
    a higher run_id (store.allocate_run_id() is a monotonic per-store
    counter, not asof-scoped). write_sentinel()'s monotonicity guard
    (sma.sentinels._is_strictly_newer) only compares run_id/completed_at, so
    run2's all-zero payload counted as "newer" and clobbered run1's 11-failed
    record with tickers_failed=0 -- erasing the only evidence of a
    100%-failed night.

    sma.sentinels itself is DELIBERATELY left unchanged: last-run-wins-on-
    run_id is the right contract for every other job (predict/decide/
    reconcile/retrain/...), which runs once per asof and must simply replace.
    Only agents can legitimately run several times for one asof (a
    deferred-then-caught-up run, a deadline-skip re-kick, a post-outage
    catch-up), so the merge lives here, scoped to this one CLI command --
    not in the generic sentinel module every job shares.

    Field-by-field policy:
      - Per-ticker LOOP counters (tickers_processed, tickers_failed,
        tickers_skipped_budget, tickers_skipped_existing,
        tickers_skipped_deadline, tickers_cached_fallback) are SUMMED. Each
        one increments exactly once per ticker-iteration within a SINGLE
        run's loop, so summing across today's runs is simply "total
        loop-iteration outcomes today" -- correct whether or not the
        underlying tickers overlap between runs (a ticker retried after an
        earlier failure counts as two real attempts, which IS the truth: see
        _thesis_exists, which only skips a ticker that already has a
        PERSISTED thesis -- a failed ticker has none, so it is fair game for
        a later run to reattempt and either fail again or succeed). Crucially,
        summing means a later run can only ever ADD to tickers_failed, never
        erase it: 11 + 0 = 11, exactly the 9/14 case. And a later run that
        actually succeeds grows tickers_processed on top of the preserved
        failure count, rather than replacing it -- "updates counts" without
        rewriting history.
      - budget_spent_usd SUMS too, matching CostTracker's own daily total
        (SUM(est_cost_usd) over agent_calls for the date) -- the sentinel
        should agree with the ledger it is derived from.
      - force_full is OR'd (a "max" over booleans): if ANY run today did a
        full-universe refresh, the day counts as full-universe.
      - tickers_skipped_no_trigger is a SNAPSHOT of the day's trigger-set
        size computed ONCE before the loop even starts
        (len(universe) - len(triggered)), not a per-iteration counter -- it
        does not represent additional work done, so it is NOT summed (that
        would double-count the same static number on every same-day rerun);
        the current run's own value is kept.
      - label/asof/completed_at/run_id always reflect the CURRENT run (when
        the sentinel was last touched); write_sentinel()'s run_id
        monotonicity guard still protects against a genuinely out-of-order
        write clobbering a newer one.
    """
    if not existing:
        return this_run
    merged = dict(this_run)
    summed_fields = (
        "tickers_processed",
        "tickers_failed",
        "tickers_skipped_budget",
        "tickers_skipped_existing",
        "tickers_skipped_deadline",
        "tickers_cached_fallback",
    )
    for key in summed_fields:
        merged[key] = int(existing.get(key, 0) or 0) + int(this_run.get(key, 0) or 0)
    merged["budget_spent_usd"] = round(
        float(existing.get("budget_spent_usd", 0.0) or 0.0)
        + float(this_run.get("budget_spent_usd", 0.0) or 0.0),
        6,
    )
    merged["force_full"] = bool(existing.get("force_full")) or bool(this_run.get("force_full"))
    return merged


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

    # Upstream gate (2026-08-12): the reboot fired the whole launchd manifest at
    # 20:11, so agents started at 20:15 while ingest was still running (it did not
    # finish until 20:23) and every thesis that night was built on the PREVIOUS
    # day's news and filings. decide refuses to run on a missing upstream via
    # preflight; agents had no equivalent, and it feeds decide's thesis overlay.
    #
    # Defer contract: exit 0 writing NO sentinel. The watchdog re-kicks a job past
    # its deadline with no sentinel inside late_kick_max_hours (agents: deadline
    # 20:15 + the 6h default, with checkpoints at 21/22/23/00:30/01:30), so the
    # run self-heals once ingest lands.
    #
    # Same-day only: historical/manual asofs are ungated, matching the deadline
    # budget's SCHEDULED-run rule below (ingest doctrine).
    if _now_et().date() == asof and read_sentinel(
        label="com.sma.ingest.daily", asof=asof
    ) is None:
        logger.warning(
            "agents: today's ingest sentinel ({}) is absent; deferring rather than "
            "building theses on stale news. No sentinel written — the watchdog "
            "re-kicks this job at its next checkpoint.", asof.isoformat(),
        )
        click.echo(
            f"agents: deferred — com.sma.ingest.daily has not completed for "
            f"{asof.isoformat()}; refusing to build theses on stale news"
        )
        return

    Path(db).parent.mkdir(parents=True, exist_ok=True)

    tickers_processed = 0
    tickers_skipped_no_trigger = 0
    tickers_skipped_budget = 0
    tickers_skipped_existing = 0
    tickers_cached_fallback = 0
    tickers_failed = 0
    failure_messages: dict[str, int] = {}

    with writer_lock(label="agents", timeout_s=600.0):  # queue behind slow predict (2026-06-12)
        store = Store(path=db).connect()
        try:
            pipeline = _build_pipeline(settings, store)
            run_id = store.allocate_run_id()

            # Currently-held names ALWAYS get refreshed, and always go first. The
            # thesis overlay's strong_bearish EXIT trigger only fires on held
            # tickers, but holding was never a refresh trigger — so positions went
            # thesis-less indefinitely (2026-07-29: 7 of 11 held names stale or
            # missing, DKNG 42d / MPC 30d / MRNA 27d / HUM never). Ordering them
            # first matters because the 19:58 cutoff below stops STARTING tickers,
            # so anything in the tail is silently skipped. See refresh_order().
            held = sorted(ledger_net_positions(store))
            is_friday = asof.weekday() == 4
            if is_friday or force_full:
                tickers = refresh_order(
                    triggered=sorted(set(universe_list)),
                    held=held,
                    universe=universe_list,
                )
                click.echo(
                    f"Friday full refresh: {len(tickers)} tickers "
                    f"(book first: {[t for t in held if t in set(universe_list)]})"
                )
            else:
                cfg = TriggerConfig(
                    price_move_pct=settings.agents.triggers.price_move_pct,
                    material_filing_types=settings.agents.triggers.material_filing_types,
                )
                triggered = tickers_needing_refresh(store, asof, universe_list, cfg)
                tickers = refresh_order(
                    triggered=triggered, held=held, universe=universe_list
                )
                tickers_skipped_no_trigger = len(universe_list) - len(tickers)
                n_held = len([t for t in held if t in set(universe_list)])
                click.echo(
                    f"Mon-Thu refresh: {len(tickers)} tickers "
                    f"({n_held} held + {len(tickers) - n_held} triggered): {tickers}"
                )

            # Deadline budget (2026-06-12; gate reworked 2026-08-13, corrected
            # 2026-08-16): the first 267-name night had 81 new tickers with no
            # theses; agents ground ~2.5 min/ticker for HOURS holding the
            # writer lock — decide died at 20:00 on lock timeout. Theses are
            # ADVISORY, so when decide needs the lock, stop STARTING tickers,
            # count the rest as deadline-skipped, write the sentinel, exit 0.
            #
            # Originally a flat 19:58 ET wall-clock cutoff. da322f7
            # (2026-08-12) made agents DEFER when today's ingest hasn't
            # landed, so a post-outage watchdog re-kick can now fire well
            # after 19:58 with ZERO tickers processed yet — under the flat
            # cutoff that re-kick tripped instantly on ticker ONE and skipped
            # the entire universe: a run that "succeeds" (exit 0, sentinel
            # written) but produces no fresh theses at all (found 2026-08-13).
            # 91e0baa then swung the other way and gated purely on decide's
            # sentinel EXISTING — circular, because agents holds the lock
            # decide needs to write it (see _deadline_reached for the two
            # no-trade paths that produced). The rule is now the union:
            # before 19:58 on asof, OR decide has already run. decide can
            # still wait up to 900s for the lock (2026-06-12), so the
            # in-flight ticker finishing a bit past 19:58 is fine.
            _cutoff_reached = False
            tickers_skipped_deadline = 0
            for t in tickers:
                now_et = _now_et()
                if _cutoff_reached or _deadline_reached(now_et=now_et, asof=asof):
                    if not _cutoff_reached:
                        logger.warning(
                            "agents deadline budget: cutoff reached at {} for "
                            "asof {} (past 19:58 ET with decide still pending, "
                            "or the 02:30 floor tripped); releasing the writer "
                            "lock and skipping remaining tickers",
                            now_et.strftime("%H:%M"), asof.isoformat(),
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
                    _msg = str(e)[:300] or type(e).__name__
                    failure_messages[_msg] = failure_messages.get(_msg, 0) + 1
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
        #
        # Merge with an EARLIER sentinel for this same asof (a prior run
        # today) before computing quality, so a later same-day run — in
        # particular a 0-attempt deadline-skip re-kick — can never erase an
        # earlier run's real attempt/failure record. See
        # _merge_agents_daily_sentinel's docstring for the 2026-09-14 incident
        # this fixes and the field-by-field merge policy.
        _existing_sentinel = read_sentinel(label="com.sma.agents.daily", asof=asof)
        _merged = _merge_agents_daily_sentinel(
            _existing_sentinel,
            {
                "tickers_processed": tickers_processed,
                "tickers_skipped_no_trigger": tickers_skipped_no_trigger,
                "tickers_skipped_budget": tickers_skipped_budget,
                "tickers_skipped_existing": tickers_skipped_existing,
                "tickers_skipped_deadline": tickers_skipped_deadline,
                "tickers_cached_fallback": tickers_cached_fallback,
                "tickers_failed": tickers_failed,
                "budget_spent_usd": total_cost,
                "force_full": force_full,
            },
        )

        # quality.passed is required so live_readiness() can gate decide on
        # this sentinel. budget_exhausted is surfaced as a blocking_failure
        # so the budget_exhausted waiver in decide's JobSchedule is live.
        # Computed from the MERGED (whole-day) counters, not just this run's
        # own, so the quality verdict reflects the day as a whole.
        _m_processed = _merged["tickers_processed"]
        _m_failed = _merged["tickers_failed"]
        _m_skipped_budget = _merged["tickers_skipped_budget"]
        _m_cached_fallback = _merged["tickers_cached_fallback"]
        _budget_exhausted = (
            _m_skipped_budget + _m_cached_fallback > 0
            and _m_processed == 0
        )
        # A run that had work to do but produced ZERO theses because every
        # ticker errored is a systemic failure (LLM down, code bug) and must NOT
        # green-light decide via live_readiness(). Partial failures degrade
        # gracefully (decide falls back to model predictions + prior theses), so
        # they're recorded in tickers_failed but don't block.
        _all_failed = _m_failed > 0 and _m_processed == 0
        # A run where MOST attempted tickers errored is a systemic problem
        # (LLM flaking, quota, code bug) even when a few succeeded; it must
        # not read as a healthy advisory. Agents stays an ADVISORY dep, so
        # this never blocks trading — it surfaces honestly in readiness +
        # the dashboard instead of masquerading as passed=True (audit MED).
        _majority_failed = (
            not _all_failed and _m_failed > _m_processed
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
                "tickers_processed": _merged["tickers_processed"],
                "tickers_skipped_no_trigger": _merged["tickers_skipped_no_trigger"],
                "tickers_skipped_budget": _merged["tickers_skipped_budget"],
                "tickers_skipped_existing": _merged["tickers_skipped_existing"],
                "tickers_skipped_deadline": _merged["tickers_skipped_deadline"],
                "tickers_cached_fallback": _merged["tickers_cached_fallback"],
                "tickers_failed": _merged["tickers_failed"],
                "budget_spent_usd": _merged["budget_spent_usd"],
                "force_full": _merged["force_full"],
                "quality": {
                    "passed": not _blocking,
                    "blocking_failures": _blocking,
                },
            },
        )
        if _all_failed or _majority_failed:
            _page_dead_thesis_layer(
                asof=asof,
                processed=_m_processed,
                failed=_m_failed,
                failure_messages=failure_messages,
            )


def _page_dead_thesis_layer(
    *, asof, processed: int, failed: int, failure_messages: dict[str, int]
) -> None:
    """Page when most or all thesis calls failed tonight (2026-10-01: the
    Anthropic credit ran out, 23 of 23 failed, exit 0, nothing paged). decide
    keeps trading on the model alone because agents is advisory; this makes
    sure a human knows the overlay is off and why."""
    from sma.backtest.strategies.xgb_top_k import XGBoostTopKStrategy

    attempted = processed + failed
    if failure_messages:
        top = max(failure_messages, key=failure_messages.get)
        reason = f"most common error ({failure_messages[top]}x): {top}"
    else:
        reason = "no error text captured this run (see agents.err.log)"
    message = (
        f"{asof.isoformat()}: {processed} of {attempted} theses written, {failed} failed; "
        f"{reason}. decide still trades tonight, without fresh theses (agents is "
        f"advisory). Theses older than {XGBoostTopKStrategy.THESIS_STALE_DAYS} days stop "
        "counting, so the veto and exit overlay goes dark if this keeps up."
    )
    logger.error("agents: thesis layer failed: {}", message)
    try:
        notify_failure(title="SMA agents: thesis layer failed tonight", message=message)
    except Exception as e:  # a page must never fail the job that sends it
        logger.warning("agents: page failed: {!r}", e)


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
