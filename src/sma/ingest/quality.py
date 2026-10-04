"""Post-run quality checks.

Six SQL assertions per the spec. Each returns a QualityCheck. The aggregate
QualityReport is `passed` iff all checks passed. Result is also written to
logs/quality/YYYY-MM-DD.txt by the runner.
"""

import contextlib
from dataclasses import dataclass
from datetime import date, timedelta

from sma.ingest.notify import notify_failure
from sma.ingest.store import Store

# Single source of truth for "what does the ledger say we hold" — shared with the
# 09:25 pre-open guard rather than duplicating the paper_fills netting SQL.
# No import cycle: preopen_guard imports nothing from sma at module level.
from sma.live.preopen_guard import ledger_net_positions

# Sentinel-based (not DuckDB-based) read of the agents job's own health, used
# by _check_agents_last_run_healthy below. No import cycle: sma.sentinels
# imports nothing from sma.ingest at module level.
from sma.sentinels import read_sentinel

EXPECTED_SOURCES = [
    "yfinance",
    "alpaca",
    "finnhub_news",
    "alpaca_news",
    "finnhub_fundamentals",
    "newsapi",
    "edgar",
]

# Subset of EXPECTED_SOURCES that MUST succeed for the live decide preflight to
# proceed. Excludes `newsapi` because the free-tier daily quota is regularly
# exhausted (see project note: "NewsAPI is effectively dead; Finnhub news still
# covers the full universe; the redundancy was the point"). EDGAR stays critical
# because 8-K filings feed the Phase 4 thesis pipeline.
CRITICAL_INGEST_SOURCES = [s for s in EXPECTED_SOURCES if s != "newsapi"]

# The truly trade-critical sources: without PRICES there is nothing to trade.
# News/fundamentals/edgar feed overlay features + theses and can flake (DNS,
# quota) without stopping a trade, so they must NOT hard-block the source gate.
PRICE_SOURCES = ("yfinance", "alpaca")
_PRICE_SOURCES = PRICE_SOURCES  # back-compat alias (runner retry pass keys off this set too)

# The 11 SPDR sector ETFs, added to the universe 2026-05-26 as benchmarks for
# the rel_strength_sector_etf_30d feature. They are reference data, never
# tradeable single names, and — being funds — they report neither per-ticker
# earnings nor per-fund news. Listed EXPLICITLY rather than matched on an "XL"
# prefix, which would also swallow real single names (XLNX-style tickers) and
# silently turn a genuine gap into a pass.
_SECTOR_ETFS: frozenset[str] = frozenset({
    "XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY",
})

# Tickers exempt from per-ticker news coverage. ETFs report on their
# holdings, not the fund itself, so per-ticker news feeds usually return
# nothing for them. SPY has always been excluded; NANC (Unusual Whales
# Subversive Democratic Trading ETF) was added 2026-05-14 after the
# universe expansion surfaced it as a perpetual failure.
_NO_NEWS_EXEMPT: frozenset[str] = frozenset({"SPY", "NANC"}) | _SECTOR_ETFS

# Tickers exempt from quarterly-earnings coverage. ETFs report no per-fund
# earnings. Other gaps indicate a real ingest problem and should still fail.
#
# NOTE (2026-07-29): this set used to contain "BRK.B" on the stated grounds that
# "Berkshire Hathaway does not file quarterly". That is factually WRONG — the
# live DB holds quarterly BRK-B reports (2026-08-01, 2026-05-02, 2026-02-28,
# 2025-11-01, ...) and BRK-B passes this check unaided. The entry was also dead
# code: the universe is yfinance-canonical ("BRK-B"), so the dotted string never
# matched anything. Removed rather than re-spelled — keeping it would exempt
# Berkshire from a check it passes, masking a real future earnings gap.
_NO_EARNINGS_EXEMPT: frozenset[str] = frozenset({"SPY", "NANC"}) | _SECTOR_ETFS


@dataclass(frozen=True)
class QualityCheck:
    name: str
    passed: bool
    detail: str
    # Does a failure of this check BLOCK downstream consumers (i.e. live
    # `decide`)? Trade-critical checks (price coverage, sources) block. Overlay /
    # feature-freshness checks (news, earnings, theses) are recorded but must NOT
    # block trading — a flaky news day shouldn't stop the bot from trading.
    blocking: bool = True
    # A check that PASSED but only via a tolerance/fallback path worth a human
    # noticing (a handful of missing tickers under the 97% floor, or SPY's
    # adj_close falling back to a prior session — see _SPY_FALLBACK_MAX_AGE_DAYS
    # below). Meaningless when passed=False. Never gates decide (that's what
    # `blocking` is for); it only drives notify_degraded_quality_checks so a
    # real-but-tolerated data gap doesn't go completely unseen the way the
    # 2026-09-09 SPY gap did before this field existed (log-only, no page).
    degraded: bool = False


@dataclass(frozen=True)
class QualityReport:
    asof_date: date
    run_id: int
    checks: list[QualityCheck]

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def blocking_failures(self, waivers: frozenset[str] = frozenset()) -> list[str]:
        """Failed BLOCKING checks not in `waivers` — the set that gates decide."""
        return [
            c.name for c in self.checks
            if not c.passed and c.blocking and c.name not in waivers
        ]

    def summary(self, waivers: frozenset[str] = frozenset()) -> str:
        """Human-readable report. The OVERALL line is WAIVER-AWARE.

        It used to read `PASS if self.passed`, i.e. all checks — while the sentinel
        gated on `blocking_failures(waivers)`. So the nightly artifact printed
        `OVERALL: FAIL` on runs the pipeline itself considered good (e.g. only the
        waived, non-blocking `theses_freshness` failing). That mismatch is the
        fourth permanently-red gauge found on 2026-07-29 and the same pattern that
        trains a reader to ignore the dashboard entirely.

        Per-check `[PASS]/[FAIL]` lines still report raw truth — nothing is hidden.
        The OVERALL line now answers the question a reader actually has: did this
        run block anything?
        """
        lines = [f"Quality report for {self.asof_date} (run_id={self.run_id})", ""]
        for c in self.checks:
            mark = "PASS" if c.passed else "FAIL"
            lines.append(f"  [{mark}] {c.name}: {c.detail}")
        lines.append("")

        blocking = self.blocking_failures(waivers)
        if blocking:
            lines.append(f"OVERALL: FAIL — blocking: {', '.join(blocking)}")
        else:
            tolerated = [
                c.name for c in self.checks
                if not c.passed and (not c.blocking or c.name in waivers)
            ]
            if tolerated:
                lines.append(
                    f"OVERALL: PASS — nothing blocking; "
                    f"tolerated failure(s): {', '.join(tolerated)}"
                )
            else:
                lines.append("OVERALL: PASS")
        return "\n".join(lines)


# How many calendar days back we'll accept an EARLIER SPY price as a fallback
# when today's run has no usable (non-null adj_close) current-run SPY row.
# 2026-09-09: a
# single-day yfinance batch hiccup (partial "possibly delisted" / sqlite-cache
# OperationalError failures — vendor-side, not a dead network: alpaca and
# every overlay source succeeded that same run) dropped SPY's yfinance row
# while 263/264 other tickers still got a price from *some* source. Because
# alpaca's adj_close is always NULL, SPY had no usable current-run price at
# all and `all_tickers_have_price` hard-blocked the whole book on that one
# name/field — the MRNA-freeze pattern again, and the rebalance was lost.
#
# A CALENDAR-day cutoff, deliberately NOT "the N most recent DISTINCT dates
# present in `prices`" (the style _stale_flags below uses): during a REAL
# total outage nothing lands for ANYONE on the missed days, so "distinct
# dates present" would silently skip straight past the gap to the last
# pre-outage healthy day -- however far back that is -- and wrongly call it
# "recent". A fixed calendar window has no such hole: no rows in
# [asof - N days, asof) means genuinely no market data that recently,
# full stop. 4 days comfortably covers one weekend (or a single adjacent
# holiday) between the current run and the last real session without
# reaching further back than that.
#
# Bounded deliberately small: this must fall back to "a real, recent SPY
# print exists" and nothing looser. It does NOT touch the run_id scoping
# above — that guards a DIFFERENT failure mode (a stale SAME-DATE row from an
# earlier crashed attempt at TODAY papering over a real outage, 2026-06-05
# audit) and stays exactly as strict. This fallback only ever looks at
# STRICTLY EARLIER dates, which are never re-attempted, so an earlier date's
# row can only mean "we have a genuinely recent market price," never "today's
# fetch secretly succeeded." A real multi-day outage (the 6/1-6/3 freeze)
# still has no SPY price within this window and still blocks.
_SPY_FALLBACK_MAX_AGE_DAYS = 4


def _check_all_tickers_have_price(
    store: Store, asof: date, universe: list[str], run_id: int
) -> QualityCheck:
    # Scope to the CURRENT run_id (prices use INSERT OR REPLACE, so a real
    # re-fetch carries the new run_id): otherwise stale same-date rows from an
    # earlier run let a zero-price outage run pass (2026-06-05 audit).
    rows = store.conn.execute(
        "SELECT DISTINCT ticker FROM prices WHERE date = ? AND run_id = ?",
        [asof, run_id],
    ).fetchall()
    have = {r[0] for r in rows}
    missing = sorted(set(universe) - have)
    coverage = (len(universe) - len(missing)) / len(universe) if universe else 1.0
    # SPY needs a USABLE (non-null adj_close) current-run row — the predictor
    # always needs rel_strength_spy; an Alpaca-only null-adj_close SPY row breaks
    # every prediction even though the ticker is "present".
    spy_needed = "SPY" in universe
    spy_ok_current_run = (not spy_needed) or store.conn.execute(
        "SELECT 1 FROM prices WHERE ticker = 'SPY' AND date = ? AND run_id = ? "
        "AND adj_close IS NOT NULL LIMIT 1",
        [asof, run_id],
    ).fetchone() is not None

    spy_fallback_used = False
    spy_ok = spy_ok_current_run
    if spy_needed and not spy_ok_current_run:
        cutoff = asof - timedelta(days=_SPY_FALLBACK_MAX_AGE_DAYS)
        spy_ok = store.conn.execute(
            "SELECT 1 FROM prices WHERE ticker = 'SPY' AND date < ? AND date >= ? "
            "AND adj_close IS NOT NULL LIMIT 1",
            [asof, cutoff],
        ).fetchone() is not None
        spy_fallback_used = spy_ok

    passed = coverage >= 0.97 and spy_ok
    # Passed only via tolerance (a few missing names) or via the SPY fallback
    # above — worth a human's attention even though it doesn't block.
    degraded = passed and (bool(missing) or spy_fallback_used)
    detail = f"coverage={coverage:.1%} spy_ok={spy_ok}"
    if spy_fallback_used:
        detail += " (spy fallback: prior-session adj_close)"
    detail += f" missing={missing[:10]}{'...' if len(missing) > 10 else ''}"
    return QualityCheck(
        name="all_tickers_have_price",
        # Tolerate a handful of missing names (thin tickers / a slow fetch) — the
        # predictor drops missing-price names gracefully, so decide can safely
        # trade the rest. Block only on a real outage (coverage < 97%) OR if SPY
        # has no usable price (current run or the recent-session fallback above).
        passed=passed,
        detail=detail if (missing or not spy_ok or spy_fallback_used) else "all present",
        degraded=degraded,
    )


# Cross-source confirmation tolerance: how closely two vendors' closes must
# agree, on BOTH the day before and the day of a >50% move, for the move to
# count as independently confirmed. Same 2% used by _check_price_divergence,
# and for the same reason — it is wide enough for normal end-of-day print
# differences between market makers, tight enough that a corrupt print can't
# hide inside it.
_CROSS_SOURCE_AGREE_TOL = 0.02

# How large a move a second vendor must itself show to count as confirming.
# Same 50% as the gate: "similar magnitude", enforced precisely by the 2%
# level-agreement below.
_EXTREME_MOVE_THRESHOLD = 0.5


def _cross_source_confirms(
    store: Store, ticker: str, day: date, prev_day: date, direction: int
) -> int:
    """How many INDEPENDENT price vendors show this same extreme move?

    Returns the number of confirming sources, and 0 unless they also agree on
    the actual price LEVEL (within `_CROSS_SOURCE_AGREE_TOL`) on both days —
    agreeing that "something big happened" is not enough; two vendors must be
    describing the same prices.

    COALESCE(adj_close, close) because alpaca stores adj_close as NULL (its
    placeholder); filtering on adj_close would drop every alpaca row and make
    confirmation structurally impossible. Inside a 30-day window adjusted≈raw,
    and where they do drift the effect is to WITHHOLD the exemption, which is
    the safe direction.
    """
    rows = store.conn.execute(
        """
        SELECT source,
               MAX(CASE WHEN date = ? THEN COALESCE(adj_close, close) END) AS px,
               MAX(CASE WHEN date = ? THEN COALESCE(adj_close, close) END) AS prev_px
        FROM prices
        WHERE ticker = ? AND date IN (?, ?)
        GROUP BY source
        """,
        [day, prev_day, ticker, day, prev_day],
    ).fetchall()

    confirming: list[tuple[float, float]] = []
    for _source, px, prev_px in rows:
        if px is None or prev_px is None or prev_px <= 0:
            continue
        move = (px - prev_px) / prev_px
        if abs(move) > _EXTREME_MOVE_THRESHOLD and (move > 0) == (direction > 0):
            confirming.append((px, prev_px))

    if len(confirming) < 2:
        return len(confirming)
    for levels in ([c[0] for c in confirming], [c[1] for c in confirming]):
        hi, lo = max(levels), min(levels)
        if lo <= 0 or (hi - lo) / lo > _CROSS_SOURCE_AGREE_TOL:
            return 0
    return len(confirming)


def _check_no_unjustified_extreme_moves(store: Store, asof: date) -> QualityCheck:
    # Restrict to yfinance because it's the only source that actually applies
    # historical split-adjustment to adj_close. Alpaca's adj_close is a
    # placeholder (= close), so its rows show artificial 50%+ drops on every
    # split day, which would be a constant stream of false positives.
    #
    # NOT isnan(...) (2026-09-18 incident): DuckDB compares/orders NaN as the
    # LARGEST value -- `NaN > 0.5` is TRUE -- so a NaN adj_close (Yahoo
    # returned NaN closes for the whole universe that night) satisfied this
    # gate's >50%-move test against ANY real prior value and got flagged as
    # an "unjustified extreme move", freezing the whole rebalance on a data-
    # availability problem rather than a real price move. yfinance_prices.py
    # now refuses to insert a NaN-close row at all; this filter is defense in
    # depth for legacy/pre-repair rows, and excludes a NaN row from being
    # used as the LAG "previous" value for the next real day too.
    suspect = store.conn.execute("""
        WITH p AS (
            SELECT ticker, date, adj_close, close,
                   LAG(adj_close) OVER (PARTITION BY ticker ORDER BY date) AS prev,
                   LAG(close) OVER (PARTITION BY ticker ORDER BY date) AS prev_close,
                   LAG(date) OVER (PARTITION BY ticker ORDER BY date) AS prev_date
            FROM prices
            WHERE date >= ? - INTERVAL '30 days'
              AND adj_close IS NOT NULL
              AND NOT isnan(adj_close)
              AND (close IS NULL OR NOT isnan(close))
              AND source = 'yfinance'
        )
        SELECT p.ticker, p.date,
               ABS(p.adj_close - p.prev) / NULLIF(p.prev, 0) AS move,
               EXISTS (
                   SELECT 1 FROM filings f
                   WHERE f.ticker = p.ticker
                     AND f.filing_type = '8-K'
                     AND ABS(EXTRACT(EPOCH FROM (f.filed_at - p.date::TIMESTAMP)) / 86400) <= 2
               ) AS has_8k,
               ABS(p.close - p.prev_close) / NULLIF(p.prev_close, 0) AS raw_move,
               p.prev_date,
               p.adj_close - p.prev AS delta
        FROM p
        WHERE p.prev IS NOT NULL
          AND ABS(p.adj_close - p.prev) / NULLIF(p.prev, 0) > 0.5
    """, [asof]).fetchall()

    bad = []
    for ticker, day, move, has_8k, raw_move, prev_day, delta in suspect:
        # EXEMPTION 1 — 8-K corroborated by the RAW close.
        # A bare 8-K is NOT enough to exempt a split-scale adj_close jump: splits
        # and spinoffs file their OWN 8-Ks on the ex-date (Item 2.01 / 5.03 /
        # 8.01), so the un-back-adjusted-split desync this gate exists to catch
        # (KLAC's 3-week 10x corruption) came with a coincident 8-K and slipped
        # through. A GENUINE price-moving event (crash/M&A) moves the RAW close
        # too; an adjustment-basis desync moves adj_close alone while raw close
        # stays smooth. So exempt only when an 8-K is present AND raw close
        # corroborates the move (>50%).
        if has_8k and raw_move is not None and raw_move > 0.5:
            continue

        # EXEMPTION 2 — CROSS-SOURCE CONFIRMATION (added 2026-08-24).
        # What this gate actually catches is SINGLE-SOURCE corruption: a bad
        # fetch, or a split mis-adjustment in one vendor. Two independent
        # vendors that each show the same-direction, same-magnitude move AND
        # agree on the price level on both days are not corrupting in lockstep —
        # they are reporting market reality. That evidence is strictly STRONGER
        # than the 8-K heuristic, which only asks whether a filing happened to
        # land nearby (and which splits themselves satisfy).
        #
        # A ticker carried by ONE source gets no new exemption: with nothing to
        # cross-check against, a lone spike is exactly the corruption signature,
        # so it still fails. Conservative by construction.
        #
        # The case that forced this: MRNA's real +177% on 2026-08-19
        # ($62.93 -> $174.27), identical in alpaca and yfinance, raw AND
        # adjusted, with no 8-K on file since 8/1. Under the 8-K-only rule the
        # gate failed every ingest from 8/19 on and — since predict and decide
        # gate on it via the ingest sentinel — froze live trading for six
        # sessions while the book carried MRNA at 22%. The row would only have
        # aged out of the 30-day window around 9/18.
        if _cross_source_confirms(store, ticker, day, prev_day, 1 if delta > 0 else -1) >= 2:
            continue

        bad.append((ticker, day, move))

    return QualityCheck(
        name="no_unjustified_extreme_moves",
        passed=len(bad) == 0,
        detail=f"unjustified={[(t, d, round(m, 3)) for t, d, m in bad[:5]]}"
               if bad else "no extreme moves",
    )


def _check_news_table_grew(store: Store, asof: date, run_id: int) -> QualityCheck:
    if asof.weekday() >= 5:
        return QualityCheck("news_table_grew", True, "skipped (weekend)", blocking=False)
    n = store.conn.execute(
        "SELECT COUNT(*) FROM news WHERE run_id = ?", [run_id],
    ).fetchone()[0]
    return QualityCheck(
        name="news_table_grew",
        passed=n >= 10,
        detail=f"rows_in_run={n}",
        blocking=False,  # news is an overlay feature — never block trading on it
    )


def _check_no_excessive_nulls(store: Store, run_id: int) -> QualityCheck:
    """Check NULL fractions for THIS RUN's rows only.

    Scoping by run_id is critical: counting NULLs across the whole table
    surfaces stale state from any past partial ingest (e.g. an earlier
    run that wrote rows mid-flight before another source had committed).
    Those would generate confusing FAIL reports forever after they
    self-resolved.
    """
    failures = []
    for table, col in [("prices", "close"), ("news", "headline"),
                       ("filings", "url"), ("sentiment", "score")]:
        row = store.conn.execute(
            f"SELECT COUNT(*), COUNT({col}) FROM {table} WHERE run_id = ?",
            [run_id],
        ).fetchone()
        total, non_null = row
        if total == 0:
            continue
        null_frac = (total - non_null) / total
        if null_frac > 0.05:
            failures.append((table, col, round(null_frac, 3)))
    return QualityCheck(
        name="no_excessive_nulls",
        passed=len(failures) == 0,
        detail=f"failures={failures}" if failures else "all tables under 5% nulls (this run)",
    )


def _check_enough_sources_succeeded(store: Store, run_id: int) -> QualityCheck:
    rows = store.conn.execute(
        "SELECT source, status, rows_inserted FROM ingest_log WHERE run_id = ?", [run_id],
    ).fetchall()
    by_source = {s: st for s, st, _ in rows}
    rows_by_source = {s: (ri or 0) for s, _, ri in rows}
    # A price source counts ONLY if it succeeded AND actually wrote rows:
    # yfinance/alpaca can return status='ok' with 0 rows (empty response), which
    # is no tradable data (2026-06-05 audit).
    price_ok = [
        s for s in _PRICE_SOURCES
        if by_source.get(s) == "ok" and rows_by_source.get(s, 0) > 0
    ]
    ok = sum(1 for s in EXPECTED_SOURCES if by_source.get(s) == "ok")
    return QualityCheck(
        name="enough_sources_succeeded",
        # At least ONE price source must succeed — a run where every price source
        # failed has no tradable data and must block, even if 5 news/overlay
        # sources are 'ok' (the prior `any 5` rule let that through).
        passed=len(price_ok) >= 1,
        detail=f"price_ok={price_ok} ok={ok}/{len(EXPECTED_SOURCES)}",
    )


def _check_price_divergence(store: Store, asof: date) -> QualityCheck:
    """Detect cross-source disagreement on the same ticker-day.

    Threshold raised from 1% → 2% on 2026-05-18 after the universe
    expansion produced ~daily flags at the 1.1-1.2% boundary on
    low-priced names (F ~$12) and ETFs (NANC). The check is here to
    catch missing dividend adjustments and bad data — not normal
    end-of-day print differences between market makers. 2% catches
    every real issue we've ever seen; 1% just spammed the sentinel.

    Excludes NANC + other ETFs (already exempt from news/earnings).
    """
    # COALESCE(adj_close, close): alpaca stores adj_close as NULL (its
    # placeholder), so filtering on adj_close alone dropped every alpaca row
    # — nsrc was always 1 and sources were NEVER actually compared (audit
    # quality.py:218). Inside the 7-day window adjusted≈raw (a week's
    # dividend drift is far below the 2% threshold), so the mixed-scale
    # comparison is safe.
    #
    # NOT isnan(...) (2026-09-18 incident): a NaN close/adj_close (Yahoo
    # returned NaN for the whole universe that night) is not NULL, so it
    # survived the IS NOT NULL filter and DuckDB's MAX/MIN treat NaN as the
    # largest value -- `(hi - lo) / lo > 0.02` came out TRUE for every ticker,
    # flagging a data-availability problem as cross-source divergence.
    # yfinance_prices.py now refuses to insert a NaN-close row at all; this
    # filter is defense in depth for legacy/pre-repair rows.
    bad = store.conn.execute("""
        WITH s AS (
            SELECT ticker, date,
                   MAX(COALESCE(adj_close, close)) AS hi,
                   MIN(COALESCE(adj_close, close)) AS lo,
                   COUNT(DISTINCT source) AS nsrc
            FROM prices
            WHERE date >= ? - INTERVAL '7 days'
              AND COALESCE(adj_close, close) IS NOT NULL
              AND NOT isnan(COALESCE(adj_close, close))
              AND ticker NOT IN ('SPY', 'NANC')
            GROUP BY ticker, date
        )
        SELECT ticker, date, hi, lo, (hi - lo) / NULLIF(lo, 0) AS spread
        FROM s
        WHERE nsrc > 1 AND (hi - lo) / NULLIF(lo, 0) > 0.02
    """, [asof]).fetchall()
    return QualityCheck(
        name="no_cross_source_price_divergence",
        passed=len(bad) == 0,
        detail=f"divergent={[(r[0], r[1], round(r[4], 4)) for r in bad[:5]]}"
               if bad else "all sources agree within 2%",
        # Informational, not trade-critical — schedule.py already waives it for
        # decide. Mark non-blocking so code and config agree (Codex follow-up):
        # cross-source disagreement is a data-quality flag, but the canonical
        # (yfinance-first) read is still tradable.
        blocking=False,
    )


def _check_news_per_ticker_minimum(store: Store, asof: date, universe: list[str]) -> QualityCheck:
    """Every active ticker should have >=1 news article in the last 30 days.

    Some tickers legitimately don't generate per-ticker news (ETFs in
    particular — exchanges report on the fund's holdings, not the fund
    itself). Exclude them so the check stays signal-rich.
    """
    rows = store.conn.execute(
        """
        SELECT ticker, COUNT(*) AS n
        FROM news
        WHERE published_at >= ? - INTERVAL '30 days'
        GROUP BY ticker
        """,
        [asof],
    ).fetchall()
    have = {r[0] for r in rows}
    missing = sorted(set(universe) - have - _NO_NEWS_EXEMPT)
    return QualityCheck(
        name="news_per_ticker_minimum",
        passed=len(missing) == 0,
        detail=(
            f"missing_news={missing[:10]}{'...' if len(missing) > 10 else ''}"
            if missing else "all tickers have news in last 30 days"
        ),
        blocking=False,  # overlay-feature coverage — record, don't block trading
    )


def _check_earnings_coverage(store: Store, asof: date, universe: list[str]) -> QualityCheck:
    """Every ticker should have >=1 earnings record in the last 12 months.

    ETFs (SPY, NANC, the 11 SPDR sector funds) report no per-fund earnings, so
    they are excluded to keep the check signal-rich. Other gaps usually indicate
    a Finnhub quota / API flakiness issue and are fixable via
    `backfill-earnings`.
    """
    rows = store.conn.execute(
        """
        SELECT ticker, COUNT(*) AS n
        FROM earnings
        WHERE report_date >= ? - INTERVAL '12 months'
        GROUP BY ticker
        """,
        [asof],
    ).fetchall()
    have = {r[0] for r in rows}
    missing = sorted(set(universe) - have - _NO_EARNINGS_EXEMPT)
    return QualityCheck(
        name="earnings_coverage",
        passed=len(missing) == 0,
        detail=(
            f"missing_earnings={missing[:10]}{'...' if len(missing) > 10 else ''}"
            if missing else "all tickers have recent earnings"
        ),
        blocking=False,  # overlay-feature coverage — record, don't block trading
    )


def _check_theses_freshness(store: Store, asof: date, universe: list[str]) -> QualityCheck:
    """Every ticker WE HOLD should have a thesis within the last 7 calendar days.

    Scope changed 2026-07-29, from universe-wide to held-only. Two reasons:

    1. Universe-wide was unsatisfiable by construction. The agents job writes
       4-63 theses/night (event-triggered + budget-capped), so ~114 of 266 within
       7 days is the steady state — the check was red EVERY night and became one
       of four permanently-red gauges that trained the habit of ignoring the
       dashboard entirely.
    2. It was the wrong question. In `xgb_top_k` a stale/missing thesis is treated
       as NEUTRAL and cannot fire any rule, so a stale thesis on the #200-ranked
       name is harmless. What costs money is a stale thesis on a name we HOLD,
       because the `strong_bearish` EXIT trigger only fires on held tickers.
       Measured on the live book that day: 7 of 11 held names stale or missing
       (DKNG 42d, MPC 30d, MRNA 27d, HUM never had one) — the exit rule was dead
       on the majority of the book during the project's worst drawdown.

    This is NOT a loosening to force green: it points the gauge at the money path
    and stays red until the book is actually covered. Universe coverage is still
    reported as context so a slow slide stays visible. `refresh_order()` in
    sma.agents.triggers is the corresponding fix that keeps the book covered.

    SPY (and other ETF benchmarks) never get theses, so they are excluded.
    """
    rows = store.conn.execute(
        """
        SELECT ticker, MAX(asof_date) AS latest
        FROM theses
        WHERE asof_date >= ? - INTERVAL '14 days'
        GROUP BY ticker
        """,
        [asof],
    ).fetchall()
    have = {r[0]: r[1] for r in rows}

    def _is_stale(t: str) -> bool:
        latest = have.get(t)
        return latest is None or (asof - latest).days > 7

    universe_set = set(universe)
    exempt = _NO_NEWS_EXEMPT | {"SPY"}
    # Held per the paper_fills ledger, intersected with the universe.
    held = sorted(
        t for t in ledger_net_positions(store)
        if t in universe_set and t not in exempt
    )
    stale_held = [t for t in held if _is_stale(t)]

    # Informational only: overall coverage, so a slide is still visible.
    tracked = [t for t in universe if t not in exempt]
    covered = sum(1 for t in tracked if not _is_stale(t))
    coverage = f"universe coverage {covered}/{len(tracked)}"

    return QualityCheck(
        name="theses_freshness",
        passed=len(stale_held) == 0,
        detail=(
            f"stale HELD names={stale_held[:10]}"
            f"{'...' if len(stale_held) > 10 else ''} "
            f"({len(stale_held)}/{len(held)} of the book); {coverage}"
            if stale_held
            else f"all {len(held)} held names have theses within 7 days; {coverage}"
        ),
        blocking=False,  # agent-thesis freshness is gated by the agents sentinel,
        # not the ingest gate — recording here is fine, but it must not block decide
    )


# ---------------------------------------------------------------------------
# agents_last_run_healthy (2026-09-18) -- unmasks a failed-but-hidden agents run.
#
# Diagnosed 2026-09-18: the agents job failed 100% of its LLM calls on 8/31,
# 9/4, 9/14, and 9/16 (host DNS outages, not a code bug), yet theses_freshness
# above stayed PASS throughout -- the 11 held names got real theses on one
# good night (9/15) and theses_freshness's 7-day recency window hides a
# failed night as long as SOME night in the trailing week produced a fresh
# thesis. A 100%-failed agents run is real signal (host DNS down, a code bug,
# an exhausted budget) worth a human's attention even though agents stays an
# ADVISORY dependency that must never block trading -- exactly the same
# "passed, but only via tolerance" shape as all_tickers_have_price's SPY
# fallback (c9165d0), so this reuses that same degraded-notify path rather
# than inventing a new one.
#
# Reads the agents sentinel (sma.sentinels.read_sentinel), NOT the DuckDB
# theses table: a 0-attempt deadline-skip run (_deadline_reached in
# sma.agents.__main__) leaves no trace in the DB at all -- indistinguishable
# from "nothing was due" without the sentinel's explicit tickers_skipped_
# deadline counter. That distinction is exactly what keeps a schedule miss
# (already covered by sentinel-missing/late-kick detection) from being
# double-counted here as a failure.
# ---------------------------------------------------------------------------

_AGENTS_SENTINEL_LABEL = "com.sma.agents.daily"

# "Failed 100%" or "failed >=50%" of ticker attempts in one run is a systemic
# signal (LLM host down, code bug, exhausted budget) worth flagging even
# though agents stays advisory. Halfway between "any failure" (too noisy --
# one flaky ticker happens most nights) and "total failure only" (would miss
# a majority-degraded night that is still clearly broken).
_AGENTS_FAILURE_RATIO_THRESHOLD = 0.5

# "the last N agents runs" for the persistent-zero-success check below.
_AGENTS_HEALTH_LOOKBACK_RUNS = 2

# How many calendar days to scan backward looking for the last
# _AGENTS_HEALTH_LOOKBACK_RUNS sentinels that actually attempted tickers.
# Comfortably covers weekends/holidays and any run of 0-attempt deadline-skip
# nights without scanning indefinitely.
_AGENTS_HEALTH_LOOKBACK_WINDOW_DAYS = 14


def _agents_attempted(sentinel: dict) -> int:
    """Tickers where the LLM pipeline was actually invoked for this sentinel's
    asof -- processed + failed + budget-skipped + cached-fallback. Excludes
    tickers_skipped_deadline/_no_trigger/_existing, none of which ever reached
    pipeline.run() (see sma.agents.__main__.run). After the same-day sentinel
    merge (_merge_agents_daily_sentinel), these are already summed across
    every run for the day, so this is the WHOLE day's attempted count."""
    return sum(
        int(sentinel.get(k, 0) or 0)
        for k in (
            "tickers_processed",
            "tickers_failed",
            "tickers_skipped_budget",
            "tickers_cached_fallback",
        )
    )


def _agents_failure_signature(store: Store, asof: date) -> str:
    """Most common agent_calls error message for `asof` (e.g. "Connection
    error"), read via the existing store connection -- agent_calls lives in
    the same DuckDB file every other check here already queries, so this is
    an ordinary SELECT on an already-open read connection, not a second
    writer (see sma.sentinels' module docstring on the no-mixed-reader-with-
    writer constraint that governs sentinel files, not in-process queries).

    Scoped by asof_date, NOT run_id: a merged sentinel's run_id is always the
    LATEST run's (see _merge_agents_daily_sentinel), which may not be the run
    that actually failed. asof_date naturally covers every run_id that
    touched this trading day, matching the SUMMED tickers_failed semantics of
    the merge.
    """
    row = store.conn.execute(
        """
        SELECT error, COUNT(*) AS n
        FROM agent_calls
        WHERE asof_date = ? AND status = 'error' AND error IS NOT NULL
        GROUP BY error
        ORDER BY n DESC
        LIMIT 1
        """,
        [asof],
    ).fetchone()
    return row[0] if row else "unknown error"


def _check_agents_last_run_healthy(store: Store, asof: date) -> QualityCheck:
    """DEGRADED (non-blocking) companion to theses_freshness: flags an agents
    job that is failing its LLM calls right now, even on a night
    theses_freshness itself still reports PASS (its 7-day window can hide a
    failed night sitting next to a good one).

    Fires degraded when EITHER:
      1. The most recent agents run for `asof` attempted >0 tickers and
         failed >= _AGENTS_FAILURE_RATIO_THRESHOLD (50%) of them -- e.g. an
         "11/11 calls failed: Connection error" DNS-outage night.
      2. The last _AGENTS_HEALTH_LOOKBACK_RUNS (2) agents runs that actually
         attempted tickers (0-attempt deadline-skip nights don't count --
         those are schedule misses, not runs) both produced ZERO fresh
         theses (tickers_processed == 0) despite having tickers to work on.
         Catches a lower-ratio-but-still-broken pattern (mostly budget-
         skipped/cached-fallback with a few real errors) that never crosses
         the single-night ratio in (1).

    A run with 0 tickers attempted (deadline-skip, or nothing due) is NEVER
    counted as failed here -- that's a schedule miss, already covered by
    sentinel-missing/late-kick detection, not a health signal about the LLM
    pipeline itself.
    """
    sentinel = read_sentinel(label=_AGENTS_SENTINEL_LABEL, asof=asof)
    if sentinel is None:
        # Ingest calls this at 18:30, before tonight's agents run (19:45), so
        # today's sentinel never exists yet here. Judge the most recent prior
        # run instead; reading only today's made this check say "no agents
        # sentinel yet" on every report and miss every failed night (flaw
        # hunt 2026-10-01 B7: the 10/1 credit-exhausted night, 23/23 failed).
        day = asof - timedelta(days=1)
        for _ in range(_AGENTS_HEALTH_LOOKBACK_WINDOW_DAYS):
            sentinel = read_sentinel(label=_AGENTS_SENTINEL_LABEL, asof=day)
            if sentinel is not None:
                asof = day
                break
            day -= timedelta(days=1)
    if sentinel is None:
        return QualityCheck(
            name="agents_last_run_healthy",
            passed=True,
            detail=f"no agents sentinel in the {_AGENTS_HEALTH_LOOKBACK_WINDOW_DAYS} days "
            f"up to {asof.isoformat()}",
            blocking=False,
        )

    attempted = _agents_attempted(sentinel)
    failed = int(sentinel.get("tickers_failed", 0) or 0)
    processed = int(sentinel.get("tickers_processed", 0) or 0)

    if attempted == 0:
        return QualityCheck(
            name="agents_last_run_healthy",
            passed=True,
            detail=(
                f"0 tickers attempted for {asof.isoformat()} "
                "(deadline-skip or nothing due -- a schedule miss, not a health signal)"
            ),
            blocking=False,
        )

    if failed / attempted >= _AGENTS_FAILURE_RATIO_THRESHOLD:
        sig = _agents_failure_signature(store, asof)
        return QualityCheck(
            name="agents_last_run_healthy",
            passed=True,
            degraded=True,
            detail=f"agents: {failed}/{attempted} calls failed: {sig} (asof {asof.isoformat()})",
            blocking=False,
        )

    recent: list[tuple[date, dict]] = []
    day = asof
    for _ in range(_AGENTS_HEALTH_LOOKBACK_WINDOW_DAYS):
        s = read_sentinel(label=_AGENTS_SENTINEL_LABEL, asof=day)
        if s is not None and _agents_attempted(s) > 0:
            recent.append((day, s))
            if len(recent) == _AGENTS_HEALTH_LOOKBACK_RUNS:
                break
        day -= timedelta(days=1)

    if len(recent) == _AGENTS_HEALTH_LOOKBACK_RUNS and all(
        int(s.get("tickers_processed", 0) or 0) == 0 for _, s in recent
    ):
        days_desc = ", ".join(d.isoformat() for d, _ in reversed(recent))
        return QualityCheck(
            name="agents_last_run_healthy",
            passed=True,
            degraded=True,
            detail=(
                f"agents: zero fresh theses across the last "
                f"{_AGENTS_HEALTH_LOOKBACK_RUNS} runs with tickers attempted ({days_desc})"
            ),
            blocking=False,
        )

    return QualityCheck(
        name="agents_last_run_healthy",
        passed=True,
        detail=f"{processed}/{attempted} tickers produced fresh theses for {asof.isoformat()}",
        blocking=False,
    )


# ---------------------------------------------------------------------------
# no_dead_or_frozen_tickers (2026-08-31) -- corporate-action detector.
#
# EA was delisted 2026-08-04 (LBO) and AVB merged away 2026-08-17 (converted to
# EQR shares), yet nothing noticed for weeks: Yahoo kept serving EA a FROZEN
# price ($209.70, bit-identical close after close) so `all_tickers_have_price`
# and `enough_sources_succeeded` both stayed green -- ingest quality passed and
# the model ranked a dead ticker on phantom data (see
# scripts/verify_avb_ea_delisting.py for the full diagnosis). Neither existing
# check looks at the SHAPE of a ticker's own price history over time; both only
# ask "did *a* row land today". This check closes that gap for two corporate-
# action signatures:
#
#   FROZEN -- the last `frozen_run` (default 5) deduped closes are bit-
#             identical. Real equities essentially never print 5 identical
#             closes in a row; a vendor serving a stale last-known quote after
#             a delisting/halt does exactly this.
#   STALE  -- no price row from ANY source for the last `stale_sessions`
#             (default 3) sessions, while most of the rest of the universe
#             DOES have rows in that window (guards against flagging every
#             ticker during a systemic source outage, which
#             all_tickers_have_price / enough_sources_succeeded already catch).
#
# NON-BLOCKING by design (blocking=False below): a single dead name must not
# freeze the whole book, the way the MRNA false-positive briefly did before
# the cross-source-confirmation exemption above. Instead, a NEW flag (one this
# codebase hasn't already paged on, per sma.ingest.dead_ticker_state's dedup)
# triggers a notify_failure ntfy alert once per ticker, not every night it
# stays dead -- see notify_new_dead_or_frozen_tickers below, wired from
# sma.ingest.__main__ (not from run_quality_checks itself, so importing/
# calling this module in tests, backtests, and the dashboard never fires a
# real notification as a side effect of just running the checks).
# ---------------------------------------------------------------------------

DEFAULT_FROZEN_RUN_LENGTH = 5
DEFAULT_STALE_SESSIONS = 3

# How many trailing deduped rows to pull per ticker when looking for a frozen
# run. Must be >= frozen_run; the extra headroom lets the flag report the
# TRUE run length (e.g. "19 identical closes") rather than clamping it to the
# minimum that triggers the check.
_FROZEN_REPORT_LOOKBACK = 90


@dataclass(frozen=True)
class DeadOrFrozenFlag:
    """One ticker flagged by no_dead_or_frozen_tickers."""

    ticker: str
    kind: str  # "frozen" or "stale"
    last_price_date: date | None
    # frozen: consecutive identical deduped closes ending at last_price_date.
    # stale: the `stale_sessions` window size that found zero rows.
    run_length: int


def _frozen_flags(
    store: Store, asof: date, universe: list[str], frozen_run: int
) -> list[DeadOrFrozenFlag]:
    if not universe or frozen_run < 1:
        return []
    # Same dedup convention as sma.model.predictor / scripts/verify_avb_ea_
    # delisting.py: one row per ticker-day, yfinance preferred over alpaca
    # (alpaca's adj_close is always NULL, so in practice this is yfinance's
    # own price history -- the exact series that showed EA's frozen $209.70).
    rows = store.conn.execute(
        """
        WITH deduped AS (
            SELECT ticker, date, adj_close AS px,
                   ROW_NUMBER() OVER (
                       PARTITION BY ticker, date
                       ORDER BY CASE source WHEN 'yfinance' THEN 0
                                             WHEN 'alpaca' THEN 1
                                             ELSE 2 END
                   ) AS src_rn
            FROM prices
            WHERE ticker = ANY(?) AND date <= ? AND adj_close IS NOT NULL
        ),
        ranked AS (
            SELECT ticker, date, px,
                   ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date DESC) AS day_rn
            FROM deduped
            WHERE src_rn = 1
        )
        SELECT ticker, date, px
        FROM ranked
        WHERE day_rn <= ?
        ORDER BY ticker, date DESC
        """,
        [universe, asof, max(frozen_run, _FROZEN_REPORT_LOOKBACK)],
    ).fetchall()

    by_ticker: dict[str, list[tuple[date, float]]] = {}
    for ticker, day, px in rows:
        by_ticker.setdefault(ticker, []).append((day, px))

    flags = []
    for ticker, series in by_ticker.items():
        # `series` is already DATE DESC (most-recent first) per the query's
        # ORDER BY. Need at least frozen_run rows of history to judge at all.
        if len(series) < frozen_run:
            continue
        newest_px = series[0][1]
        run_length = 0
        for _day, px in series:
            if px == newest_px:
                run_length += 1
            else:
                break
        if run_length >= frozen_run:
            flags.append(
                DeadOrFrozenFlag(
                    ticker=ticker,
                    kind="frozen",
                    last_price_date=series[0][0],
                    run_length=run_length,
                )
            )
    return flags


# STALE only fires for the minority of the universe missing data when the
# MAJORITY has it -- otherwise a systemic source outage (already caught by
# all_tickers_have_price / enough_sources_succeeded) would page once per
# ticker for the entire universe instead of surfacing as the one outage it is.
_STALE_MIN_UNIVERSE_COVERAGE = 0.8


def _stale_flags(
    store: Store, asof: date, universe: list[str], stale_sessions: int
) -> list[DeadOrFrozenFlag]:
    if not universe or stale_sessions < 1:
        return []
    session_rows = store.conn.execute(
        "SELECT DISTINCT date FROM prices WHERE date <= ? ORDER BY date DESC LIMIT ?",
        [asof, stale_sessions],
    ).fetchall()
    session_dates = [r[0] for r in session_rows]
    if len(session_dates) < stale_sessions:
        # Not enough session history in the DB yet to judge "the last
        # `stale_sessions` sessions" -- skip rather than false-flag early
        # history / a fresh test DB.
        return []

    rows = store.conn.execute(
        "SELECT ticker, COUNT(DISTINCT date) FROM prices "
        "WHERE ticker = ANY(?) AND date = ANY(?) GROUP BY ticker",
        [universe, session_dates],
    ).fetchall()
    have_rows = {t for t, n in rows if n > 0}

    coverage = len(have_rows) / len(universe)
    if coverage < _STALE_MIN_UNIVERSE_COVERAGE:
        return []

    flags = []
    for ticker in universe:
        if ticker in have_rows:
            continue
        last_row = store.conn.execute(
            "SELECT MAX(date) FROM prices WHERE ticker = ?", [ticker]
        ).fetchone()
        last_price_date = last_row[0] if last_row else None
        flags.append(
            DeadOrFrozenFlag(
                ticker=ticker,
                kind="stale",
                last_price_date=last_price_date,
                run_length=stale_sessions,
            )
        )
    return flags


def find_dead_or_frozen_tickers(
    store: Store,
    asof_date: date,
    universe: list[str],
    *,
    frozen_run: int = DEFAULT_FROZEN_RUN_LENGTH,
    stale_sessions: int = DEFAULT_STALE_SESSIONS,
) -> list[DeadOrFrozenFlag]:
    """Every FROZEN and STALE flag for `universe` as of `asof_date`. Pure/
    read-only -- safe to call from tests, backtests, and the dashboard.
    Public (no leading underscore) because sma.ingest.runner calls it directly
    to hand flags to the CLI's notify wiring, not just from the check below."""
    return _frozen_flags(store, asof_date, universe, frozen_run) + _stale_flags(
        store, asof_date, universe, stale_sessions
    )


def _check_no_dead_or_frozen_tickers(
    store: Store,
    asof: date,
    universe: list[str],
    *,
    frozen_run: int = DEFAULT_FROZEN_RUN_LENGTH,
    stale_sessions: int = DEFAULT_STALE_SESSIONS,
) -> QualityCheck:
    flags = find_dead_or_frozen_tickers(
        store, asof, universe, frozen_run=frozen_run, stale_sessions=stale_sessions
    )
    frozen = [f for f in flags if f.kind == "frozen"]
    stale = [f for f in flags if f.kind == "stale"]
    detail = (
        f"frozen={[(f.ticker, str(f.last_price_date), f.run_length) for f in frozen[:5]]} "
        f"stale={[(f.ticker, str(f.last_price_date)) for f in stale[:5]]}"
        if flags
        else "no dead or frozen tickers"
    )
    return QualityCheck(
        name="no_dead_or_frozen_tickers",
        passed=len(flags) == 0,
        detail=detail,
        # A single delisted/halted name must not freeze the whole book (the
        # MRNA false-positive already taught this lesson for a different
        # check). notify_new_dead_or_frozen_tickers below pages once per
        # ticker independently of this non-blocking status.
        blocking=False,
    )


# ---------------------------------------------------------------------------
# no_nan_prices (2026-09-18 fix, layer 3) -- dedicated NaN-price visibility
# check.
#
# The 9/18 incident: Yahoo returned that night's bar with NaN close/adj_close
# for ALL 264 tickers. yfinance_prices.py's insert path now refuses to write
# a NaN-close row at all (dropped at the source, one WARNING per ticker-run),
# and no_unjustified_extreme_moves / no_cross_source_price_divergence above
# now explicitly ignore any NaN that still makes it into the table (legacy
# rows, or a future price writer that doesn't go through that guard). But
# NONE of those fixes puts the underlying data-availability problem in front
# of a human -- a NaN close is silently just "not there" to every other
# check. This check exists purely to surface it.
#
# Deliberately NEVER blocking and ALWAYS passed=True: a NaN close is a data-
# availability problem for the source-level fix above (and
# enough_sources_succeeded / all_tickers_have_price) to handle, not grounds
# to freeze the whole book the way the un-fixed extreme-moves/divergence
# checks did. `degraded=True` when found routes it through the EXISTING
# notify_degraded_quality_checks pager (same mechanism as
# all_tickers_have_price's SPY fallback) -- so this class of failure PAGES
# instead of freezing, which is the whole point of this fix.
# ---------------------------------------------------------------------------

_NAN_PRICES_LOOKBACK_SESSIONS = 5


def _check_adjusted_price_coverage(
    store: Store, asof: date, universe: list[str]
) -> QualityCheck:
    """DEGRADED (never blocking): universe names with no usable bar for asof.

    all_tickers_have_price counts any row for the date, and Alpaca writes its
    rows with adj_close NULL on purpose, while the predictor reads only rows
    WHERE adj_close IS NOT NULL. So a ticker yfinance missed passes coverage
    and is scored on its previous bar with nothing said (flaw hunt 2026-10-01
    A3: 9/24 had 237 failed yfinance downloads, quality PASS, 264 predictions;
    9/28 had 21, 9/30 had 19). This names them and pages. It deliberately
    does not block: on the 97% floor it would have cancelled 9/24, 9/28 and
    9/30, and whether a stale-bar night should trade is Rayan's call, not a
    data-check side effect.
    """
    if not universe:
        return QualityCheck(
            name="adjusted_price_coverage", passed=True, detail="empty universe",
            blocking=False,
        )
    rows = store.conn.execute(
        "SELECT DISTINCT ticker FROM prices WHERE date = ? AND adj_close IS NOT NULL",
        [asof],
    ).fetchall()
    have = {r[0] for r in rows}
    stale = sorted(set(universe) - have)
    detail = f"usable (adj_close) bars for {len(universe) - len(stale)}/{len(universe)}"
    if stale:
        detail += (
            f"; scored on a previous bar: {stale[:15]}"
            f"{f' (+{len(stale) - 15} more)' if len(stale) > 15 else ''}"
        )
    return QualityCheck(
        name="adjusted_price_coverage",
        passed=True,
        detail=detail,
        blocking=False,
        degraded=bool(stale),
    )


def _check_no_nan_prices(store: Store, asof: date) -> QualityCheck:
    session_rows = store.conn.execute(
        "SELECT DISTINCT date FROM prices WHERE date <= ? ORDER BY date DESC LIMIT ?",
        [asof, _NAN_PRICES_LOOKBACK_SESSIONS],
    ).fetchall()
    session_dates = [r[0] for r in session_rows]
    if not session_dates:
        return QualityCheck(
            name="no_nan_prices",
            passed=True,
            detail="no price rows yet",
            blocking=False,
        )

    bad = store.conn.execute(
        """
        SELECT ticker, date, source
        FROM prices
        WHERE date = ANY(?)
          AND (
              (close IS NOT NULL AND isnan(close))
              OR (adj_close IS NOT NULL AND isnan(adj_close))
          )
        ORDER BY date DESC, ticker
        """,
        [session_dates],
    ).fetchall()

    clean = len(bad) == 0
    detail = (
        f"no NaN close/adj_close rows in the last {len(session_dates)} session(s)"
        if clean
        else (
            f"NaN price rows: {[(t, str(d), s) for t, d, s in bad[:10]]}"
            f"{'...' if len(bad) > 10 else ''}"
        )
    )
    return QualityCheck(
        name="no_nan_prices",
        passed=True,  # never blocks -- a data-availability problem, not a reason to halt
        detail=detail,
        blocking=False,
        degraded=not clean,
    )


# ---------------------------------------------------------------------------
# no_split_inconsistency (2026-10-01) -- MNST 2:1 on 2026-08-11 left the
# yfinance series with a fake -51% day (07-17 -> 07-20) for a month: rows
# inside the nightly window were re-fetched split-adjusted, older rows never
# were. The model read it and ranked MNST top 10 all September. This runs the
# cross-source rule in sma.ingest.split_audit over the trailing
# _SPLIT_INCONSISTENCY_LOOKBACK_SESSIONS sessions for every ticker: a day
# where yfinance and alpaca returns differ by more than the threshold, or a
# one-source jump >40% with the other source missing. Alpaca is raw by
# design, so a real split (alpaca jumps by a clean split ratio, yfinance
# flat) is expected and ignored; everything else is a discontinuity in the
# series the features read.
#
# Never blocking, always passed=True: one bad ticker must not freeze the
# book. degraded=True pages via notify_degraded_quality_checks and names the
# tickers; the fix is `python -m sma.ingest repair-splits --all-flagged`.
# Threshold knob: config.yaml ingest.split_inconsistency_threshold.
# ---------------------------------------------------------------------------

DEFAULT_SPLIT_INCONSISTENCY_THRESHOLD = 0.20
_SPLIT_INCONSISTENCY_LOOKBACK_SESSIONS = 400


def _check_no_split_inconsistency(
    store: Store,
    asof: date,
    *,
    threshold: float = DEFAULT_SPLIT_INCONSISTENCY_THRESHOLD,
    sessions: int = _SPLIT_INCONSISTENCY_LOOKBACK_SESSIONS,
) -> QualityCheck:
    from sma.ingest.split_audit import (
        DEFAULT_SOLO_THRESHOLD,
        find_split_inconsistencies,
        trailing_session_start,
    )

    since = trailing_session_start(store.conn, asof=asof, sessions=sessions)
    if since is None:
        return QualityCheck(
            name="no_split_inconsistency", passed=True, detail="no price rows yet",
            blocking=False,
        )
    flags = [
        f for f in find_split_inconsistencies(
            store.conn,
            threshold=threshold,
            solo_threshold=max(DEFAULT_SOLO_THRESHOLD, threshold),
            since=since,
            until=asof,
        )
        if f.feature_affecting
    ]
    tickers = sorted({f.ticker for f in flags})
    if not flags:
        detail = (
            f"yfinance vs alpaca returns consistent within {threshold:.0%} "
            f"over {sessions} sessions (since {since})"
        )
    else:
        detail = (
            f"split-inconsistent tickers={tickers}: "
            f"{[f.describe() for f in flags[:10]]}{'...' if len(flags) > 10 else ''} "
            f"-- fix: python -m sma.ingest repair-splits --all-flagged"
        )
    return QualityCheck(
        name="no_split_inconsistency",
        passed=True,  # never blocks -- pages instead (see block comment above)
        detail=detail,
        blocking=False,
        degraded=bool(flags),
    )


def notify_new_dead_or_frozen_tickers(
    flags: list[DeadOrFrozenFlag],
    *,
    asof: date,
    notify_fn=notify_failure,
) -> list[str]:
    """Page once per ticker for a NEW no_dead_or_frozen_tickers flag, deduped
    via sma.ingest.dead_ticker_state (same one-small-overwritten-file
    convention as sma.monitoring.regime_state) so a dead ticker pages once,
    not every night it stays dead. A ticker that heals (missing from `flags`)
    is dropped from state, so a later recurrence pages again instead of
    staying silent off a stale record.

    Deliberately NOT called from _check_no_dead_or_frozen_tickers or
    run_quality_checks -- those must stay side-effect-free so tests,
    backtests, and the dashboard can call them freely. Call this separately
    from the ingest CLI (sma.ingest.__main__), after the quality report is
    computed.

    Never raises -- a notification bug must not break the ingest run this
    fires from. Returns the tickers notified THIS call (for logging/tests).
    """
    from sma.ingest.dead_ticker_state import (
        read_dead_ticker_state,
        write_dead_ticker_state,
    )

    try:
        prior = read_dead_ticker_state()
    except Exception:
        prior = {}

    new_state: dict = {}
    notified: list[str] = []
    for flag in flags:
        prior_entry = prior.get(flag.ticker)
        is_new = prior_entry is None or prior_entry.get("kind") != flag.kind
        if is_new:
            notified.append(flag.ticker)
            last_price = flag.last_price_date.isoformat() if flag.last_price_date else "never"
            if flag.kind == "frozen":
                run_desc = f"{flag.run_length} identical closes in a row"
            else:
                run_desc = f"no price row for the last {flag.run_length} sessions"
            with contextlib.suppress(Exception):
                notify_fn(
                    title=f"sma: {flag.kind.upper()} ticker {flag.ticker}",
                    message=(
                        f"{flag.ticker} looks {flag.kind} as of {asof.isoformat()} "
                        f"(no_dead_or_frozen_tickers, non-blocking): last price "
                        f"{last_price}, {run_desc}. Check for a delisting/merger/halt."
                    ),
                )
        new_state[flag.ticker] = {
            "kind": flag.kind,
            "last_price_date": (flag.last_price_date.isoformat() if flag.last_price_date else None),
            "run_length": flag.run_length,
            "first_flagged_asof": (
                asof.isoformat()
                if is_new
                else prior_entry.get("first_flagged_asof", asof.isoformat())
            ),
        }

    with contextlib.suppress(Exception):
        write_dead_ticker_state(new_state)

    return notified


def notify_degraded_quality_checks(
    report: QualityReport,
    *,
    asof: date,
    notify_fn=notify_failure,
) -> list[str]:
    """Page once for every check that PASSED only via a tolerance/fallback
    path (QualityCheck.degraded) — e.g. all_tickers_have_price tolerating a
    few missing names, or SPY's adj_close falling back to a prior session.

    Deliberately NOT called from run_quality_checks itself, same reason as
    notify_new_dead_or_frozen_tickers: the checks must stay side-effect-free
    so tests, backtests, and the dashboard can call them freely. Call this
    separately from the ingest CLI once the quality report is computed.

    No per-check dedup (unlike the per-ticker dead/frozen state): a degraded
    quality check is a per-run signal, not a persistent per-ticker condition,
    and it self-clears the moment the underlying data gap is gone — paging
    again each day it recurs is the point, not noise.

    Never raises -- a notification bug must not break the ingest run this
    fires from. Returns the check names notified (for logging/tests).
    """
    notified: list[str] = []
    for check in report.checks:
        if check.passed and check.degraded:
            notified.append(check.name)
            with contextlib.suppress(Exception):
                notify_fn(
                    title=f"sma: {check.name} degraded (non-blocking)",
                    message=f"{asof.isoformat()}: {check.detail}",
                )
    return notified


def run_quality_checks(
    store: Store,
    asof_date: date,
    universe: list[str],
    run_id: int,
    *,
    split_threshold: float = DEFAULT_SPLIT_INCONSISTENCY_THRESHOLD,
) -> QualityReport:
    checks = [
        _check_all_tickers_have_price(store, asof_date, universe, run_id),
        _check_adjusted_price_coverage(store, asof_date, universe),
        _check_no_unjustified_extreme_moves(store, asof_date),
        _check_news_table_grew(store, asof_date, run_id),
        _check_no_excessive_nulls(store, run_id),
        _check_enough_sources_succeeded(store, run_id),
        _check_price_divergence(store, asof_date),
        _check_news_per_ticker_minimum(store, asof_date, universe),
        _check_earnings_coverage(store, asof_date, universe),
        _check_theses_freshness(store, asof_date, universe),
        _check_agents_last_run_healthy(store, asof_date),
        _check_no_dead_or_frozen_tickers(store, asof_date, universe),
        _check_no_nan_prices(store, asof_date),
        _check_no_split_inconsistency(store, asof_date, threshold=split_threshold),
    ]
    return QualityReport(asof_date=asof_date, run_id=run_id, checks=checks)
