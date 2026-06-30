"""Post-run quality checks.

Six SQL assertions per the spec. Each returns a QualityCheck. The aggregate
QualityReport is `passed` iff all checks passed. Result is also written to
logs/quality/YYYY-MM-DD.txt by the runner.
"""

from dataclasses import dataclass
from datetime import date

from sma.ingest.store import Store

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

# Tickers exempt from per-ticker news coverage. ETFs report on their
# holdings, not the fund itself, so per-ticker news feeds usually return
# nothing for them. SPY has always been excluded; NANC (Unusual Whales
# Subversive Democratic Trading ETF) was added 2026-05-14 after the
# universe expansion surfaced it as a perpetual failure.
_NO_NEWS_EXEMPT: frozenset[str] = frozenset({"SPY", "NANC"})

# Tickers exempt from quarterly-earnings coverage. ETFs report no per-fund
# earnings; BRK.B reports annually only (Berkshire Hathaway does not file
# quarterly), so a 12-month quarterly window legitimately misses it on
# Finnhub. Other gaps indicate a real ingest problem and should still fail.
_NO_EARNINGS_EXEMPT: frozenset[str] = frozenset({"SPY", "NANC", "BRK.B"})


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

    def summary(self) -> str:
        lines = [f"Quality report for {self.asof_date} (run_id={self.run_id})", ""]
        for c in self.checks:
            mark = "PASS" if c.passed else "FAIL"
            lines.append(f"  [{mark}] {c.name}: {c.detail}")
        lines.append("")
        lines.append("OVERALL: " + ("PASS" if self.passed else "FAIL"))
        return "\n".join(lines)


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
    spy_ok = (not spy_needed) or store.conn.execute(
        "SELECT 1 FROM prices WHERE ticker = 'SPY' AND date = ? AND run_id = ? "
        "AND adj_close IS NOT NULL LIMIT 1",
        [asof, run_id],
    ).fetchone() is not None
    return QualityCheck(
        name="all_tickers_have_price",
        # Tolerate a handful of missing names (thin tickers / a slow fetch) — the
        # predictor drops missing-price names gracefully, so decide can safely
        # trade the rest. Block only on a real outage (coverage < 97%) OR if SPY
        # has no usable price.
        passed=coverage >= 0.97 and spy_ok,
        detail=(
            f"coverage={coverage:.1%} spy_ok={spy_ok} "
            f"missing={missing[:10]}{'...' if len(missing) > 10 else ''}"
            if missing or not spy_ok else "all present"
        ),
    )


def _check_no_unjustified_extreme_moves(store: Store, asof: date) -> QualityCheck:
    # Restrict to yfinance because it's the only source that actually applies
    # historical split-adjustment to adj_close. Alpaca's adj_close is a
    # placeholder (= close), so its rows show artificial 50%+ drops on every
    # split day, which would be a constant stream of false positives.
    suspect = store.conn.execute("""
        WITH p AS (
            SELECT ticker, date, adj_close,
                   LAG(adj_close) OVER (PARTITION BY ticker ORDER BY date) AS prev
            FROM prices
            WHERE date >= ? - INTERVAL '30 days'
              AND adj_close IS NOT NULL
              AND source = 'yfinance'
        )
        SELECT p.ticker, p.date,
               ABS(p.adj_close - p.prev) / NULLIF(p.prev, 0) AS move,
               EXISTS (
                   SELECT 1 FROM filings f
                   WHERE f.ticker = p.ticker
                     AND f.filing_type = '8-K'
                     AND ABS(EXTRACT(EPOCH FROM (f.filed_at - p.date::TIMESTAMP)) / 86400) <= 2
               ) AS has_8k
        FROM p
        WHERE p.prev IS NOT NULL
          AND ABS(p.adj_close - p.prev) / NULLIF(p.prev, 0) > 0.5
    """, [asof]).fetchall()
    bad = [r for r in suspect if not r[3]]
    return QualityCheck(
        name="no_unjustified_extreme_moves",
        passed=len(bad) == 0,
        detail=f"flagged_without_8k={[(r[0], r[1], round(r[2], 3)) for r in bad[:5]]}"
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
    bad = store.conn.execute("""
        WITH s AS (
            SELECT ticker, date,
                   MAX(COALESCE(adj_close, close)) AS hi,
                   MIN(COALESCE(adj_close, close)) AS lo,
                   COUNT(DISTINCT source) AS nsrc
            FROM prices
            WHERE date >= ? - INTERVAL '7 days'
              AND COALESCE(adj_close, close) IS NOT NULL
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

    ETFs report no per-ticker earnings; BRK.B reports annually so a 12-month
    quarterly window may legitimately miss it. Exclude these so the check
    stays signal-rich. Other gaps usually indicate a Finnhub quota / API
    flakiness issue and are fixable via `backfill-earnings`.
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
    """Every non-SPY ticker should have a thesis dated within the last 7 calendar days.

    Phase 4 refreshes weekly + on event triggers. Stale-beyond-7-days means the Friday
    job didn't run OR an event trigger missed something — both worth surfacing.
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
    stale = sorted([
        t for t in universe
        if t != "SPY" and (have.get(t) is None or (asof - have[t]).days > 7)
    ])
    return QualityCheck(
        name="theses_freshness",
        passed=len(stale) == 0,
        detail=f"stale={stale[:10]}" if stale else "all theses within 7 days",
        blocking=False,  # agent-thesis freshness is gated by the agents sentinel,
        # not the ingest gate — recording here is fine, but it must not block decide
    )


def run_quality_checks(
    store: Store,
    asof_date: date,
    universe: list[str],
    run_id: int,
) -> QualityReport:
    checks = [
        _check_all_tickers_have_price(store, asof_date, universe, run_id),
        _check_no_unjustified_extreme_moves(store, asof_date),
        _check_news_table_grew(store, asof_date, run_id),
        _check_no_excessive_nulls(store, run_id),
        _check_enough_sources_succeeded(store, run_id),
        _check_price_divergence(store, asof_date),
        _check_news_per_ticker_minimum(store, asof_date, universe),
        _check_earnings_coverage(store, asof_date, universe),
        _check_theses_freshness(store, asof_date, universe),
    ]
    return QualityReport(asof_date=asof_date, run_id=run_id, checks=checks)
