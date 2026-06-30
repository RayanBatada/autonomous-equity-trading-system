from datetime import date, datetime, timedelta

import pytest

from sma.ingest.quality import (
    _check_earnings_coverage,
    _check_news_per_ticker_minimum,
    _check_theses_freshness,
    run_quality_checks,
)
from sma.ingest.store import Store


@pytest.fixture
def store():
    s = Store(":memory:").connect()
    yield s
    s.close()


def _seed_minimum_passing_data(store: Store, asof: date, tickers: list[str], rid: int):
    for src in ["yfinance", "alpaca", "finnhub_news", "finnhub_sentiment",
                "finnhub_fundamentals", "newsapi", "edgar"]:
        store.log_run_start(rid, source=src)
        store.log_run_end(rid, source=src, rows_inserted=10, status="ok", error=None)
    for t in tickers:
        store.conn.execute(
            "INSERT INTO prices VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [t, asof, 100.0, 102.0, 99.0, 101.0, 101.0, 1_000_000, "yfinance", rid],
        )
    for i in range(11):
        h = f"hash-{i}"
        store.conn.execute(
            "INSERT INTO news VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [tickers[0], datetime.combine(asof, datetime.min.time()),
             "finnhub", f"head {i}", f"https://e/{i}", None, h, rid],
        )
    # Seed >=1 news article and >=1 earnings row per non-SPY ticker so the
    # new per-ticker coverage checks pass on a "minimum healthy" run.
    for j, t in enumerate(tickers):
        if t == "SPY":
            continue
        store.conn.execute(
            "INSERT INTO news VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [t, datetime.combine(asof, datetime.min.time()),
             "alpaca_news", f"per-ticker {t}",
             f"https://e/per-ticker/{t}", None, f"hash-pt-{j}", rid],
        )
        store.conn.execute(
            "INSERT INTO earnings VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [t, asof, 1.0, 1.1, 1_000_000.0, 1_100_000.0, "finnhub", rid],
        )
        # Seed a fresh thesis so the new theses_freshness check passes
        # on a "minimum healthy" run.
        store.conn.execute(
            """
            INSERT INTO theses (ticker, asof_date, run_id, news_summary,
                key_developments, notable_filings, bull_case, bear_case,
                asymmetric_risks, catalyst_window,
                conviction, score, flags, action_hint, reasoning)
            VALUES (?, ?, ?, '', '[]', '[]', '', '', '[]', 'far',
                    'neutral', 0.0, '[]', 'hold', '')
            """,
            [t, asof, rid],
        )


def test_quality_report_passes_with_minimum_data(store):
    asof = date(2026, 4, 23)  # Thursday
    tickers = ["AAPL", "MSFT"]
    rid = store.allocate_run_id()
    _seed_minimum_passing_data(store, asof, tickers, rid)

    report = run_quality_checks(store, asof_date=asof, universe=tickers, run_id=rid)
    assert report.passed, report.summary()


def test_quality_fails_when_ticker_missing_price(store):
    asof = date(2026, 4, 23)
    tickers = ["AAPL", "MSFT"]
    rid = store.allocate_run_id()
    _seed_minimum_passing_data(store, asof, ["AAPL"], rid)

    report = run_quality_checks(store, asof_date=asof, universe=tickers, run_id=rid)
    assert not report.passed
    assert any(c.name == "all_tickers_have_price" and not c.passed for c in report.checks)


def test_quality_fails_on_extreme_price_move_without_filing(store):
    asof = date(2026, 4, 23)
    tickers = ["AAPL"]
    rid = store.allocate_run_id()
    _seed_minimum_passing_data(store, asof, tickers, rid)
    store.conn.execute(
        "INSERT INTO prices VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ["AAPL", asof - timedelta(days=1), 220.0, 225.0, 219.0, 222.0, 222.0,
         1_000_000, "yfinance", rid],
    )

    report = run_quality_checks(store, asof_date=asof, universe=tickers, run_id=rid)
    assert not report.passed
    assert any(c.name == "no_unjustified_extreme_moves" and not c.passed for c in report.checks)


def test_quality_fails_when_news_table_empty_on_weekday(store):
    asof = date(2026, 4, 23)
    tickers = ["AAPL"]
    rid = store.allocate_run_id()
    for src in ["yfinance", "alpaca", "finnhub_news", "finnhub_sentiment",
                "finnhub_fundamentals", "newsapi", "edgar"]:
        store.log_run_start(rid, source=src)
        store.log_run_end(rid, source=src, rows_inserted=10, status="ok", error=None)
    store.conn.execute(
        "INSERT INTO prices VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ["AAPL", asof, 100.0, 102.0, 99.0, 101.0, 101.0, 1_000_000, "yfinance", rid],
    )

    report = run_quality_checks(store, asof_date=asof, universe=tickers, run_id=rid)
    assert not report.passed
    assert any(c.name == "news_table_grew" and not c.passed for c in report.checks)


def test_quality_fails_when_no_price_source_succeeded(store):
    """A run where BOTH price sources fail has no tradable data and must block,
    even if the news/overlay sources are all 'ok' (the prior 'any 5' rule let
    that through)."""
    asof = date(2026, 4, 23)
    tickers = ["AAPL"]
    rid = store.allocate_run_id()
    for src in ["yfinance", "alpaca"]:  # both PRICE sources fail
        store.log_run_start(rid, source=src)
        store.log_run_end(rid, source=src, rows_inserted=0, status="error", error="boom")
    for src in ["finnhub_news", "finnhub_sentiment", "finnhub_fundamentals", "newsapi", "edgar"]:
        store.log_run_start(rid, source=src)
        store.log_run_end(rid, source=src, rows_inserted=10, status="ok", error=None)
    store.conn.execute(
        "INSERT INTO prices VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ["AAPL", asof, 100.0, 102.0, 99.0, 101.0, 101.0, 1_000_000, "yfinance", rid],
    )
    for i in range(11):
        store.conn.execute(
            "INSERT INTO news VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ["AAPL", datetime.combine(asof, datetime.min.time()),
             "finnhub", f"h{i}", f"https://e/{i}", None, f"hash{i}", rid],
        )

    report = run_quality_checks(store, asof_date=asof, universe=tickers, run_id=rid)
    assert not report.passed
    assert any(c.name == "enough_sources_succeeded" and not c.passed for c in report.checks)


def test_null_check_ignores_stale_rows_from_prior_runs(store):
    """A past run with NULL close values shouldn't make today's check fail."""
    asof = date(2026, 4, 23)
    tickers = ["AAPL"]

    # Past run: many rows with NULL close (e.g., a partial ingest from days ago).
    # Each row needs a unique (ticker, date, source) to satisfy the PK.
    old_rid = store.allocate_run_id()
    for i in range(50):
        store.conn.execute(
            "INSERT INTO prices VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ["AAPL", date(2026, 1, 1) + timedelta(days=i), 100.0, 102.0, 99.0,
             None, 101.0, 1_000_000, "alpaca", old_rid],
        )

    # Today's run: clean rows
    rid = store.allocate_run_id()
    _seed_minimum_passing_data(store, asof, tickers, rid)

    report = run_quality_checks(store, asof_date=asof, universe=tickers, run_id=rid)
    null_check = next(c for c in report.checks if c.name == "no_excessive_nulls")
    assert null_check.passed, (
        f"null check should not flag stale rows from prior runs, but got: {null_check.detail}"
    )


def test_null_check_flags_excessive_nulls_in_current_run(store):
    """If THIS run wrote rows with NULL close above the threshold, the check fires."""
    asof = date(2026, 4, 23)
    tickers = ["AAPL"]
    rid = store.allocate_run_id()
    _seed_minimum_passing_data(store, asof, tickers, rid)

    # Add 20 NULL-close rows from this same run, well above the 5% threshold.
    # Use the alpaca source so we don't collide with the seeded yfinance row.
    for i in range(20):
        store.conn.execute(
            "INSERT INTO prices VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ["AAPL", date(2026, 3, 1) + timedelta(days=i), 100.0, 102.0, 99.0,
             None, 101.0, 1_000_000, "alpaca", rid],
        )

    report = run_quality_checks(store, asof_date=asof, universe=tickers, run_id=rid)
    null_check = next(c for c in report.checks if c.name == "no_excessive_nulls")
    assert not null_check.passed, "null check should flag excessive nulls in current run"
    assert "prices" in null_check.detail and "close" in null_check.detail


def test_news_per_ticker_minimum_pass(store):
    asof = date(2026, 4, 26)
    rid = store.allocate_run_id()
    # Insert news for AAPL and MSFT within the last 30 days. SPY has no news
    # but should still pass because the check excludes SPY.
    for i, t in enumerate(["AAPL", "MSFT"]):
        store.conn.execute(
            "INSERT INTO news VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [t, datetime(2026, 4, 20), "alpaca_news", f"head {t}",
             f"https://e/{t}", None, f"hash-{i}", rid],
        )

    check = _check_news_per_ticker_minimum(store, asof, ["AAPL", "MSFT", "SPY"])
    assert check.name == "news_per_ticker_minimum"
    assert check.passed, check.detail


def test_news_per_ticker_minimum_fail(store):
    asof = date(2026, 4, 26)
    rid = store.allocate_run_id()
    # Only AAPL has news in window; MSFT is missing.
    store.conn.execute(
        "INSERT INTO news VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ["AAPL", datetime(2026, 4, 20), "alpaca_news", "head AAPL",
         "https://e/AAPL", None, "hash-aapl", rid],
    )
    # Stale row for MSFT outside the 30-day window — should not satisfy the check.
    store.conn.execute(
        "INSERT INTO news VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ["MSFT", datetime(2025, 12, 1), "alpaca_news", "head MSFT old",
         "https://e/MSFT-old", None, "hash-msft-old", rid],
    )

    check = _check_news_per_ticker_minimum(store, asof, ["AAPL", "MSFT"])
    assert check.name == "news_per_ticker_minimum"
    assert not check.passed
    assert "MSFT" in check.detail


def test_earnings_coverage_pass(store):
    asof = date(2026, 4, 26)
    rid = store.allocate_run_id()
    for t in ["AAPL", "MSFT"]:
        store.conn.execute(
            "INSERT INTO earnings VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [t, date(2026, 4, 1), 1.0, 1.1, 1_000_000.0, 1_100_000.0,
             "finnhub", rid],
        )

    check = _check_earnings_coverage(store, asof, ["AAPL", "MSFT", "SPY"])
    assert check.name == "earnings_coverage"
    assert check.passed, check.detail


def test_earnings_coverage_fail(store):
    asof = date(2026, 4, 26)
    rid = store.allocate_run_id()
    # Only AAPL has earnings in window.
    store.conn.execute(
        "INSERT INTO earnings VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ["AAPL", date(2026, 4, 1), 1.0, 1.1, 1_000_000.0, 1_100_000.0,
         "finnhub", rid],
    )
    # Stale TSLA earnings outside the 12-month window — shouldn't count.
    store.conn.execute(
        "INSERT INTO earnings VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ["TSLA", date(2024, 12, 1), 1.0, 1.1, 1_000_000.0, 1_100_000.0,
         "finnhub", rid],
    )

    check = _check_earnings_coverage(store, asof, ["AAPL", "MSFT", "TSLA"])
    assert check.name == "earnings_coverage"
    assert not check.passed
    assert "MSFT" in check.detail
    assert "TSLA" in check.detail


def _insert_thesis(store, ticker, asof_date):
    store.conn.execute(
        """
        INSERT INTO theses (ticker, asof_date, run_id, news_summary,
            key_developments, notable_filings, bull_case, bear_case,
            asymmetric_risks, catalyst_window,
            conviction, score, flags, action_hint, reasoning)
        VALUES (?, ?, 1, '', '[]', '[]', '', '', '[]', 'far',
                'neutral', 0.0, '[]', 'hold', '')
        """,
        [ticker, asof_date],
    )


def test_theses_freshness_pass_case(store):
    """Every non-SPY ticker has a thesis within last 7 days → passes.

    SPY is an ETF and is excluded from the check, so even without a thesis SPY won't fail it.
    """
    universe = ["AAPL", "MSFT", "SPY"]
    asof = date(2026, 4, 26)
    _insert_thesis(store, "AAPL", asof)
    _insert_thesis(store, "MSFT", asof)
    check = _check_theses_freshness(store, asof, universe)
    assert check.passed is True
    assert "all theses within 7 days" in check.detail


def test_theses_freshness_fail_case(store):
    """A non-SPY ticker with no thesis (or only stale theses) fails the check."""
    universe = ["AAPL", "MSFT", "SPY"]
    asof = date(2026, 4, 26)
    _insert_thesis(store, "AAPL", asof)  # fresh
    _insert_thesis(store, "MSFT", date(2026, 4, 15))  # stale (>7 days)
    check = _check_theses_freshness(store, asof, universe)
    assert check.passed is False
    assert "MSFT" in check.detail


def test_theses_freshness_excludes_spy(store):
    """SPY without a thesis does not cause failure."""
    universe = ["AAPL", "SPY"]
    asof = date(2026, 4, 26)
    _insert_thesis(store, "AAPL", asof)
    check = _check_theses_freshness(store, asof, universe)
    assert check.passed is True


def test_theses_freshness_missing_ticker_entirely(store):
    """A non-SPY ticker with NO thesis at all fails."""
    universe = ["AAPL", "MSFT"]
    asof = date(2026, 4, 26)
    _insert_thesis(store, "AAPL", asof)  # only AAPL seeded
    check = _check_theses_freshness(store, asof, universe)
    assert check.passed is False
    assert "MSFT" in check.detail


def test_run_quality_checks_includes_theses_freshness(store):
    """The wiring: theses_freshness shows up in run_quality_checks output."""
    from sma.ingest.quality import run_quality_checks
    universe = ["AAPL", "SPY"]
    asof = date(2026, 4, 26)
    _insert_thesis(store, "AAPL", asof)
    # Need a run_id; use 1.
    report = run_quality_checks(store, asof_date=asof, universe=universe, run_id=1)
    names = [c.name for c in report.checks]
    assert "theses_freshness" in names


def test_blocking_failures_excludes_non_blocking_checks():
    """A failed OVERLAY check (blocking=False) must NOT gate decide. Regression:
    pre-fix, a flaky news day (news_table_grew) blocked ALL live trading."""
    from datetime import date

    from sma.ingest.quality import QualityCheck, QualityReport
    checks = [
        QualityCheck("all_tickers_have_price", passed=False, detail="", blocking=True),
        QualityCheck("news_table_grew", passed=False, detail="", blocking=False),
        QualityCheck("earnings_coverage", passed=False, detail="", blocking=False),
    ]
    report = QualityReport(asof_date=date(2026, 1, 1), run_id=1, checks=checks)
    assert report.blocking_failures() == ["all_tickers_have_price"]


def test_blocking_failures_respects_waivers():
    from datetime import date

    from sma.ingest.quality import QualityCheck, QualityReport
    checks = [QualityCheck("all_tickers_have_price", passed=False, detail="", blocking=True)]
    report = QualityReport(asof_date=date(2026, 1, 1), run_id=1, checks=checks)
    assert report.blocking_failures(frozenset({"all_tickers_have_price"})) == []


# ---- 2026-06-05 audit: ingest quality-gate holes -----------------------------


def test_all_tickers_have_price_is_scoped_to_run_id():
    """Coverage must count only the CURRENT run's prices — stale same-date rows
    from an earlier run must not make a zero-price outage run pass."""
    from sma.ingest.quality import _check_all_tickers_have_price

    s = Store(":memory:").connect()
    asof = date(2026, 5, 1)
    for t in ["AAPL", "SPY"]:  # present, but under run_id=1
        s.conn.execute(
            "INSERT INTO prices VALUES (?, ?, 1,1,1,1, 1.0, 100, 'yfinance', ?)",
            [t, asof, 1],
        )
    # The current run (2) inserted nothing.
    assert not _check_all_tickers_have_price(s, asof, ["AAPL", "SPY"], 2).passed
    assert _check_all_tickers_have_price(s, asof, ["AAPL", "SPY"], 1).passed


def test_all_tickers_have_price_requires_spy_adj_close():
    """SPY (rel_strength_spy) must have a current-run row with a usable
    (non-null) adj_close — an Alpaca-only null-adj_close SPY row is not enough."""
    from sma.ingest.quality import _check_all_tickers_have_price

    s = Store(":memory:").connect()
    asof = date(2026, 5, 1)
    rid = s.allocate_run_id()
    s.conn.execute(
        "INSERT INTO prices VALUES ('AAPL', ?, 1,1,1,1, 1.0, 100, 'yfinance', ?)", [asof, rid]
    )
    s.conn.execute(
        "INSERT INTO prices VALUES ('SPY', ?, 1,1,1,1, NULL, 100, 'alpaca', ?)", [asof, rid]
    )
    assert not _check_all_tickers_have_price(s, asof, ["AAPL", "SPY"], rid).passed


def test_enough_sources_requires_price_source_with_rows():
    """A price source with status='ok' but rows_inserted=0 (empty fetch) is NOT
    tradable data and must not satisfy the price-source gate."""
    from sma.ingest.quality import _check_enough_sources_succeeded

    s = Store(":memory:").connect()
    rid = s.allocate_run_id()
    s.log_run_start(rid, source="yfinance")
    s.log_run_end(rid, source="yfinance", rows_inserted=0, status="ok", error=None)
    s.log_run_start(rid, source="alpaca")
    s.log_run_end(rid, source="alpaca", rows_inserted=0, status="error", error="boom")
    assert not _check_enough_sources_succeeded(s, rid).passed


def test_price_divergence_compares_null_adj_close_sources(store):
    """audit quality.py:218 — alpaca writes adj_close=NULL (1127/1315 rows in
    prod), so the divergence check's `adj_close IS NOT NULL` filter dropped
    every alpaca row: nsrc was always 1 and sources were NEVER compared.
    COALESCE(adj_close, close) is safe inside the 7-day window (a week's
    dividend drift is far under the 2% threshold)."""
    from sma.ingest.quality import _check_price_divergence

    asof = date(2026, 4, 30)
    rid = store.allocate_run_id()
    store.conn.execute(
        "INSERT INTO prices VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ["AAPL", asof, 100.0, 102.0, 99.0, 101.0, 101.0, 1_000_000, "yfinance", rid],
    )
    # alpaca disagrees by ~9% and has NULL adj_close (its placeholder)
    store.conn.execute(
        "INSERT INTO prices VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ["AAPL", asof, 100.0, 112.0, 99.0, 110.0, None, 1_000_000, "alpaca", rid],
    )
    check = _check_price_divergence(store, asof)
    assert check.passed is False, (
        "a 9% cross-source disagreement must be flagged even when one source "
        "stores adj_close as NULL"
    )
