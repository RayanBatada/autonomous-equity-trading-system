from datetime import date, datetime, timedelta

import pytest

from sma.ingest.quality import (
    DeadOrFrozenFlag,
    _check_earnings_coverage,
    _check_news_per_ticker_minimum,
    _check_no_dead_or_frozen_tickers,
    _check_no_unjustified_extreme_moves,
    _check_theses_freshness,
    find_dead_or_frozen_tickers,
    notify_new_dead_or_frozen_tickers,
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


def test_extreme_move_split_desync_not_exempted_by_coincident_8k(store):
    """A >50% adj_close jump that raw `close` does NOT corroborate is a split-
    adjustment desync (the KLAC 10x bug), not a real price move — and splits/
    spinoffs file their own 8-Ks on the ex-date, so a bare coincident 8-K must
    NOT exempt it. Regression: the gate used to waive ANY move near an 8-K,
    reopening the exact corruption it exists to catch."""
    asof = date(2026, 4, 23)
    tickers = ["AAPL"]
    rid = store.allocate_run_id()
    _seed_minimum_passing_data(store, asof, tickers, rid)
    # KLAC adj_close craters 200 -> 20 (10x desync artifact) while raw close is
    # smooth (200 -> 198): the boundary is an adjustment-basis mismatch, not a
    # real price move.
    store.conn.execute(
        "INSERT INTO prices VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ["KLAC", asof - timedelta(days=1), 200.0, 205.0, 199.0, 200.0, 200.0,
         1_000_000, "yfinance", rid],
    )
    store.conn.execute(
        "INSERT INTO prices VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ["KLAC", asof, 198.0, 205.0, 19.0, 198.0, 20.0, 1_000_000, "yfinance", rid],
    )
    # The split's own 8-K, filed on the ex-date — must NOT waive the desync.
    store.conn.execute(
        "INSERT INTO filings VALUES (?, ?, ?, ?, ?, ?, ?)",
        ["KLAC", "8-K", datetime.combine(asof, datetime.min.time()),
         "acc-split", "https://e/8k", None, rid],
    )

    report = run_quality_checks(store, asof_date=asof, universe=tickers, run_id=rid)
    check = next(c for c in report.checks if c.name == "no_unjustified_extreme_moves")
    assert not check.passed, f"desync should be flagged despite the 8-K: {check.detail}"


def test_extreme_move_real_event_with_8k_is_exempted(store):
    """A genuine >50% move corroborated by BOTH adj_close AND raw close (a real
    crash / buyout, which is a real 8-K event) is still exempted — the raw-close
    cross-check must not turn legitimate exemptions into false positives."""
    asof = date(2026, 4, 23)
    tickers = ["AAPL"]
    rid = store.allocate_run_id()
    _seed_minimum_passing_data(store, asof, tickers, rid)
    # MRNA gaps down ~55% on real news: adj_close AND raw close move together.
    store.conn.execute(
        "INSERT INTO prices VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ["MRNA", asof - timedelta(days=1), 200.0, 205.0, 199.0, 200.0, 200.0,
         1_000_000, "yfinance", rid],
    )
    store.conn.execute(
        "INSERT INTO prices VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ["MRNA", asof, 92.0, 95.0, 88.0, 90.0, 90.0, 1_000_000, "yfinance", rid],
    )
    store.conn.execute(
        "INSERT INTO filings VALUES (?, ?, ?, ?, ?, ?, ?)",
        ["MRNA", "8-K", datetime.combine(asof, datetime.min.time()),
         "acc-real", "https://e/8k", None, rid],
    )

    report = run_quality_checks(store, asof_date=asof, universe=tickers, run_id=rid)
    check = next(c for c in report.checks if c.name == "no_unjustified_extreme_moves")
    assert check.passed, f"a real 8-K-backed move should be exempted: {check.detail}"


# ---------------------------------------------------------------------------
# Cross-source confirmation exemption (2026-08-24).
#
# The gate exists to catch SINGLE-SOURCE corruption (a bad fetch, a split
# mis-adjustment in one vendor). When two independent vendors each show the
# same huge move and agree on the actual price levels, that is market reality,
# not corruption — evidence strictly stronger than the 8-K heuristic. See the
# comment on _check_no_unjustified_extreme_moves.
# ---------------------------------------------------------------------------


def _price(store, ticker, day, px, source, rid, *, adj=True):
    """Insert one price row. adj=False mirrors alpaca, which stores adj_close NULL."""
    store.conn.execute(
        "INSERT INTO prices VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [ticker, day, px, px * 1.01, px * 0.99, px, (px if adj else None),
         1_000_000, source, rid],
    )


def test_extreme_move_from_a_single_source_still_fails(store):
    """The guard's whole purpose: one vendor spiking alone is corruption until
    proven otherwise. No second source, no 8-K -> FAIL."""
    asof = date(2026, 4, 23)
    rid = store.allocate_run_id()
    _seed_minimum_passing_data(store, asof, ["AAPL"], rid)
    # Only yfinance has ZZZZ, and it triples overnight.
    _price(store, "ZZZZ", asof - timedelta(days=1), 100.0, "yfinance", rid)
    _price(store, "ZZZZ", asof, 300.0, "yfinance", rid)

    check = next(
        c for c in run_quality_checks(store, asof_date=asof, universe=["AAPL"], run_id=rid).checks
        if c.name == "no_unjustified_extreme_moves"
    )
    assert not check.passed, f"a single-source spike must still be flagged: {check.detail}"
    assert "ZZZZ" in check.detail


def test_extreme_move_confirmed_by_two_sources_is_exempt(store):
    """Two independent vendors showing the SAME huge move, agreeing on the price
    level on both days, is market reality — exempt even with no 8-K on file."""
    asof = date(2026, 4, 23)
    rid = store.allocate_run_id()
    _seed_minimum_passing_data(store, asof, ["AAPL"], rid)
    prev = asof - timedelta(days=1)
    _price(store, "ZZZZ", prev, 100.0, "yfinance", rid)
    _price(store, "ZZZZ", asof, 300.0, "yfinance", rid)
    # alpaca agrees within 2% on BOTH days, and stores adj_close as NULL.
    _price(store, "ZZZZ", prev, 99.9, "alpaca", rid, adj=False)
    _price(store, "ZZZZ", asof, 299.5, "alpaca", rid, adj=False)

    check = next(
        c for c in run_quality_checks(store, asof_date=asof, universe=["AAPL"], run_id=rid).checks
        if c.name == "no_unjustified_extreme_moves"
    )
    assert check.passed, f"a cross-source-confirmed move should be exempt: {check.detail}"


def test_extreme_move_with_disagreeing_sources_still_fails(store):
    """One vendor spikes, the other does not: that is exactly the single-source
    corruption signature. No exemption."""
    asof = date(2026, 4, 23)
    rid = store.allocate_run_id()
    _seed_minimum_passing_data(store, asof, ["AAPL"], rid)
    prev = asof - timedelta(days=1)
    _price(store, "ZZZZ", prev, 100.0, "yfinance", rid)
    _price(store, "ZZZZ", asof, 300.0, "yfinance", rid)
    # alpaca sees a normal day — it never confirms the spike.
    _price(store, "ZZZZ", prev, 100.0, "alpaca", rid, adj=False)
    _price(store, "ZZZZ", asof, 101.0, "alpaca", rid, adj=False)

    check = next(
        c for c in run_quality_checks(store, asof_date=asof, universe=["AAPL"], run_id=rid).checks
        if c.name == "no_unjustified_extreme_moves"
    )
    assert not check.passed, f"disagreeing sources must not exempt: {check.detail}"
    assert "ZZZZ" in check.detail


# Exact production rows for MRNA, read out of data/sma.duckdb on 2026-08-24.
# The +177% on 8/19 ($62.93 -> $174.27) is real and present IDENTICALLY in both
# vendors; MRNA has filed no 8-K since 8/1. Under the 8-K-only rule this single
# row failed the gate on every ingest from 8/19 on, and — because predict and
# decide gate on it via the ingest sentinel — the bot stopped trading for six
# sessions while holding MRNA at 22% of book.
_MRNA_PROD_ROWS = [
    # (date, alpaca close, yfinance close/adj_close)
    (date(2026, 8, 14), 63.285, 63.31999969482422),
    (date(2026, 8, 17), 64.45, 64.45999908447266),
    (date(2026, 8, 18), 62.93, 62.959999084472656),
    (date(2026, 8, 19), 174.27, 174.3800048828125),
    (date(2026, 8, 20), 133.23, 133.32000732421875),
]


def test_mrna_2026_08_19_passes_on_a_production_shaped_db(tmp_path):
    """DB-backed regression for the outage: the real MRNA data shape, in a real
    DuckDB file, read back READ-ONLY exactly as the nightly check reads prod."""
    db = tmp_path / "prod-shape.duckdb"
    w = Store(db).connect()
    rid = w.allocate_run_id()
    for day, alpaca_px, yf_px in _MRNA_PROD_ROWS:
        _price(w, "MRNA", day, yf_px, "yfinance", rid)
        _price(w, "MRNA", day, alpaca_px, "alpaca", rid, adj=False)
    w.close()

    r = Store(db).connect(read_only=True)
    try:
        check = _check_no_unjustified_extreme_moves(r, date(2026, 8, 24))
    finally:
        r.close()
    assert check.passed, f"the real MRNA 8/19 move must not block ingest: {check.detail}"


def test_mrna_shape_with_only_yfinance_would_still_have_failed(tmp_path):
    """Control for the test above: strip alpaca's confirming rows out of the SAME
    production shape and the gate fires again. Proves the new pass comes from
    cross-source agreement, not from a loosened threshold."""
    db = tmp_path / "single-source.duckdb"
    w = Store(db).connect()
    rid = w.allocate_run_id()
    for day, _alpaca_px, yf_px in _MRNA_PROD_ROWS:
        _price(w, "MRNA", day, yf_px, "yfinance", rid)
    w.close()

    r = Store(db).connect(read_only=True)
    try:
        check = _check_no_unjustified_extreme_moves(r, date(2026, 8, 24))
    finally:
        r.close()
    assert not check.passed
    assert "MRNA" in check.detail


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


def test_earnings_coverage_exempts_sector_etfs(store):
    """The 11 SPDR sector ETFs report no per-fund earnings, so they must be
    exempt — an ETF can never satisfy this check.

    Regression (found 2026-07-29): the sector ETFs were added to the universe on
    2026-05-26 for the rel_strength_sector_etf_30d feature but never added to
    _NO_EARNINGS_EXEMPT, so `earnings_coverage` FAILED every single day for two
    months on funds that structurally cannot pass. Verified against the live DB:
    0 earnings rows for XL* all time. A permanently-red check trains you to
    ignore the report, which is how the real failures get missed.
    """
    asof = date(2026, 4, 26)
    rid = store.allocate_run_id()
    store.conn.execute(
        "INSERT INTO earnings VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ["AAPL", date(2026, 4, 1), 1.0, 1.1, 1_000_000.0, 1_100_000.0,
         "finnhub", rid],
    )
    etfs = ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU",
            "XLV", "XLY"]

    check = _check_earnings_coverage(store, asof, ["AAPL", *etfs])
    assert check.passed, check.detail


def test_news_per_ticker_minimum_exempts_sector_etfs(store):
    """Same for news: exchanges report on a fund's holdings, not the fund."""
    asof = date(2026, 4, 26)
    rid = store.allocate_run_id()
    store.conn.execute(
        "INSERT INTO news VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ["AAPL", datetime(2026, 4, 20, 12, 0), "alpaca", "h",
         "https://example.com/a", None, "hash-a", rid],
    )
    etfs = ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU",
            "XLV", "XLY"]

    check = _check_news_per_ticker_minimum(store, asof, ["AAPL", *etfs])
    assert check.passed, check.detail


def test_earnings_coverage_still_flags_a_real_single_name_gap(store):
    """The ETF exemption must NOT become a blanket pass: a genuine single-name
    gap still fails. Guards against over-broad exemption (e.g. matching on a
    prefix, which would also swallow XLNX-style real tickers)."""
    asof = date(2026, 4, 26)
    rid = store.allocate_run_id()
    store.conn.execute(
        "INSERT INTO earnings VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ["AAPL", date(2026, 4, 1), 1.0, 1.1, 1_000_000.0, 1_100_000.0,
         "finnhub", rid],
    )

    check = _check_earnings_coverage(store, asof, ["AAPL", "XLK", "MSFT"])
    assert not check.passed
    assert "MSFT" in check.detail
    assert "XLK" not in check.detail


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


def _hold(store, ticker, shares=10):
    """Give the paper_fills ledger a net-long position in `ticker`."""
    store.conn.execute(
        "INSERT INTO paper_fills (alpaca_order_id, asof_date, ticker, side, "
        " filled_shares, fill_price, status, submitted_at, filled_at, run_id) "
        "VALUES (?, DATE '2026-04-01', ?, 'BUY', ?, 100.0, 'filled', "
        " TIMESTAMP '2026-04-01 15:00:00', TIMESTAMP '2026-04-01 15:00:00', 1)",
        [f"o-{ticker}", ticker, shares],
    )


# ---------------------------------------------------------------------------
# theses_freshness — SEMANTICS CHANGED 2026-07-29.
#
# It used to demand a <=7d thesis for EVERY non-SPY ticker in the universe. That
# is unsatisfiable by construction: the agents job writes 4-63 theses/night
# (event-triggered + budget-capped), so ~114 of 266 is the steady state and the
# check was red every single night — one of four permanently-red gauges that
# together trained the habit of ignoring the dashboard.
#
# It is also the WRONG question. In xgb_top_k the overlay treats a stale/missing
# thesis as NEUTRAL — it simply cannot fire a rule — so a stale thesis on the
# #200-ranked name is harmless. What actually costs money is a stale thesis on a
# name we HOLD, because the `strong_bearish` EXIT trigger only fires on held
# tickers. Measured on the live book that day: 7 of 11 held names were stale or
# missing (DKNG 42d, MPC 30d, MRNA 27d, HUM never), i.e. the exit rule was dead
# on the majority of the book during the project's worst drawdown.
#
# So the check now fails on HELD names only, and reports universe coverage as
# informational context. This is not loosening it to go green — it is pointing it
# at the money path. It stays red until the book is covered.
# ---------------------------------------------------------------------------


def test_theses_freshness_passes_when_every_held_name_is_fresh(store):
    universe = ["AAPL", "MSFT", "NVDA", "SPY"]
    asof = date(2026, 4, 26)
    _hold(store, "AAPL")
    _hold(store, "MSFT")
    _insert_thesis(store, "AAPL", asof)
    _insert_thesis(store, "MSFT", asof)
    # NVDA is un-held and thesis-less: irrelevant to the overlay, must not fail.
    check = _check_theses_freshness(store, asof, universe)
    assert check.passed is True, check.detail


def test_theses_freshness_fails_on_a_stale_held_name(store):
    universe = ["AAPL", "MSFT", "SPY"]
    asof = date(2026, 4, 26)
    _hold(store, "AAPL")
    _hold(store, "MSFT")
    _insert_thesis(store, "AAPL", asof)                 # fresh
    _insert_thesis(store, "MSFT", date(2026, 4, 15))    # stale (>7d)
    check = _check_theses_freshness(store, asof, universe)
    assert check.passed is False
    assert "MSFT" in check.detail


def test_theses_freshness_fails_on_a_held_name_with_no_thesis_at_all(store):
    """The HUM case: a position that has never had a thesis."""
    universe = ["AAPL", "HUM"]
    asof = date(2026, 4, 26)
    _hold(store, "AAPL")
    _hold(store, "HUM")
    _insert_thesis(store, "AAPL", asof)
    check = _check_theses_freshness(store, asof, universe)
    assert check.passed is False
    assert "HUM" in check.detail


def test_theses_freshness_ignores_stale_unheld_names(store):
    """The whole point: 152 stale non-held names must not mask a real problem."""
    universe = ["AAPL"] + [f"X{i}" for i in range(40)]
    asof = date(2026, 4, 26)
    _hold(store, "AAPL")
    _insert_thesis(store, "AAPL", asof)
    check = _check_theses_freshness(store, asof, universe)
    assert check.passed is True, check.detail


def test_theses_freshness_passes_on_an_empty_book(store):
    """Flat book -> the exit trigger has nothing to act on -> nothing to fail."""
    check = _check_theses_freshness(store, date(2026, 4, 26), ["AAPL", "MSFT"])
    assert check.passed is True


def test_theses_freshness_reports_universe_coverage_as_context(store):
    """Coverage is still surfaced, just not used as a pass/fail gate, so a slow
    slide in overall thesis coverage stays visible."""
    universe = ["AAPL", "MSFT", "NVDA", "SPY"]
    asof = date(2026, 4, 26)
    _hold(store, "AAPL")
    _insert_thesis(store, "AAPL", asof)
    _insert_thesis(store, "MSFT", asof)
    check = _check_theses_freshness(store, asof, universe)
    assert check.passed is True
    assert "coverage" in check.detail.lower()


def test_theses_freshness_excludes_spy_even_when_held(store):
    """SPY is an ETF benchmark; the agents pipeline never writes it a thesis."""
    universe = ["AAPL", "SPY"]
    asof = date(2026, 4, 26)
    _hold(store, "AAPL")
    _hold(store, "SPY")
    _insert_thesis(store, "AAPL", asof)
    check = _check_theses_freshness(store, asof, universe)
    assert check.passed is True, check.detail


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
    check = _check_all_tickers_have_price(s, asof, ["AAPL", "SPY"], rid)
    assert not check.passed
    assert not check.degraded  # a failing check is never "degraded" -- it's just failed


# ---- 2026-09-09 incident: single-day yfinance flake dropped SPY's row -------
# (partial "possibly delisted" / sqlite-cache OperationalError failures from
# yfinance; alpaca and every overlay source succeeded that run). Coverage was
# 100% -- every OTHER ticker still had a price from some source -- but SPY's
# only current-run row was alpaca's null-adj_close placeholder, so the old
# all-or-nothing SPY floor blocked the whole book's rebalance over one name.


def test_all_tickers_have_price_spy_falls_back_to_prior_session_adj_close():
    """When today's run has no usable SPY row, a real adj_close from within
    the last _SPY_FALLBACK_MAX_AGE_DAYS days is an acceptable substitute --
    the 2026-09-09 shape: 263/264 tickers fine, SPY's yfinance fetch alone
    came back empty."""
    from sma.ingest.quality import _check_all_tickers_have_price

    s = Store(":memory:").connect()
    yesterday = date(2026, 9, 8)
    asof = date(2026, 9, 9)
    rid = s.allocate_run_id()
    # A real SPY price landed YESTERDAY under an earlier run_id.
    s.conn.execute(
        "INSERT INTO prices VALUES ('SPY', ?, 1,1,1,1, 500.0, 100, 'yfinance', 1)",
        [yesterday],
    )
    # Today: AAPL is fine; SPY only has alpaca's null-adj_close placeholder.
    s.conn.execute(
        "INSERT INTO prices VALUES ('AAPL', ?, 1,1,1,1, 1.0, 100, 'yfinance', ?)", [asof, rid]
    )
    s.conn.execute(
        "INSERT INTO prices VALUES ('SPY', ?, 1,1,1,1, NULL, 100, 'alpaca', ?)", [asof, rid]
    )
    check = _check_all_tickers_have_price(s, asof, ["AAPL", "SPY"], rid)
    assert check.passed
    assert check.degraded
    assert "fallback" in check.detail


def test_all_tickers_have_price_spy_fallback_bounded_dead_network_still_blocks():
    """The fallback must NOT reach back into a real multi-day outage (the
    6/1-6/3 freeze this check exists for) -- only the last
    _SPY_FALLBACK_MAX_AGE_DAYS days count, and it must not silently skip PAST
    a gap where nothing landed for anyone to find an older healthy day."""
    from sma.ingest.quality import _SPY_FALLBACK_MAX_AGE_DAYS, _check_all_tickers_have_price

    s = Store(":memory:").connect()
    asof = date(2026, 6, 4)
    rid = s.allocate_run_id()
    # SPY's last REAL price is well outside the fallback window (a stand-in
    # for the 6/1-6/3 dead-network gap this check was built to catch) -- and
    # NOTHING landed for anyone in between, so a "nearest distinct date
    # present" style lookback would wrongly reach back to this day too.
    stale_day = asof - timedelta(days=_SPY_FALLBACK_MAX_AGE_DAYS + 5)
    s.conn.execute(
        "INSERT INTO prices VALUES ('SPY', ?, 1,1,1,1, 500.0, 100, 'yfinance', 1)",
        [stale_day],
    )
    s.conn.execute(
        "INSERT INTO prices VALUES ('AAPL', ?, 1,1,1,1, 1.0, 100, 'yfinance', ?)", [asof, rid]
    )
    s.conn.execute(
        "INSERT INTO prices VALUES ('SPY', ?, 1,1,1,1, NULL, 100, 'alpaca', ?)", [asof, rid]
    )
    check = _check_all_tickers_have_price(s, asof, ["AAPL", "SPY"], rid)
    assert not check.passed
    assert not check.degraded


def test_all_tickers_have_price_tolerates_one_missing_name_and_flags_degraded():
    """A coverage-tolerated gap (>=97%, below the 6/4 audit's floor) must still
    PASS (unchanged since 2026-06-04) but is now flagged degraded so it stops
    being completely silent (2026-09-09 finding: previously nothing surfaced
    this outside logs/quality/*.txt)."""
    from sma.ingest.quality import _check_all_tickers_have_price

    s = Store(":memory:").connect()
    asof = date(2026, 5, 1)
    rid = s.allocate_run_id()
    universe = [f"T{i}" for i in range(40)] + ["SPY"]
    # Every ticker gets a price except T0 -- 40/41 = 97.6% coverage.
    for t in universe:
        if t == "T0":
            continue
        s.conn.execute(
            "INSERT INTO prices VALUES (?, ?, 1,1,1,1, 1.0, 100, 'yfinance', ?)", [t, asof, rid]
        )
    check = _check_all_tickers_have_price(s, asof, universe, rid)
    assert check.passed
    assert check.degraded
    assert "T0" in check.detail


def test_all_tickers_have_price_fully_healthy_run_is_not_degraded():
    """The common case: no fallback, nothing missing -- must not be flagged
    degraded (that would page every single healthy night)."""
    from sma.ingest.quality import _check_all_tickers_have_price

    s = Store(":memory:").connect()
    asof = date(2026, 5, 1)
    rid = s.allocate_run_id()
    for t in ["AAPL", "SPY"]:
        s.conn.execute(
            "INSERT INTO prices VALUES (?, ?, 1,1,1,1, 1.0, 100, 'yfinance', ?)", [t, asof, rid]
        )
    check = _check_all_tickers_have_price(s, asof, ["AAPL", "SPY"], rid)
    assert check.passed
    assert not check.degraded


# ---- notify_degraded_quality_checks -----------------------------------------


def test_notify_degraded_quality_checks_fires_for_degraded_passing_checks():
    from sma.ingest.quality import QualityCheck, QualityReport, notify_degraded_quality_checks

    calls = []
    report = QualityReport(
        asof_date=date(2026, 9, 9),
        run_id=1,
        checks=[
            QualityCheck(
                "all_tickers_have_price", passed=True,
                detail="coverage=100.0% spy_ok=True (spy fallback: prior-session adj_close) "
                       "missing=[]",
                degraded=True,
            ),
            QualityCheck("news_table_grew", passed=True, detail="rows_in_run=100"),
        ],
    )
    stub_notify = lambda title, message: calls.append((title, message))  # noqa: E731
    notified = notify_degraded_quality_checks(report, asof=date(2026, 9, 9), notify_fn=stub_notify)
    assert notified == ["all_tickers_have_price"]
    assert len(calls) == 1
    title, message = calls[0]
    assert "all_tickers_have_price" in title
    assert "fallback" in message


def test_notify_degraded_quality_checks_skips_when_nothing_degraded():
    from sma.ingest.quality import QualityCheck, QualityReport, notify_degraded_quality_checks

    report = QualityReport(
        asof_date=date(2026, 9, 9),
        run_id=1,
        checks=[QualityCheck("all_tickers_have_price", passed=True, detail="all present")],
    )
    calls = []
    stub_notify = lambda title, message: calls.append((title, message))  # noqa: E731
    notified = notify_degraded_quality_checks(report, asof=date(2026, 9, 9), notify_fn=stub_notify)
    assert notified == []
    assert calls == []


def test_notify_degraded_quality_checks_ignores_a_failing_blocking_check():
    """A BLOCKING failure is handled by the separate notify_failure call in the
    CLI (SMA Ingest FAILED) -- this function only ever pages for checks that
    PASSED via a tolerance/fallback path."""
    from sma.ingest.quality import QualityCheck, QualityReport, notify_degraded_quality_checks

    report = QualityReport(
        asof_date=date(2026, 9, 9),
        run_id=1,
        checks=[QualityCheck("all_tickers_have_price", passed=False, detail="coverage=10.0%")],
    )
    calls = []
    stub_notify = lambda title, message: calls.append((title, message))  # noqa: E731
    notified = notify_degraded_quality_checks(report, asof=date(2026, 9, 9), notify_fn=stub_notify)
    assert notified == []
    assert calls == []


def test_notify_degraded_quality_checks_never_raises():
    from sma.ingest.quality import QualityCheck, QualityReport, notify_degraded_quality_checks

    def boom(title, message):
        raise RuntimeError("ntfy is down")

    report = QualityReport(
        asof_date=date(2026, 9, 9),
        run_id=1,
        checks=[
            QualityCheck("all_tickers_have_price", passed=True, detail="degraded", degraded=True)
        ],
    )
    notified = notify_degraded_quality_checks(report, asof=date(2026, 9, 9), notify_fn=boom)
    assert notified == ["all_tickers_have_price"]  # still reports the attempt


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


# ---------------------------------------------------------------------------
# 2026-09-18 incident: Yahoo returned NaN closes for the whole universe.
# yfinance_prices.py now refuses to INSERT a NaN-close row at all (fixed at
# the source), but these checks must independently ignore any NaN/NULL move
# rather than flag it -- defense in depth for legacy/pre-repair rows, and
# because DuckDB compares/orders NaN as the LARGEST value: `NaN > 0.5` is
# TRUE, so an un-guarded NaN adj_close silently satisfied both
# no_unjustified_extreme_moves' >50% test and no_cross_source_price_
# divergence's >2% test, freezing the whole night's rebalance on garbage
# data instead of surfacing it as the coverage/availability problem it is.
# ---------------------------------------------------------------------------


def test_extreme_moves_ignores_nan_adj_close_row(store):
    """A NaN adj_close row (the exact 2026-09-18 shape) must not be flagged as
    an unjustified extreme move against the prior day's real close."""
    from sma.ingest.quality import _check_no_unjustified_extreme_moves

    asof = date(2026, 9, 18)
    rid = 1
    store.conn.execute(
        "INSERT INTO prices VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ["AAPL", asof - timedelta(days=1), 200.0, 205.0, 199.0, 200.0, 200.0,
         1_000_000, "yfinance", rid],
    )
    store.conn.execute(
        "INSERT INTO prices VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ["AAPL", asof, float("nan"), float("nan"), float("nan"), float("nan"),
         float("nan"), 1_000_000, "yfinance", rid],
    )
    check = _check_no_unjustified_extreme_moves(store, asof)
    assert check.passed, f"a NaN row must be ignored, not flagged: {check.detail}"


def test_extreme_moves_ignores_nan_as_the_prior_days_value(store):
    """The NaN row must also be excluded as the LAG "previous" value feeding
    the NEXT real day's comparison -- a NaN yesterday must not make today's
    real close look like a >50% move relative to it."""
    from sma.ingest.quality import _check_no_unjustified_extreme_moves

    asof = date(2026, 9, 19)
    rid = 1
    # 9/18: NaN (the incident night)
    store.conn.execute(
        "INSERT INTO prices VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ["AAPL", asof - timedelta(days=1), float("nan"), float("nan"), float("nan"),
         float("nan"), float("nan"), 1_000_000, "yfinance", rid],
    )
    # 9/17: last real price
    store.conn.execute(
        "INSERT INTO prices VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ["AAPL", asof - timedelta(days=2), 200.0, 205.0, 199.0, 200.0, 200.0,
         1_000_000, "yfinance", rid],
    )
    # 9/19: real price again, close to 9/17's -- must NOT look like an extreme
    # move relative to the (excluded) NaN row.
    store.conn.execute(
        "INSERT INTO prices VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ["AAPL", asof, 201.0, 206.0, 200.0, 201.0, 201.0, 1_000_000, "yfinance", rid],
    )
    check = _check_no_unjustified_extreme_moves(store, asof)
    assert check.passed, (
        f"a NaN prior-day row must not corrupt the next real comparison: {check.detail}"
    )


def test_price_divergence_ignores_nan_price(store):
    """A NaN close/adj_close row must not be treated as a cross-source
    divergence outlier (NaN compares as 'larger than everything' in DuckDB,
    so an un-guarded MAX/MIN would report a bogus 'divergent' spread)."""
    from sma.ingest.quality import _check_price_divergence

    asof = date(2026, 9, 18)
    rid = 1
    store.conn.execute(
        "INSERT INTO prices VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ["AAPL", asof, float("nan"), float("nan"), float("nan"), float("nan"),
         float("nan"), 1_000_000, "yfinance", rid],
    )
    store.conn.execute(
        "INSERT INTO prices VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ["AAPL", asof, 150.0, 151.0, 149.0, 150.62, None, 1_000_000, "alpaca", rid],
    )
    check = _check_price_divergence(store, asof)
    assert check.passed, f"a NaN row must be ignored, not treated as divergent: {check.detail}"


# ---------------------------------------------------------------------------
# no_nan_prices (2026-09-18 fix, layer 3) -- dedicated, non-blocking-but-
# degraded check: reports any NaN close/adj_close rows in the last 5
# sessions. Deliberately never blocks (passed is always True) -- a NaN price
# is a data-availability problem the OTHER checks above now correctly ignore
# rather than misread as an extreme move, and this check exists purely so the
# condition PAGES (via notify_degraded_quality_checks) instead of vanishing
# silently, the way it did before this incident.
# ---------------------------------------------------------------------------


def test_no_nan_prices_clean_run_is_not_degraded(store):
    from sma.ingest.quality import _check_no_nan_prices

    asof = date(2026, 9, 18)
    rid = 1
    store.conn.execute(
        "INSERT INTO prices VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ["AAPL", asof, 150.0, 151.0, 149.0, 150.62, 150.62, 1_000_000, "yfinance", rid],
    )
    check = _check_no_nan_prices(store, asof)
    assert check.passed
    assert not check.degraded
    assert not check.blocking


def test_no_nan_prices_flags_nan_close_as_degraded(store):
    from sma.ingest.quality import _check_no_nan_prices

    asof = date(2026, 9, 18)
    rid = 1
    store.conn.execute(
        "INSERT INTO prices VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ["AAPL", asof, float("nan"), float("nan"), float("nan"), float("nan"),
         float("nan"), 1_000_000, "yfinance", rid],
    )
    check = _check_no_nan_prices(store, asof)
    assert check.passed, "must never block -- a data-availability problem, not a trading halt"
    assert check.degraded, "must be visible via the degraded-notify path"
    assert not check.blocking
    assert "AAPL" in check.detail


def test_no_nan_prices_flags_nan_adj_close_alone(store):
    """close is real but adj_close alone is NaN -- still a NaN price row."""
    from sma.ingest.quality import _check_no_nan_prices

    asof = date(2026, 9, 18)
    rid = 1
    store.conn.execute(
        "INSERT INTO prices VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ["AAPL", asof, 150.0, 151.0, 149.0, 150.62, float("nan"), 1_000_000, "yfinance", rid],
    )
    check = _check_no_nan_prices(store, asof)
    assert check.degraded


def test_no_nan_prices_ignores_null_adj_close(store):
    """A NULL adj_close (alpaca's ordinary placeholder) is NOT a NaN -- must
    not be flagged; that would page every single night on normal data."""
    from sma.ingest.quality import _check_no_nan_prices

    asof = date(2026, 9, 18)
    rid = 1
    store.conn.execute(
        "INSERT INTO prices VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ["AAPL", asof, 150.0, 151.0, 149.0, 150.62, None, 1_000_000, "alpaca", rid],
    )
    check = _check_no_nan_prices(store, asof)
    assert check.passed
    assert not check.degraded


def test_no_nan_prices_only_scans_last_5_sessions(store):
    """A NaN row well outside the lookback window must not still be paging
    weeks later."""
    from sma.ingest.quality import _check_no_nan_prices

    asof = date(2026, 9, 18)
    rid = 1
    old_day = asof - timedelta(days=30)
    store.conn.execute(
        "INSERT INTO prices VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ["AAPL", old_day, float("nan"), float("nan"), float("nan"), float("nan"),
         float("nan"), 1_000_000, "yfinance", rid],
    )
    # Recent sessions are all clean.
    for i in range(5):
        store.conn.execute(
            "INSERT INTO prices VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ["AAPL", asof - timedelta(days=i), 100.0 + i, 101.0, 99.0, 100.0 + i,
             100.0 + i, 1_000_000, "yfinance", rid],
        )
    check = _check_no_nan_prices(store, asof)
    assert check.passed
    assert not check.degraded


def test_run_quality_checks_includes_no_nan_prices(store):
    asof = date(2026, 4, 26)
    universe = ["AAPL"]
    report = run_quality_checks(store, asof_date=asof, universe=universe, run_id=1)
    names = [c.name for c in report.checks]
    assert "no_nan_prices" in names


# ---------------------------------------------------------------------------
# summary() OVERALL line must be waiver-aware (2026-07-29)
# ---------------------------------------------------------------------------


def test_summary_overall_is_pass_when_only_waived_checks_failed():
    """The nightly report printed `OVERALL: FAIL` on runs the pipeline considered
    GOOD, because `summary()` used `self.passed` (all checks) while the sentinel
    used `blocking_failures(waivers)`. So the human-facing artifact said FAIL
    while the machine gate said pass — the fourth permanently-red gauge, and the
    same pattern that trained everyone to ignore the dashboard.
    """
    from sma.ingest.quality import QualityCheck, QualityReport

    report = QualityReport(
        asof_date=date(2026, 7, 29),
        run_id=1,
        checks=[
            QualityCheck("all_tickers_have_price", passed=True, detail="ok"),
            QualityCheck("theses_freshness", passed=False, detail="stale", blocking=False),
        ],
    )
    out = report.summary(waivers=frozenset({"theses_freshness"}))
    assert "OVERALL: PASS" in out, out
    assert "theses_freshness" in out          # still visible, not hidden
    assert "[FAIL] theses_freshness" in out   # per-check truth preserved


def test_summary_overall_is_fail_on_a_real_blocking_failure():
    from sma.ingest.quality import QualityCheck, QualityReport

    report = QualityReport(
        asof_date=date(2026, 7, 29),
        run_id=1,
        checks=[QualityCheck("all_tickers_have_price", passed=False, detail="coverage=10%")],
    )
    out = report.summary()
    assert "OVERALL: FAIL" in out
    assert "all_tickers_have_price" in out


def test_summary_defaults_to_no_waivers():
    """Called with no args (e.g. the backtest CLI) a non-blocking failure is still
    surfaced rather than silently swallowed."""
    from sma.ingest.quality import QualityCheck, QualityReport

    report = QualityReport(
        asof_date=date(2026, 7, 29),
        run_id=1,
        checks=[QualityCheck("theses_freshness", passed=False, detail="stale", blocking=False)],
    )
    out = report.summary()
    assert "theses_freshness" in out


# ---------------------------------------------------------------------------
# no_dead_or_frozen_tickers (2026-08-31) -- corporate-action detector.
#
# EA was delisted 2026-08-04 (LBO) and AVB merged away 2026-08-17, yet nothing
# noticed for weeks: Yahoo kept serving EA a FROZEN last-known quote
# ($209.6999969482422, bit-identical) for several sessions, so
# all_tickers_have_price / enough_sources_succeeded both stayed green — a
# price row DID land every day, it just never changed. See
# scripts/verify_avb_ea_delisting.py and commit 92474dc for the full
# diagnosis, and scripts/verify_no_dead_or_frozen_tickers.py for the
# read-only prod-DB confirmation this check would have caught it.
# ---------------------------------------------------------------------------


def test_no_dead_or_frozen_tickers_silent_on_healthy_universe(store):
    """Normal, varying, fresh-every-session prices -> silent."""
    asof = date(2026, 8, 24)
    rid = store.allocate_run_id()
    for i in range(10):
        _price(store, "AAPL", asof - timedelta(days=9 - i), 100.0 + i, "yfinance", rid)

    check = _check_no_dead_or_frozen_tickers(store, asof, ["AAPL"])
    assert check.passed, check.detail
    assert check.name == "no_dead_or_frozen_tickers"
    assert check.blocking is False


def test_no_dead_or_frozen_tickers_frozen_fixture(store):
    """The core FROZEN fixture: F identical closes in a row -> flagged."""
    asof = date(2026, 8, 24)
    rid = store.allocate_run_id()
    for i in range(5):
        _price(store, "ZOMBIE", asof - timedelta(days=4 - i), 42.0, "yfinance", rid)

    check = _check_no_dead_or_frozen_tickers(store, asof, ["ZOMBIE"])
    assert not check.passed
    assert "ZOMBIE" in check.detail
    assert check.blocking is False

    flags = find_dead_or_frozen_tickers(store, asof, ["ZOMBIE"])
    assert len(flags) == 1
    flag = flags[0]
    assert flag.ticker == "ZOMBIE"
    assert flag.kind == "frozen"
    assert flag.run_length == 5
    assert flag.last_price_date == asof


def test_frozen_requires_full_trailing_run_not_just_some_repeats(store):
    """4 identical closes then a real move is NOT a 5-run -- the run must be
    the TRAILING (most recent) one, not any repeat anywhere in the window."""
    asof = date(2026, 8, 24)
    rid = store.allocate_run_id()
    values = [42.0, 42.0, 42.0, 42.0, 43.0]  # oldest -> newest; last day moves
    for i, v in enumerate(values):
        _price(store, "AAPL", asof - timedelta(days=4 - i), v, "yfinance", rid)

    assert find_dead_or_frozen_tickers(store, asof, ["AAPL"]) == []


def test_frozen_run_length_is_configurable(store):
    """F defaults to 5 but must be overridable (per task: 'make F config-able')."""
    asof = date(2026, 8, 24)
    rid = store.allocate_run_id()
    for i in range(3):
        _price(store, "ZOMBIE", asof - timedelta(days=2 - i), 42.0, "yfinance", rid)

    # Default F=5 needs 5 rows of history; only 3 exist -> not enough to judge.
    assert find_dead_or_frozen_tickers(store, asof, ["ZOMBIE"]) == []

    # F=3 explicitly configured -> flags off the same 3 rows.
    flags = find_dead_or_frozen_tickers(store, asof, ["ZOMBIE"], frozen_run=3)
    assert len(flags) == 1
    assert flags[0].run_length == 3


def test_no_dead_or_frozen_tickers_stale_fixture(store):
    """The core STALE fixture: no price row (any source) for the last S
    sessions, while the rest of the universe has them."""
    asof = date(2026, 8, 24)
    rid = store.allocate_run_id()
    for offset, day in enumerate((asof, asof - timedelta(days=1), asof - timedelta(days=2))):
        for j, t in enumerate(["AAPL", "MSFT", "NVDA", "GOOGL"]):
            _price(store, t, day, 100.0 + j + offset, "yfinance", rid)
    # GHOST has some OLD history but nothing in the last 3 sessions.
    _price(store, "GHOST", asof - timedelta(days=10), 50.0, "yfinance", rid)

    universe = ["AAPL", "MSFT", "NVDA", "GOOGL", "GHOST"]
    flags = find_dead_or_frozen_tickers(store, asof, universe)
    stale = [f for f in flags if f.kind == "stale"]
    assert len(stale) == 1
    assert stale[0].ticker == "GHOST"
    assert stale[0].last_price_date == asof - timedelta(days=10)
    assert stale[0].run_length == 3  # default stale_sessions

    check = _check_no_dead_or_frozen_tickers(store, asof, universe)
    assert not check.passed
    assert "GHOST" in check.detail


def test_stale_sessions_is_configurable(store):
    """S defaults to 3 but must be overridable (per task: 'make S config-able').
    4 healthy tickers + GHOST (never has ANY price row) keeps universe
    coverage at exactly 80%, clearing the systemic-outage guard."""
    asof = date(2026, 8, 24)
    rid = store.allocate_run_id()
    healthy = ["AAPL", "MSFT", "NVDA", "GOOGL"]
    for day in (asof, asof - timedelta(days=1)):
        for t in healthy:
            _price(store, t, day, 100.0, "yfinance", rid)
    universe = [*healthy, "GHOST"]

    # S=2: exactly 2 session-dates exist -> enough history; GHOST flags
    # stale with no known last price.
    flags = find_dead_or_frozen_tickers(store, asof, universe, stale_sessions=2)
    stale = [f for f in flags if f.kind == "stale"]
    assert len(stale) == 1
    assert stale[0].ticker == "GHOST"
    assert stale[0].last_price_date is None

    # S=5: only 2 distinct dates exist in the whole DB -- not enough session
    # history to judge "the last 5 sessions" -> skip rather than false-flag.
    flags_s5 = find_dead_or_frozen_tickers(store, asof, universe, stale_sessions=5)
    assert [f for f in flags_s5 if f.kind == "stale"] == []


def test_stale_check_skips_when_not_enough_session_history(store):
    """Fewer than `stale_sessions` distinct dates exist in the DB at all --
    too little history to judge "the last S sessions" (this is also what
    keeps a fresh/small test DB silent)."""
    asof = date(2026, 8, 24)
    rid = store.allocate_run_id()
    _price(store, "AAPL", asof, 100.0, "yfinance", rid)  # only ONE session exists

    assert find_dead_or_frozen_tickers(store, asof, ["AAPL", "MSFT"]) == []


def test_stale_check_skips_on_a_systemic_outage(store):
    """When MOST of the universe is missing recent data too, that's a
    systemic source outage -- all_tickers_have_price / enough_sources_
    succeeded already catch it. Firing STALE here as well would page once per
    ticker for the entire universe instead of surfacing as the one outage it
    is."""
    asof = date(2026, 8, 24)
    rid = store.allocate_run_id()
    for day in (asof, asof - timedelta(days=1), asof - timedelta(days=2)):
        _price(store, "AAPL", day, 100.0, "yfinance", rid)
    # MSFT, NVDA, GOOGL are ALSO missing recent data -- not one dead ticker,
    # an outage.
    universe = ["AAPL", "MSFT", "NVDA", "GOOGL"]

    flags = find_dead_or_frozen_tickers(store, asof, universe)
    assert [f for f in flags if f.kind == "stale"] == []


# Exact production rows for EA, read out of data/sma.duckdb on 2026-08-31.
# yfinance served an IDENTICAL last-known close (209.6999969482422) for five
# straight sessions after EA's real 2026-08-04 delisting (Tue 8/4 through Mon
# 8/10 -- weekend has no trading session so no row), then stopped returning EA
# at all. This is exactly the pattern the check exists to catch: a price row
# DID land every session, so all_tickers_have_price / enough_sources_succeeded
# both stayed green while the underlying security had already stopped
# trading.
_EA_PROD_FROZEN_ROWS = [
    (date(2026, 8, 4), 209.6999969482422),
    (date(2026, 8, 5), 209.6999969482422),
    (date(2026, 8, 6), 209.6999969482422),
    (date(2026, 8, 7), 209.6999969482422),
    (date(2026, 8, 10), 209.6999969482422),
]


def test_ea_real_frozen_pattern_flags_within_5_sessions_of_delisting(store):
    """Regression for the outage this check exists to close: check_asof
    (2026-08-10) is exactly the 5th trading session after EA's real
    2026-08-04 delisting -- it must fire by then, not weeks later when a
    human audit finally noticed."""
    check_asof = date(2026, 8, 10)
    rid = store.allocate_run_id()
    for day, px in _EA_PROD_FROZEN_ROWS:
        _price(store, "EA", day, px, "yfinance", rid)

    check = _check_no_dead_or_frozen_tickers(store, check_asof, ["EA"])
    assert not check.passed, f"EA's real frozen pattern must be flagged: {check.detail}"

    flags = find_dead_or_frozen_tickers(store, check_asof, ["EA"])
    frozen = next(f for f in flags if f.ticker == "EA")
    assert frozen.kind == "frozen"
    assert frozen.run_length == 5
    assert frozen.last_price_date == check_asof


def test_run_quality_checks_includes_no_dead_or_frozen_tickers(store):
    """The wiring: no_dead_or_frozen_tickers shows up in run_quality_checks
    output, and stays non-blocking."""
    universe = ["AAPL", "SPY"]
    asof = date(2026, 4, 26)
    report = run_quality_checks(store, asof_date=asof, universe=universe, run_id=1)
    check = next(c for c in report.checks if c.name == "no_dead_or_frozen_tickers")
    assert check.blocking is False


def test_check_itself_never_calls_notify(store, monkeypatch):
    """_check_no_dead_or_frozen_tickers / run_quality_checks must stay pure --
    notification is wired separately (notify_new_dead_or_frozen_tickers),
    called explicitly by the ingest CLI, never as a side effect of just
    running the checks (tests, backtests, the dashboard all call these)."""

    def boom(*_a, **_k):
        raise AssertionError("the check itself must never call notify_failure")

    monkeypatch.setattr("sma.ingest.quality.notify_failure", boom)

    asof = date(2026, 8, 10)
    rid = store.allocate_run_id()
    for day, px in _EA_PROD_FROZEN_ROWS:
        _price(store, "EA", day, px, "yfinance", rid)

    report = run_quality_checks(store, asof_date=asof, universe=["EA"], run_id=rid)
    check = next(c for c in report.checks if c.name == "no_dead_or_frozen_tickers")
    # Confirms this isn't vacuous: the check DOES see the flag, it just never
    # calls notify_failure itself while doing so.
    assert not check.passed


# ---------------------------------------------------------------------------
# notify_new_dead_or_frozen_tickers -- per-ticker dedup via
# sma.ingest.dead_ticker_state (same one-small-overwritten-file convention as
# sma.monitoring.regime_state).
# ---------------------------------------------------------------------------


def test_notify_new_dead_or_frozen_fires_once_then_dedupes(tmp_path, monkeypatch):
    monkeypatch.setenv("SMA_DEAD_TICKER_STATE_PATH", str(tmp_path / "state.json"))
    calls = []

    def stub_notify(title, message):
        calls.append((title, message))

    flag = DeadOrFrozenFlag(
        ticker="EA", kind="frozen", last_price_date=date(2026, 8, 10), run_length=5
    )

    notified1 = notify_new_dead_or_frozen_tickers(
        [flag], asof=date(2026, 8, 10), notify_fn=stub_notify
    )
    assert notified1 == ["EA"]
    assert len(calls) == 1
    title, message = calls[0]
    assert "EA" in title
    assert "EA" in message
    assert "2026-08-10" in message
    assert "5" in message

    # Same flag again on a LATER run -> deduped, no second page.
    notified2 = notify_new_dead_or_frozen_tickers(
        [flag], asof=date(2026, 8, 11), notify_fn=stub_notify
    )
    assert notified2 == []
    assert len(calls) == 1


def test_notify_clears_state_when_ticker_heals_and_realerts_on_recurrence(tmp_path, monkeypatch):
    monkeypatch.setenv("SMA_DEAD_TICKER_STATE_PATH", str(tmp_path / "state.json"))
    calls = []
    stub_notify = lambda title, message: calls.append((title, message))  # noqa: E731
    flag = DeadOrFrozenFlag(
        ticker="EA", kind="frozen", last_price_date=date(2026, 8, 10), run_length=5
    )

    notify_new_dead_or_frozen_tickers([flag], asof=date(2026, 8, 10), notify_fn=stub_notify)
    assert len(calls) == 1

    # EA heals (no longer in the flag list) -- no notification, and its dedup
    # record must be dropped, not carried forward forever.
    notify_new_dead_or_frozen_tickers([], asof=date(2026, 8, 11), notify_fn=stub_notify)
    assert len(calls) == 1

    # EA goes bad again later -- must page again, not stay silent off the
    # stale record from the first occurrence.
    notify_new_dead_or_frozen_tickers([flag], asof=date(2026, 9, 1), notify_fn=stub_notify)
    assert len(calls) == 2


def test_notify_message_includes_ticker_last_price_date_and_run_length_for_stale(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("SMA_DEAD_TICKER_STATE_PATH", str(tmp_path / "state.json"))
    calls = []
    stub_notify = lambda title, message: calls.append((title, message))  # noqa: E731
    flag = DeadOrFrozenFlag(
        ticker="GHOST", kind="stale", last_price_date=date(2026, 8, 1), run_length=3
    )

    notify_new_dead_or_frozen_tickers([flag], asof=date(2026, 8, 24), notify_fn=stub_notify)

    title, message = calls[0]
    assert "GHOST" in title
    assert "2026-08-01" in message
    assert "3" in message


def test_notify_never_raises_even_if_notify_fn_explodes(tmp_path, monkeypatch):
    """Same never-break-the-caller contract as sma.ingest.notify.notify_failure
    and sma.monitoring.check_regime_turn: a notification bug must not break
    the ingest run this fires from."""
    monkeypatch.setenv("SMA_DEAD_TICKER_STATE_PATH", str(tmp_path / "state.json"))

    def boom(title, message):
        raise RuntimeError("ntfy is down")

    flag = DeadOrFrozenFlag(
        ticker="EA", kind="frozen", last_price_date=date(2026, 8, 10), run_length=5
    )
    notified = notify_new_dead_or_frozen_tickers([flag], asof=date(2026, 8, 10), notify_fn=boom)
    assert notified == ["EA"]  # still reports the attempt even though the send failed


# ---------------------------------------------------------------------------
# agents_last_run_healthy (2026-09-18) -- unmasks a failed-but-hidden agents
# run that theses_freshness's 7-day window would otherwise hide.
# ---------------------------------------------------------------------------


def _write_agents_sentinel(asof, *, run_id=1, processed=0, failed=0, skipped_budget=0,
                            skipped_deadline=0, cached_fallback=0, skipped_existing=0,
                            skipped_no_trigger=0):
    from sma.sentinels import write_sentinel
    write_sentinel(
        label="com.sma.agents.daily",
        asof=asof,
        payload={
            "label": "com.sma.agents.daily",
            "asof": asof.isoformat(),
            "completed_at": f"{asof.isoformat()}T20:15:00Z",
            "run_id": run_id,
            "tickers_processed": processed,
            "tickers_skipped_no_trigger": skipped_no_trigger,
            "tickers_skipped_budget": skipped_budget,
            "tickers_skipped_existing": skipped_existing,
            "tickers_skipped_deadline": skipped_deadline,
            "tickers_cached_fallback": cached_fallback,
            "tickers_failed": failed,
            "budget_spent_usd": 0.01,
            "force_full": False,
            "quality": {"passed": True, "blocking_failures": []},
        },
    )


def test_agents_last_run_healthy_no_sentinel_is_silent(store):
    from sma.ingest.quality import _check_agents_last_run_healthy

    check = _check_agents_last_run_healthy(store, date(2026, 9, 18))
    assert check.passed
    assert not check.degraded


def test_agents_last_run_healthy_healthy_night_is_silent(store):
    """The common case: agents ran, processed everything, nothing failed --
    must not page every single healthy night."""
    from sma.ingest.quality import _check_agents_last_run_healthy

    asof = date(2026, 9, 18)
    _write_agents_sentinel(asof, processed=11, failed=0)
    check = _check_agents_last_run_healthy(store, asof)
    assert check.passed
    assert not check.degraded


def test_agents_last_run_healthy_deadline_skip_is_not_counted_as_failed(store):
    """A 0-attempt deadline-skip run (the 9/14 run2 shape) is a schedule
    miss, already covered by sentinel-missing/late-kick detection -- it must
    NOT be flagged as a failed agents run."""
    from sma.ingest.quality import _check_agents_last_run_healthy

    asof = date(2026, 9, 14)
    _write_agents_sentinel(asof, processed=0, failed=0, skipped_deadline=11)
    check = _check_agents_last_run_healthy(store, asof)
    assert check.passed
    assert not check.degraded


def test_agents_last_run_healthy_flags_degraded_on_full_failure(store):
    """The masked incident this fix unmasks: 8/31, 9/4, 9/14, 9/16 -- agents
    attempted the full held set and failed every single LLM call (host DNS
    outage), while theses_freshness stayed PASS all along. The message names
    the failure signature, per spec."""
    from sma.ingest.quality import _check_agents_last_run_healthy

    asof = date(2026, 9, 16)
    rid = 42
    _write_agents_sentinel(asof, run_id=rid, processed=0, failed=11)
    for _ in range(11):
        store.conn.execute(
            "INSERT INTO agent_calls (run_id, ticker, asof_date, agent_role, model_id, "
            " status, error) VALUES (?, 'AAPL', ?, 'researcher', 'claude-haiku-4-5', "
            " 'error', 'Connection error')",
            [rid, asof],
        )
    check = _check_agents_last_run_healthy(store, asof)
    assert check.passed  # non-blocking -- must never gate decide
    assert check.degraded
    assert "11/11" in check.detail
    assert "Connection error" in check.detail


def test_agents_last_run_healthy_majority_failure_ratio_flags_degraded(store):
    """>=50% failed (not just 100%) is enough to flag -- a majority-degraded
    night is still clearly broken."""
    from sma.ingest.quality import _check_agents_last_run_healthy

    asof = date(2026, 9, 17)
    _write_agents_sentinel(asof, processed=5, failed=5)  # 5/10 = 50%
    check = _check_agents_last_run_healthy(store, asof)
    assert check.passed
    assert check.degraded
    assert "5/10" in check.detail


def test_agents_last_run_healthy_minority_failure_ratio_is_silent(store):
    """Below the 50% ratio and not part of a 2-night zero-success streak: a
    handful of flaky tickers on an otherwise-healthy night must not page."""
    from sma.ingest.quality import _check_agents_last_run_healthy

    asof = date(2026, 9, 17)
    _write_agents_sentinel(asof, processed=8, failed=2)  # 2/10 = 20%
    check = _check_agents_last_run_healthy(store, asof)
    assert check.passed
    assert not check.degraded


def test_agents_last_run_healthy_persistent_zero_success_over_two_runs_flags_degraded(store):
    """A lower-ratio-but-still-broken pattern (mostly budget-skipped/cached-
    fallback, a couple of real errors) that never crosses the single-night
    50% ratio, but produces ZERO fresh theses two runs running despite
    tickers to work on."""
    from sma.ingest.quality import _check_agents_last_run_healthy

    day1 = date(2026, 9, 10)
    day2 = date(2026, 9, 11)
    # Neither night crosses the 50% single-night ratio (1/5 = 20% each)...
    _write_agents_sentinel(day1, processed=0, failed=1, cached_fallback=4)
    _write_agents_sentinel(day2, processed=0, failed=1, cached_fallback=4)
    # ...but BOTH produced zero fresh theses despite 5 tickers attempted.
    check = _check_agents_last_run_healthy(store, day2)
    assert check.passed
    assert check.degraded
    assert "2026-09-10" in check.detail
    assert "2026-09-11" in check.detail


def test_agents_last_run_healthy_skips_deadline_skip_nights_when_looking_back(store):
    """A 0-attempt deadline-skip night sitting between two real zero-success
    runs must not break the "last N runs" lookback -- it isn't a run at all
    for this purpose."""
    from sma.ingest.quality import _check_agents_last_run_healthy

    day1 = date(2026, 9, 10)
    day2 = date(2026, 9, 11)  # deadline-skip, 0 attempted
    day3 = date(2026, 9, 12)
    _write_agents_sentinel(day1, processed=0, failed=1, cached_fallback=4)
    _write_agents_sentinel(day2, processed=0, failed=0, skipped_deadline=5)
    _write_agents_sentinel(day3, processed=0, failed=1, cached_fallback=4)
    check = _check_agents_last_run_healthy(store, day3)
    assert check.passed
    assert check.degraded
    assert "2026-09-10" in check.detail
    assert "2026-09-11" not in check.detail


def test_agents_last_run_healthy_one_good_night_in_the_lookback_is_silent(store):
    """The lookback requires BOTH of the last two real runs to have zero
    fresh theses -- one healthy night in the window must not page."""
    from sma.ingest.quality import _check_agents_last_run_healthy

    day1 = date(2026, 9, 10)
    day2 = date(2026, 9, 11)
    _write_agents_sentinel(day1, processed=0, failed=1, cached_fallback=4)
    _write_agents_sentinel(day2, processed=3, failed=1, cached_fallback=1)
    check = _check_agents_last_run_healthy(store, day2)
    assert check.passed
    assert not check.degraded


def test_run_quality_checks_includes_agents_last_run_healthy(store):
    """The wiring: agents_last_run_healthy shows up in run_quality_checks
    output, and (via the generic degraded-check contract) is automatically
    covered by notify_degraded_quality_checks -- no new notify path needed."""
    asof = date(2026, 9, 16)
    _write_agents_sentinel(asof, processed=0, failed=11)
    report = run_quality_checks(store, asof_date=asof, universe=["AAPL"], run_id=1)
    names = [c.name for c in report.checks]
    assert "agents_last_run_healthy" in names
    check = next(c for c in report.checks if c.name == "agents_last_run_healthy")
    assert check.passed and check.degraded
    assert not check.blocking  # never gates decide


def test_notify_degraded_quality_checks_fires_for_agents_last_run_healthy(store):
    """Confirms the reuse of the c9165d0 degraded-notify path end to end."""
    from sma.ingest.quality import notify_degraded_quality_checks

    asof = date(2026, 9, 16)
    _write_agents_sentinel(asof, processed=0, failed=11)
    report = run_quality_checks(store, asof_date=asof, universe=["AAPL"], run_id=1)
    calls = []
    stub_notify = lambda title, message: calls.append((title, message))  # noqa: E731
    notified = notify_degraded_quality_checks(report, asof=asof, notify_fn=stub_notify)
    assert "agents_last_run_healthy" in notified
    title, message = next(
        (t, m) for t, m in calls if "agents_last_run_healthy" in t
    )
    assert "11/11" in message


def test_agents_last_run_healthy_at_ingest_time_reads_the_previous_run(store):
    """Flaw hunt 2026-10-01 B7. Ingest runs this check at 18:30; agents for
    the same date run at 19:45. Reading only today's sentinel, it said "no
    agents sentinel yet" on every report and never saw a failed night. With
    no sentinel for today it must evaluate the most recent prior run: the
    10/1 night (23/23 failed, credit exhausted) is flagged by 10/2's ingest."""
    from sma.ingest.quality import _check_agents_last_run_healthy

    prior = date(2026, 10, 1)
    _write_agents_sentinel(prior, run_id=7, processed=0, failed=23)
    store.conn.execute(
        "INSERT INTO agent_calls (run_id, ticker, asof_date, agent_role, model_id, "
        " status, error) VALUES (7, 'MU', ?, 'researcher', 'claude-haiku-4-5', "
        " 'error', 'Your credit balance is too low')",
        [prior],
    )
    check = _check_agents_last_run_healthy(store, date(2026, 10, 2))
    assert check.passed  # never gates
    assert check.degraded
    assert "23/23" in check.detail
    assert "credit balance" in check.detail
    assert "2026-10-01" in check.detail


def test_agents_last_run_healthy_prior_healthy_run_is_silent(store):
    from sma.ingest.quality import _check_agents_last_run_healthy

    _write_agents_sentinel(date(2026, 10, 1), processed=20, failed=0)
    check = _check_agents_last_run_healthy(store, date(2026, 10, 2))
    assert check.passed
    assert not check.degraded


def test_adjusted_price_coverage_flags_names_with_only_a_null_adj_row(store):
    """Flaw hunt 2026-10-01 A3. Alpaca writes adj_close NULL on purpose and
    the predictor reads only adj_close IS NOT NULL rows, so a ticker that
    yfinance missed is present for all_tickers_have_price but is scored on
    its previous bar (9/24: 237 failed downloads, quality PASS, 264
    predictions). The new check never blocks; it names those tickers and
    pages via the degraded path."""
    from sma.ingest.quality import _check_adjusted_price_coverage

    asof = date(2026, 9, 24)
    rid = store.allocate_run_id()
    rows = [
        ["AAPL", asof, 1, 1, 1, 1, 1.0, 10, "yfinance", rid],
        ["AAPL", asof, 1, 1, 1, 1, None, 10, "alpaca", rid],
        ["MSFT", asof, 1, 1, 1, 1, None, 10, "alpaca", rid],
        ["NVDA", asof, 1, 1, 1, 1, None, 10, "alpaca", rid],
    ]
    for r in rows:
        store.conn.execute("INSERT INTO prices VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", r)

    check = _check_adjusted_price_coverage(store, asof, ["AAPL", "MSFT", "NVDA", "GOOG"])
    assert check.passed
    assert not check.blocking
    assert check.degraded
    assert "1/4" in check.detail
    assert "MSFT" in check.detail and "NVDA" in check.detail and "GOOG" in check.detail


def test_adjusted_price_coverage_full_is_silent(store):
    from sma.ingest.quality import _check_adjusted_price_coverage

    asof = date(2026, 9, 24)
    rid = store.allocate_run_id()
    store.conn.execute(
        "INSERT INTO prices VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ["AAPL", asof, 1, 1, 1, 1, 1.0, 10, "yfinance", rid],
    )
    check = _check_adjusted_price_coverage(store, asof, ["AAPL"])
    assert check.passed and not check.degraded


def test_run_quality_checks_includes_adjusted_price_coverage(store):
    asof = date(2026, 4, 23)
    rid = store.allocate_run_id()
    report = run_quality_checks(store, asof_date=asof, universe=["AAPL"], run_id=rid)
    assert any(c.name == "adjusted_price_coverage" for c in report.checks)
