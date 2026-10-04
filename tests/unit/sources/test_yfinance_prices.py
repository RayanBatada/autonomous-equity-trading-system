from datetime import date
from unittest.mock import patch

import pandas as pd
import pytest

import sma.ingest.sources.yfinance_prices as yfinance_prices_mod
from sma.ingest.sources.yfinance_prices import YFinancePricesSource
from sma.ingest.store import Store


@pytest.fixture
def store():
    s = Store(":memory:").connect()
    yield s
    s.close()


def _fake_yf_dataframe():
    return pd.DataFrame(
        {
            "Open":      [150.0, 151.0],
            "High":      [152.0, 153.0],
            "Low":       [149.0, 150.0],
            "Close":     [151.5, 152.5],
            "Adj Close": [151.5, 152.5],
            "Volume":    [10_000_000, 11_000_000],
        },
        index=pd.to_datetime(["2026-04-22", "2026-04-23"]),
    )


def test_fetch_inserts_rows(store):
    src = YFinancePricesSource(lookback_days=5)
    run_id = store.allocate_run_id()
    store.log_run_start(run_id, source="yfinance")

    with patch("sma.ingest.sources.yfinance_prices.yf.download",
               return_value=_fake_yf_dataframe()):
        result = src.fetch(["AAPL"], date(2026, 4, 23), store, run_id)

    assert result.status == "ok"
    assert result.rows_inserted == 2
    rows = store.conn.execute(
        "SELECT ticker, date, open, close, source FROM prices ORDER BY date"
    ).fetchall()
    assert len(rows) == 2
    assert rows[0][0] == "AAPL"
    assert rows[0][4] == "yfinance"


def test_fetch_handles_empty_dataframe(store):
    src = YFinancePricesSource(lookback_days=5)
    run_id = store.allocate_run_id()
    store.log_run_start(run_id, source="yfinance")

    with patch("sma.ingest.sources.yfinance_prices.yf.download",
               return_value=pd.DataFrame()):
        result = src.fetch(["FAKE"], date(2026, 4, 23), store, run_id)

    assert result.rows_inserted == 0
    assert result.status == "ok"


def _fake_yf_dataframe_multiindex(ticker: str = "AAPL"):
    """Mimics newer yfinance behavior where columns are MultiIndex even for one ticker."""
    df = _fake_yf_dataframe()
    df.columns = pd.MultiIndex.from_tuples([(c, ticker) for c in df.columns])
    return df


def test_fetch_handles_multiindex_columns(store):
    """Newer yfinance versions return MultiIndex columns by default. We must flatten."""
    src = YFinancePricesSource(lookback_days=5)
    run_id = store.allocate_run_id()
    store.log_run_start(run_id, source="yfinance")

    with patch("sma.ingest.sources.yfinance_prices.yf.download",
               return_value=_fake_yf_dataframe_multiindex("AAPL")):
        result = src.fetch(["AAPL"], date(2026, 4, 23), store, run_id)

    assert result.status == "ok"
    assert result.rows_inserted == 2


def test_fetch_falls_back_to_per_ticker_on_bulk_failure(store):
    src = YFinancePricesSource(lookback_days=5)
    run_id = store.allocate_run_id()
    store.log_run_start(run_id, source="yfinance")

    call_count = {"n": 0}
    def fake_download(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("bulk download flaked")
        return _fake_yf_dataframe()

    with patch("sma.ingest.sources.yfinance_prices.yf.download",
               side_effect=fake_download):
        result = src.fetch(["AAPL", "MSFT"], date(2026, 4, 23), store, run_id)

    assert result.status == "ok"
    assert call_count["n"] >= 3


def test_fetch_returns_error_when_no_rows_inserted(store):
    """All yfinance fetches failing (e.g. DNS down) → rows=0 → status MUST be
    'error', not 'ok'. status='ok' with 0 rows let enough_sources_succeeded pass
    on a dead network, masking a total price-ingest failure — the exact bug that
    hid the 6/1-6/3 bot freeze (decide blocked on stale prices, no alert)."""
    src = YFinancePricesSource(lookback_days=5)
    run_id = store.allocate_run_id()
    store.log_run_start(run_id, source="yfinance")
    with patch("sma.ingest.sources.yfinance_prices.yf.download",
               side_effect=Exception("nodename nor servname provided, or not known")):
        result = src.fetch(["AAPL"], date(2026, 4, 23), store, run_id)
    assert result.rows_inserted == 0
    assert result.status == "error"


def test_split_back_adjustment_triggers_full_history_resync(store, monkeypatch):
    """2026-07-01 review: KLAC's 10:1 split left stored history 10x desynced for
    three weeks because the nightly window never re-fetches old rows. When the
    oldest overlap row's adj_close disagrees >1% with what's stored, the source
    must refetch the ticker's FULL history in the same run."""
    import pandas as pd

    import sma.ingest.sources.yfinance_prices as mod

    src = mod.YFinancePricesSource(lookback_days=5)
    run_id = store.allocate_run_id()
    store.log_run_start(run_id, source="yfinance")

    # stored row: pre-split scale (adj_close 1000.0 on 4/20)
    store.conn.execute(
        "INSERT INTO prices (ticker, date, open, high, low, close, adj_close, "
        "volume, source, run_id) "
        "VALUES ('KLAC', DATE '2026-04-20', 1000, 1010, 990, 1000, 1000.0, 100, 'yfinance', ?)",
        [run_id],
    )

    def _df(dates, scale):
        idx = pd.to_datetime(dates)
        return pd.DataFrame(
            {"Open": [100.0 * scale] * len(idx), "High": [101.0 * scale] * len(idx),
             "Low": [99.0 * scale] * len(idx), "Close": [100.0 * scale] * len(idx),
             "Adj Close": [100.0 * scale] * len(idx), "Volume": [1000] * len(idx)},
            index=idx,
        )

    # window fetch: post-split scale — oldest overlap (4/20) now 100.0 vs stored 1000.0
    window = _df(["2026-04-20", "2026-04-21", "2026-04-22", "2026-04-23"], 1.0)
    full = _df(["2026-01-05", "2026-04-20", "2026-04-21", "2026-04-22", "2026-04-23"], 1.0)
    calls = []

    def fake_download(tickers, **kw):
        calls.append((tickers, kw.get("start")))
        return window if len(calls) == 1 else full

    monkeypatch.setattr(mod.yf, "download", fake_download)
    result = src.fetch(["KLAC"], date(2026, 4, 23), store, run_id)

    assert result.status == "ok"
    assert len(calls) == 2, "drift must trigger the full-history refetch"
    assert calls[1][1] == "2016-01-01"
    # stored history is re-synced to the post-split scale
    val = store.conn.execute(
        "SELECT adj_close FROM prices WHERE ticker='KLAC' AND date=DATE '2026-04-20'"
    ).fetchone()[0]
    assert abs(val - 100.0) < 1e-9
    early = store.conn.execute(
        "SELECT COUNT(*) FROM prices WHERE ticker='KLAC' AND date=DATE '2026-01-05'"
    ).fetchone()[0]
    assert early == 1, "full-history rows inserted"


def test_no_resync_when_adjustments_agree(store, monkeypatch):
    """Normal night: overlap row matches stored value -> exactly one download."""
    import pandas as pd

    import sma.ingest.sources.yfinance_prices as mod

    src = mod.YFinancePricesSource(lookback_days=5)
    run_id = store.allocate_run_id()
    store.log_run_start(run_id, source="yfinance")
    store.conn.execute(
        "INSERT INTO prices (ticker, date, open, high, low, close, adj_close, "
        "volume, source, run_id) "
        "VALUES ('AAPL', DATE '2026-04-20', 100, 101, 99, 100, 100.0, 100, 'yfinance', ?)",
        [run_id],
    )
    idx = pd.to_datetime(["2026-04-20", "2026-04-21"])
    window = pd.DataFrame(
        {"Open": [100.0, 100.5], "High": [101.0, 101.5], "Low": [99.0, 99.5],
         "Close": [100.0, 100.5], "Adj Close": [100.0, 100.5], "Volume": [1000, 1000]},
        index=idx,
    )
    calls = []

    def fake_download(tickers, **kw):
        calls.append(tickers)
        return window

    monkeypatch.setattr(mod.yf, "download", fake_download)
    result = src.fetch(["AAPL"], date(2026, 4, 21), store, run_id)
    assert result.status == "ok"
    assert len(calls) == 1


def test_failed_resync_preserves_drift_evidence_for_retry(store, monkeypatch):
    """2026-07-02 adversarial review H1: if the full-history refetch FAILS, the
    drifted ticker's window rows must NOT have been replaced — otherwise the
    comparison row is rewritten, detection can never re-fire, and the desync
    becomes permanent (the original KLAC bug, reintroduced). Deferring the
    window insert makes a failed heal self-retrying tomorrow."""
    import pandas as pd

    import sma.ingest.sources.yfinance_prices as mod

    src = mod.YFinancePricesSource(lookback_days=5)
    run_id = store.allocate_run_id()
    store.log_run_start(run_id, source="yfinance")
    store.conn.execute(
        "INSERT INTO prices (ticker, date, open, high, low, close, adj_close, "
        "volume, source, run_id) "
        "VALUES ('KLAC', DATE '2026-04-20', 1000, 1010, 990, 1000, 1000.0, 100, 'yfinance', ?)",
        [run_id],
    )
    idx = pd.to_datetime(["2026-04-20", "2026-04-21"])
    window = pd.DataFrame(
        {"Open": [100.0, 100.5], "High": [101.0, 101.5], "Low": [99.0, 99.5],
         "Close": [100.0, 100.5], "Adj Close": [100.0, 100.5], "Volume": [1000, 1000]},
        index=idx,
    )
    calls = []

    def fake_download(tickers, **kw):
        calls.append(kw.get("start"))
        if len(calls) == 1:
            return window
        raise RuntimeError("rate limited")  # full-history refetch fails

    monkeypatch.setattr(mod.yf, "download", fake_download)
    pages = []
    monkeypatch.setattr(
        "sma.ingest.notify.notify_failure", lambda **kw: pages.append(kw)
    )
    src.fetch(["KLAC"], date(2026, 4, 21), store, run_id)

    assert len(calls) == 2  # window + attempted full refetch
    # evidence preserved: the stored overlap row still has the OLD scale
    val = store.conn.execute(
        "SELECT adj_close FROM prices WHERE ticker='KLAC' AND date=DATE '2026-04-20'"
    ).fetchone()[0]
    assert abs(val - 1000.0) < 1e-9, "window insert must be deferred on drift"
    assert pages, "a failed resync must page"


# ---------------------------------------------------------------------------
# Cross-source outlier quarantine (live incident, 2026-08-24): Yahoo returned
# a phantom half-price MNST row (close 48.19 vs alpaca's 96.38 and ~97 on
# Yahoo's own neighboring days) that froze trading for a night via
# no_unjustified_extreme_moves. _quarantine_cross_source_outliers drops a row
# whose RAW close diverges >30% from EVERY other source's stored close for
# the same (ticker, date) before it's ever inserted. Splits/dividends move
# adj_close, never the same-day raw close across vendors, so >30% same-day
# raw-close divergence is unambiguous single-vendor corruption, not a
# corporate action.
# ---------------------------------------------------------------------------


def _insert_other_source_close(store, run_id, ticker, d, close, source="alpaca"):
    store.conn.execute(
        "INSERT INTO prices (ticker, date, open, high, low, close, adj_close, "
        "volume, source, run_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [ticker, d, close, close, close, close, close, 1000, source, run_id],
    )


def _row(ticker, d, close, source="yfinance", run_id=1):
    return (ticker, d, close, close, close, close, close, 1000, source, run_id)


def test_quarantine_skips_row_diverging_from_all_other_sources(store):
    """The actual MNST incident: 48.19 vs alpaca's 96.38 is >30% divergent —
    the corrupt row must be dropped."""
    src = YFinancePricesSource()
    run_id = store.allocate_run_id()
    store.log_run_start(run_id, source="yfinance")
    d = date(2026, 7, 31)
    _insert_other_source_close(store, run_id, "MNST", d, 96.38)

    kept = src._quarantine_cross_source_outliers(
        "MNST", [_row("MNST", d, 48.19, run_id=run_id)], store
    )

    assert kept == []


def test_quarantine_keeps_row_agreeing_with_other_source(store):
    """A row close to another source's close (normal vendor noise) is kept."""
    src = YFinancePricesSource()
    run_id = store.allocate_run_id()
    store.log_run_start(run_id, source="yfinance")
    d = date(2026, 7, 31)
    _insert_other_source_close(store, run_id, "AAPL", d, 200.0)

    row = _row("AAPL", d, 201.5, run_id=run_id)
    kept = src._quarantine_cross_source_outliers("AAPL", [row], store)

    assert kept == [row]


def test_quarantine_keeps_row_when_no_other_source_data(store):
    """No evidence to compare against (no other source has this date) means
    keep the row — a thinly-covered ticker/date must not be penalized."""
    src = YFinancePricesSource()
    run_id = store.allocate_run_id()
    store.log_run_start(run_id, source="yfinance")
    d = date(2026, 7, 31)

    row = _row("ZZZZ", d, 12.34, run_id=run_id)
    kept = src._quarantine_cross_source_outliers("ZZZZ", [row], store)

    assert kept == [row]


def test_quarantine_boundary_just_under_30pct_is_kept(store):
    """Just inside the 30% tolerance is kept, not quarantined. (Exact
    floating-point equality at the 0.30 threshold is not asserted here —
    70.0/100.0 - 1.0 lands a hair on either side of 0.30 depending on binary
    float rounding, which is not a meaningful contract to pin down; a value
    clearly on each side of the threshold is what actually matters.)"""
    src = YFinancePricesSource()
    run_id = store.allocate_run_id()
    store.log_run_start(run_id, source="yfinance")
    d = date(2026, 7, 31)
    _insert_other_source_close(store, run_id, "AAPL", d, 100.0)

    row = _row("AAPL", d, 70.1, run_id=run_id)  # |70.1/100 - 1| == 0.299
    kept = src._quarantine_cross_source_outliers("AAPL", [row], store)

    assert kept == [row]


def test_quarantine_skips_row_just_over_30pct_boundary(store):
    """Just outside the 30% tolerance IS quarantined — confirms the kept-case
    above is testing a real edge, not a loosely tolerant comparison."""
    src = YFinancePricesSource()
    run_id = store.allocate_run_id()
    store.log_run_start(run_id, source="yfinance")
    d = date(2026, 7, 31)
    _insert_other_source_close(store, run_id, "AAPL", d, 100.0)

    row = _row("AAPL", d, 69.9, run_id=run_id)  # |69.9/100 - 1| == 0.301
    kept = src._quarantine_cross_source_outliers("AAPL", [row], store)

    assert kept == []


def test_quarantine_keeps_row_when_any_other_source_agrees(store):
    """Multiple other-source rows for the same date: the source code requires
    ALL of them to diverge before quarantining (`all(...)`). If even one
    other source agrees, the row is kept — a single corrupt second source
    must not veto a good row."""
    src = YFinancePricesSource()
    run_id = store.allocate_run_id()
    store.log_run_start(run_id, source="yfinance")
    d = date(2026, 7, 31)
    _insert_other_source_close(store, run_id, "MNST", d, 96.38, source="alpaca")
    _insert_other_source_close(store, run_id, "MNST", d, 95.90, source="finnhub")

    row = _row("MNST", d, 96.0, run_id=run_id)  # agrees with both
    kept = src._quarantine_cross_source_outliers("MNST", [row], store)

    assert kept == [row]


def test_quarantine_logs_warning_for_dropped_row(store, monkeypatch):
    warnings = []
    monkeypatch.setattr(
        yfinance_prices_mod.logger, "warning",
        lambda *a, **kw: warnings.append((a, kw)),
    )
    src = YFinancePricesSource()
    run_id = store.allocate_run_id()
    store.log_run_start(run_id, source="yfinance")
    d = date(2026, 7, 31)
    _insert_other_source_close(store, run_id, "MNST", d, 96.38)

    src._quarantine_cross_source_outliers(
        "MNST", [_row("MNST", d, 48.19, run_id=run_id)], store
    )

    assert warnings, "a quarantined row must log a warning"
    assert "quarantined" in warnings[0][0][0]


def test_fetch_quarantines_corrupt_row_end_to_end(store):
    """Full fetch() path, replicating the live 2026-08-24 incident: a
    pre-existing alpaca close for MNST plus a yfinance response with a
    phantom half-price row for the same date. The corrupt row must never
    reach the prices table, and the fetch must still report status='ok'
    (quarantining a bad row is not a fetch failure)."""
    src = YFinancePricesSource(lookback_days=5)
    run_id = store.allocate_run_id()
    store.log_run_start(run_id, source="yfinance")
    d = date(2026, 7, 31)
    _insert_other_source_close(store, run_id, "MNST", d, 96.38)

    df = pd.DataFrame(
        {
            "Open": [48.0], "High": [48.5], "Low": [47.5], "Close": [48.19],
            "Adj Close": [48.19], "Volume": [5_000_000],
        },
        index=pd.to_datetime(["2026-07-31"]),
    )

    with patch("sma.ingest.sources.yfinance_prices.yf.download", return_value=df):
        result = src.fetch(["MNST"], d, store, run_id)

    assert result.status == "ok"
    assert result.rows_inserted == 0
    yfinance_rows = store.conn.execute(
        "SELECT close FROM prices WHERE ticker='MNST' AND source='yfinance'"
    ).fetchall()
    assert yfinance_rows == [], "the quarantined row must not be inserted"
    # The good alpaca row is untouched.
    alpaca_close = store.conn.execute(
        "SELECT close FROM prices WHERE ticker='MNST' AND source='alpaca'"
    ).fetchone()[0]
    assert alpaca_close == 96.38


# ---------------------------------------------------------------------------
# Isolation refinement (latent bug, found 2026-08-26): the guard above quarantines
# on ANY cross-source divergence, but Yahoo split-adjusts historical Close --
# after a split, yfinance's re-adjusted pre-split rows legitimately diverge from
# alpaca's raw (never-adjusted, adj_close NULL) rows by the split factor (live:
# CRWD 2026-06-02 yfinance close 192.24 vs alpaca 768.84, a 4:1 split later that
# month). _detect_adj_drift then triggers a full-history refetch whose inserts
# pass through this SAME quarantine guard -- every re-adjusted pre-split row was
# being rejected, silently failing the resync and leaving mixed-scale history
# (the KLAC-class bug the drift detector exists to prevent).
#
# Fix: quarantine now additionally requires the row to be an ISOLATED SPIKE
# relative to the SAME vendor's own adjacent rows -- |close/prev-1| > 30% AND
# |close/next-1| > 30% when both neighbors are available. A split shifts every
# fetched row together, so neighbors agree and nothing is isolated; a phantom
# row disagrees with its real neighbors on both sides. Neighbors come from this
# same fetched batch when available; at a window edge (incl. a single-row
# batch, which is an edge on both sides) we fall back to the vendor's own
# already-stored neighbor row in the DB, one side at a time. When only one side
# has any evidence at all (window or DB), that one side alone decides. When
# NEITHER side has evidence, isolation can't be evaluated either way, so (a)
# alone decides -- unchanged from the original guard's no-DB-history behavior
# (this keeps the original MNST test passing as-is).
# ---------------------------------------------------------------------------


def test_quarantine_keeps_uniform_split_shift_across_window(store):
    """CRWD-style: a full-history refetch after a split re-adjusts EVERY
    pre-split row to the same new scale. Each row diverges >30% from alpaca's
    raw close (satisfies the old rule alone), but every row agrees with its
    OWN vendor neighbors (nothing is isolated) -- so all must be KEPT, not
    just some. This is the exact resync the latent bug was silently failing."""
    src = YFinancePricesSource()
    run_id = store.allocate_run_id()
    store.log_run_start(run_id, source="yfinance")
    dates = [date(2026, 6, 1), date(2026, 6, 2), date(2026, 6, 3), date(2026, 6, 4)]
    for d in dates:
        _insert_other_source_close(store, run_id, "CRWD", d, 768.84, source="alpaca")

    rows = [_row("CRWD", d, 192.24, run_id=run_id) for d in dates]
    kept = src._quarantine_cross_source_outliers("CRWD", rows, store)

    assert kept == rows, "a uniform vendor-side re-adjustment must not be quarantined"


def test_quarantine_uses_db_neighbor_to_confirm_isolation_for_single_row_window(store):
    """Single-row fetch window (an edge on both sides): no in-window neighbor
    exists, so isolation falls back to the vendor's OWN stored neighbor row in
    the DB. Here that neighbor (~97, matching the real 2026-08-24 incident's
    'Yahoo's own neighboring days' value) confirms the fetched 48.19 row truly
    is isolated -> still quarantined, this time on explicit DB evidence rather
    than the no-evidence fallback."""
    src = YFinancePricesSource()
    run_id = store.allocate_run_id()
    store.log_run_start(run_id, source="yfinance")
    d = date(2026, 7, 31)
    _insert_other_source_close(store, run_id, "MNST", d, 96.38)
    # the vendor's own prior-night row, one day earlier, at the real scale
    store.conn.execute(
        "INSERT INTO prices (ticker, date, open, high, low, close, adj_close, "
        "volume, source, run_id) VALUES ('MNST', ?, 97, 98, 96, 97.0, 97.0, 1000, "
        "'yfinance', ?)",
        [date(2026, 7, 30), run_id],
    )

    kept = src._quarantine_cross_source_outliers(
        "MNST", [_row("MNST", d, 48.19, run_id=run_id)], store
    )

    assert kept == []


def test_quarantine_keeps_single_row_when_db_neighbor_agrees_with_fetched_value(store):
    """The false-positive this fix exists to prevent, in single-row form: the
    fetched row diverges >30% from another source (satisfies the old rule
    alone) but AGREES with the vendor's own stored neighbor row -- i.e. it's
    a genuine vendor-side re-adjustment/move, not an isolated phantom. Must be
    KEPT."""
    src = YFinancePricesSource()
    run_id = store.allocate_run_id()
    store.log_run_start(run_id, source="yfinance")
    d = date(2026, 6, 2)
    _insert_other_source_close(store, run_id, "CRWD", d, 768.84, source="alpaca")
    # the vendor's own already-resynced neighbor row from a prior run
    store.conn.execute(
        "INSERT INTO prices (ticker, date, open, high, low, close, adj_close, "
        "volume, source, run_id) VALUES ('CRWD', ?, 192, 193, 191, 192.0, 192.0, "
        "1000, 'yfinance', ?)",
        [date(2026, 6, 1), run_id],
    )

    row = _row("CRWD", d, 192.24, run_id=run_id)
    kept = src._quarantine_cross_source_outliers("CRWD", [row], store)

    assert kept == [row]


def test_quarantine_one_sided_at_window_edge_when_no_db_neighbor(store):
    """Two-row window, corrupt row at the first/edge position: no in-window
    prev and no DB history either (a brand-new ticker), so isolation is
    one-sided on the single available side (in-window next). That one side
    diverging is enough -- AND-ing against a nonexistent second side would
    wrongly require evidence that can't exist."""
    src = YFinancePricesSource()
    run_id = store.allocate_run_id()
    store.log_run_start(run_id, source="yfinance")
    d_corrupt, d_good = date(2026, 7, 30), date(2026, 7, 31)
    _insert_other_source_close(store, run_id, "ZZZZ", d_corrupt, 100.0)
    _insert_other_source_close(store, run_id, "ZZZZ", d_good, 100.0)

    corrupt = _row("ZZZZ", d_corrupt, 40.0, run_id=run_id)
    good = _row("ZZZZ", d_good, 100.0, run_id=run_id)
    kept = src._quarantine_cross_source_outliers("ZZZZ", [corrupt, good], store)

    assert kept == [good]


# ---------------------------------------------------------------------------
# 2026-09-18 incident: Yahoo returned the 9/18 bar with NaN closes for ALL 264
# tickers (a vendor-side hiccup, not a network outage — the bulk call
# succeeded and returned a DataFrame). `_insert_single_ticker` did
# `float(r[...])` unconditionally, so NaN-close rows were inserted as-is.
# DuckDB orders/compares NaN as the largest value (nan > 0.5 is TRUE), so
# `no_unjustified_extreme_moves` and `no_cross_source_price_divergence` both
# flagged the NaN rows as extreme moves and froze the whole night's rebalance.
#
# Fix layer 1: never insert a row whose close is NaN/None — drop it, with one
# WARNING per ticker-run reporting the count. If adj_close alone is NaN, store
# NULL (matching alpaca's existing NULL-placeholder convention) instead of NaN.
# ---------------------------------------------------------------------------


def _nan_close_df(dates: list[str]) -> pd.DataFrame:
    n = len(dates)
    return pd.DataFrame(
        {
            "Open": [100.0] * n,
            "High": [101.0] * n,
            "Low": [99.0] * n,
            "Close": [float("nan")] * n,
            "Adj Close": [float("nan")] * n,
            "Volume": [1_000_000] * n,
        },
        index=pd.to_datetime(dates),
    )


def test_drops_row_with_nan_close(store):
    src = YFinancePricesSource(lookback_days=5)
    run_id = store.allocate_run_id()
    store.log_run_start(run_id, source="yfinance")

    with patch("sma.ingest.sources.yfinance_prices.yf.download",
               return_value=_nan_close_df(["2026-09-18"])):
        result = src.fetch(["AAPL"], date(2026, 9, 18), store, run_id)

    assert result.rows_inserted == 0
    rows = store.conn.execute("SELECT * FROM prices WHERE ticker='AAPL'").fetchall()
    assert rows == [], "a NaN-close row must never be inserted"


def test_drops_row_with_nan_close_logs_one_warning_with_count(store, monkeypatch):
    warnings = []
    monkeypatch.setattr(
        yfinance_prices_mod.logger, "warning",
        lambda *a, **kw: warnings.append(a),
    )
    src = YFinancePricesSource(lookback_days=5)
    run_id = store.allocate_run_id()
    store.log_run_start(run_id, source="yfinance")

    df = _nan_close_df(["2026-09-17", "2026-09-18"])  # 2 NaN-close rows, one ticker
    with patch("sma.ingest.sources.yfinance_prices.yf.download", return_value=df):
        src.fetch(["AAPL"], date(2026, 9, 18), store, run_id)

    drop_warnings = [w for w in warnings if "AAPL" in str(w) and "NaN" in str(w)]
    assert len(drop_warnings) == 1, f"expected exactly one drop warning, got: {warnings}"
    assert "2" in str(drop_warnings[0]), "the warning must report the dropped-row count"


def test_stores_null_when_only_adj_close_is_nan(store):
    """Close is real, adj_close alone is NaN -> store NULL, not NaN, so it is
    never mistaken for a real (if degenerate) adjusted price downstream."""
    df = pd.DataFrame(
        {
            "Open": [150.0], "High": [152.0], "Low": [149.0], "Close": [151.5],
            "Adj Close": [float("nan")], "Volume": [10_000_000],
        },
        index=pd.to_datetime(["2026-09-18"]),
    )
    src = YFinancePricesSource(lookback_days=5)
    run_id = store.allocate_run_id()
    store.log_run_start(run_id, source="yfinance")

    with patch("sma.ingest.sources.yfinance_prices.yf.download", return_value=df):
        result = src.fetch(["AAPL"], date(2026, 9, 18), store, run_id)

    assert result.rows_inserted == 1
    close, adj_close = store.conn.execute(
        "SELECT close, adj_close FROM prices WHERE ticker='AAPL'"
    ).fetchone()
    assert close == 151.5
    assert adj_close is None, "NaN adj_close must be stored as NULL, never as NaN"


# ---------------------------------------------------------------------------
# Fix layer 2: if a fetch yields NaN closes for the ASOF date on more than
# ~20% of tickers, the source itself must be treated as FAILED for this run
# (status='error') rather than 'ok' — otherwise enough_sources_succeeded and
# the run_id-scoped coverage logic both see a healthy yfinance run even
# though its data for tonight is worthless, and the OTHER price source never
# gets the chance to be recognized as the one actually carrying the night.
# ---------------------------------------------------------------------------


def _bulk_df_with_nan_ratio(tickers: list[str], asof: str, nan_count: int) -> pd.DataFrame:
    frames = {}
    for i, t in enumerate(tickers):
        close = float("nan") if i < nan_count else 100.0 + i
        frames[t] = pd.DataFrame(
            {
                "Open": [100.0], "High": [101.0], "Low": [99.0], "Close": [close],
                "Adj Close": [close], "Volume": [1_000_000],
            },
            index=pd.to_datetime([asof]),
        )
    return pd.concat(frames, axis=1)


def test_source_marked_error_when_nan_asof_ratio_exceeds_20_percent(store):
    """2/5 = 40% > 20% -> the whole source must report status='error', even
    though rows for the 3 good tickers still land (best-effort; the other
    source is what's relied on to carry the night)."""
    src = YFinancePricesSource(lookback_days=1)
    run_id = store.allocate_run_id()
    store.log_run_start(run_id, source="yfinance")
    tickers = [f"T{i}" for i in range(5)]
    df = _bulk_df_with_nan_ratio(tickers, "2026-09-18", nan_count=2)

    with patch("sma.ingest.sources.yfinance_prices.yf.download", return_value=df):
        result = src.fetch(tickers, date(2026, 9, 18), store, run_id)

    assert result.status == "error"
    assert result.error is not None and "nan" in result.error.lower()
    good_rows = store.conn.execute(
        "SELECT COUNT(*) FROM prices WHERE source='yfinance'"
    ).fetchone()[0]
    assert good_rows == 3, "good tickers' rows are still inserted; only the STATUS is failed"


def test_source_stays_ok_when_nan_asof_ratio_under_20_percent(store):
    """1/10 = 10% < 20% -> tolerated; the source still reports 'ok'."""
    src = YFinancePricesSource(lookback_days=1)
    run_id = store.allocate_run_id()
    store.log_run_start(run_id, source="yfinance")
    tickers = [f"T{i}" for i in range(10)]
    df = _bulk_df_with_nan_ratio(tickers, "2026-09-18", nan_count=1)

    with patch("sma.ingest.sources.yfinance_prices.yf.download", return_value=df):
        result = src.fetch(tickers, date(2026, 9, 18), store, run_id)

    assert result.status == "ok"
    assert result.rows_inserted == 9


def test_source_marked_error_on_nan_ratio_via_per_ticker_fallback_path(store):
    """The same ratio check must also apply on the per-ticker fallback path
    (triggered when the bulk download itself raises), not just the bulk path
    — both paths funnel through the same _insert_single_ticker."""
    src = YFinancePricesSource(lookback_days=1)
    run_id = store.allocate_run_id()
    store.log_run_start(run_id, source="yfinance")
    tickers = ["AAA", "BBB", "CCC"]  # 1/3 = 33% > 20%

    def fake_download(t, **kw):
        if isinstance(t, list):
            raise RuntimeError("bulk flaked")
        close = float("nan") if t == "AAA" else 100.0
        return pd.DataFrame(
            {
                "Open": [100.0], "High": [101.0], "Low": [99.0], "Close": [close],
                "Adj Close": [close], "Volume": [1_000_000],
            },
            index=pd.to_datetime(["2026-09-18"]),
        )

    with patch("sma.ingest.sources.yfinance_prices.yf.download", side_effect=fake_download):
        result = src.fetch(tickers, date(2026, 9, 18), store, run_id)

    assert result.status == "error"
    assert result.error is not None and "nan" in result.error.lower()


def test_quarantine_keeps_both_adjacent_corrupt_rows(store):
    """Decision, documented: two ADJACENT corrupt rows that happen to agree
    with each other. Each is the other's same-vendor neighbor on one side, and
    that side shows no divergence -- the same signature a legitimate uniform
    split shift produces. Isolation requires divergence from BOTH available
    neighbors (AND), so this check cannot tell 'two corrupt rows agreeing with
    each other' apart from 'two re-adjusted rows agreeing with each other'
    using same-vendor adjacency alone, and conservatively KEEPS both rather
    than risk breaking a genuine multi-row split resync (the bug this fix
    closes) to catch a rarer double-corruption. Each row still individually
    diverges from the good row on its OTHER side, so this is a real, accepted
    false negative, not a blind spot in the isolation math."""
    src = YFinancePricesSource()
    run_id = store.allocate_run_id()
    store.log_run_start(run_id, source="yfinance")
    d1, d2, d3, d4 = (
        date(2026, 7, 28), date(2026, 7, 29), date(2026, 7, 30), date(2026, 7, 31)
    )
    for d in (d1, d2, d3, d4):
        _insert_other_source_close(store, run_id, "MNST", d, 100.0)

    rows = [
        _row("MNST", d1, 100.0, run_id=run_id),  # good
        _row("MNST", d2, 48.0, run_id=run_id),   # corrupt A
        _row("MNST", d3, 48.0, run_id=run_id),   # corrupt B, agrees with A
        _row("MNST", d4, 100.0, run_id=run_id),  # good
    ]
    kept = src._quarantine_cross_source_outliers("MNST", rows, store)

    assert kept == rows, "adjacent mutually-agreeing corrupt rows are kept, by design"
