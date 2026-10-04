"""Repair split-inconsistent yfinance history (2026-10-01).

`python -m sma.ingest repair-splits [--tickers A,B | --all-flagged] [--dry-run]`

For each ticker: re-fetch the FULL yfinance history in one request, delete
that ticker's source='yfinance' rows and insert the refetch, all in one
transaction. The refetch uses auto_adjust=False EXPLICITLY, the same
semantics as the nightly YFinancePricesSource, so repaired rows sit on the
same scale the nightly window keeps writing:
    close     = Yahoo "Close": split-adjusted, NOT dividend-adjusted
    adj_close = Yahoo "Adj Close": split- AND dividend-adjusted
(auto_adjust=True would put the dividend-adjusted series in `close` and drop
"Adj Close", i.e. a different scale from every nightly row.)

Alpaca rows stay RAW on purpose: alpaca bars are the same share scale as our
broker fills, which is exactly what fill_counterfactuals needs (CRWD fills at
682 vs yfinance's split-adjusted 168). Rescaling them would break that. What
must hold instead is that no feature reads them: the feature query takes
yfinance first and requires adj_close IS NOT NULL, and AlpacaPricesSource has
written adj_close NULL since 2026-05-18. Legacy alpaca rows (2023-04-25 ..
2026-06-03) still carry adj_close = raw close, which let 15 raw BRK.B rows
reach the feature builder. The repair NULLs those (`null_legacy_alpaca_adj`).

A ticker is only committed when the refetch does not make the audit worse
(feature-affecting flags after <= before); otherwise the transaction rolls
back and the ticker is reported. Writes need the writer_lock (the CLI takes
it); --dry-run works on an in-memory copy of each ticker's rows and never
writes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, timedelta

import pandas as pd
from loguru import logger

from sma.ingest.split_audit import find_split_inconsistencies
from sma.ingest.store import Store

REPAIR_LOG_SOURCE = "yfinance_repair_splits"
SENTINEL_LABEL = "com.sma.ingest.repair_splits"
FULL_HISTORY_START = "2016-01-01"  # same start as the nightly self-heal refetch
_PRICE_COLS = "ticker, date, open, high, low, close, adj_close, volume, source, run_id"


def fetch_full_history(ticker: str, *, end: date) -> pd.DataFrame:
    import yfinance as yf

    return yf.download(
        ticker,
        start=FULL_HISTORY_START,
        end=(end + timedelta(days=1)).isoformat(),
        progress=False,
        auto_adjust=False,  # explicit: Close split-adjusted, Adj Close also div-adjusted
        actions=False,
        multi_level_index=False,
    )


@dataclass
class TickerRepair:
    ticker: str
    status: str = "pending"
    rows_before: int = 0
    rows_after: int = 0
    flags_before: list[str] = field(default_factory=list)
    flags_after: list[str] = field(default_factory=list)
    # date -> (yfinance close before, yfinance close after, alpaca close)
    samples: dict = field(default_factory=dict)


def _flags(conn, ticker: str) -> list:
    return find_split_inconsistencies(conn, tickers=[ticker])


def _sample_dates(conn, ticker: str, flags) -> list[date]:
    out: set[date] = set()
    for f in flags:
        out.add(f.date)
        prev = conn.execute(
            "SELECT MAX(date) FROM prices WHERE ticker = ? AND date < ?", [ticker, f.date]
        ).fetchone()[0]
        if prev is not None:
            out.add(prev)
    return sorted(out)


def _closes(conn, ticker: str, dates: list[date]) -> dict:
    if not dates:
        return {}
    rows = conn.execute(
        "SELECT date, source, close FROM prices WHERE ticker = ? AND date = ANY(?) "
        "AND source IN ('yfinance', 'alpaca')",
        [ticker, dates],
    ).fetchall()
    out: dict = {}
    for d, s, c in rows:
        out.setdefault(d, {})[s] = c
    return out


def _copy_ticker_to_memory(store: Store, ticker: str) -> Store:
    """A private in-memory Store holding just this ticker's price rows. Built
    without Store.connect() (whose writable path demands the writer_lock even
    for :memory:): nothing here touches a file, so a dry run stays lock-free."""
    import duckdb

    mem = Store(":memory:")
    mem.conn = duckdb.connect(":memory:")
    mem._apply_migrations()
    rows = store.conn.execute(
        f"SELECT {_PRICE_COLS} FROM prices WHERE ticker = ?", [ticker]
    ).fetchall()
    if rows:
        mem.conn.executemany(
            f"INSERT INTO prices ({_PRICE_COLS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows
        )
    return mem


def null_legacy_alpaca_adj(store: Store, *, dry_run: bool) -> int:
    """NULL adj_close on alpaca rows (raw, unadjusted) so the feature query's
    `adj_close IS NOT NULL` can never serve one. Returns rows affected."""
    n = store.conn.execute(
        "SELECT COUNT(*) FROM prices WHERE source = 'alpaca' AND adj_close IS NOT NULL"
    ).fetchone()[0]
    if n and not dry_run:
        store.conn.execute(
            "UPDATE prices SET adj_close = NULL WHERE source = 'alpaca' AND adj_close IS NOT NULL"
        )
    return int(n)


def repair_ticker(
    store: Store,
    ticker: str,
    *,
    run_id: int,
    dry_run: bool,
    fetch_fn: Callable[[str], pd.DataFrame],
    min_coverage: float = 0.9,
) -> TickerRepair:
    from sma.ingest.sources.yfinance_prices import YFinancePricesSource

    rep = TickerRepair(ticker=ticker)
    target = _copy_ticker_to_memory(store, ticker) if dry_run else store
    try:
        conn = target.conn
        rep.rows_before = conn.execute(
            "SELECT COUNT(*) FROM prices WHERE ticker = ? AND source = 'yfinance'", [ticker]
        ).fetchone()[0]
        before = _flags(conn, ticker)
        rep.flags_before = [f.describe() for f in before]
        dates = _sample_dates(conn, ticker, before)
        before_px = _closes(conn, ticker, dates)

        try:
            df = fetch_fn(ticker)
        except Exception as e:  # network etc: leave the ticker untouched
            rep.status = f"skipped: fetch failed ({e})"
            return rep
        n_fetched = 0 if df is None or df.empty else int(len(df))
        if n_fetched == 0:
            rep.status = "skipped: refetch returned no rows"
            return rep
        if n_fetched < min_coverage * rep.rows_before:
            rep.status = (
                f"skipped: refetch has {n_fetched} rows vs {rep.rows_before} stored "
                f"(<{min_coverage:.0%}); not replacing history with a truncated series"
            )
            return rep

        conn.execute("BEGIN TRANSACTION")
        try:
            conn.execute("DELETE FROM prices WHERE ticker = ? AND source = 'yfinance'", [ticker])
            src = YFinancePricesSource()
            rep.rows_after = src._insert_single_ticker(
                ticker, df, target, run_id, detect_drift=False
            )
            after = _flags(conn, ticker)
            rep.flags_after = [f.describe() for f in after]
            worse = sum(f.feature_affecting for f in after) > sum(
                f.feature_affecting for f in before
            )
            after_px = _closes(conn, ticker, dates)
            if rep.rows_after == 0 or worse:
                conn.execute("ROLLBACK")
                rep.status = (
                    "rolled back: refetch inserted no rows"
                    if rep.rows_after == 0
                    else "rolled back: refetch is MORE inconsistent than stored history"
                )
            else:
                conn.execute("COMMIT")
                rep.status = "dry-run (not written)" if dry_run else "replaced"
        except Exception:
            conn.execute("ROLLBACK")
            raise
        for d in dates:
            rep.samples[d] = (
                before_px.get(d, {}).get("yfinance"),
                after_px.get(d, {}).get("yfinance"),
                before_px.get(d, {}).get("alpaca"),
            )
        return rep
    finally:
        if dry_run:
            target.close()


def flagged_tickers(conn) -> list[str]:
    """Tickers with a feature-affecting flag anywhere in history (yfinance close
    vs alpaca close, the same rule as the no_split_inconsistency check)."""
    return sorted({f.ticker for f in find_split_inconsistencies(conn) if f.feature_affecting})


def format_report(reps: list[TickerRepair]) -> list[str]:
    lines = []
    for r in reps:
        lines.append(f"{r.ticker}: {r.status} (yfinance rows {r.rows_before} -> {r.rows_after})")
        lines.append(f"  flags before: {r.flags_before or 'none'}")
        lines.append(f"  flags after:  {r.flags_after or 'none'}")
        for d, (b, a, al) in sorted(r.samples.items()):

            def f(x):
                return "-" if x is None else f"{x:.3f}"

            lines.append(f"  {d}  yfinance close {f(b)} -> {f(a)}   alpaca(raw) {f(al)}")
    logger.info("repair-splits: {}", "; ".join(f"{r.ticker}={r.status}" for r in reps))
    return lines
