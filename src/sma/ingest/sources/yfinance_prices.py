"""yfinance price ingestion.

Bulk-downloads OHLCV for the universe in one call. On bulk failure, falls
back to per-ticker calls so one bad ticker doesn't nuke the whole run.

Price rows are inserted with source='yfinance'. Re-ingestion for the same
(ticker, date) is handled at the runner level via DELETE WHERE run_id, not
in the source.
"""

from datetime import date, timedelta

import pandas as pd
import yfinance as yf
from loguru import logger

from sma.ingest.sources.base import IngestResult
from sma.ingest.store import Store


class YFinancePricesSource:
    name = "yfinance"

    # Lookback must EXCEED the quality gate's 30-day extreme-moves scan:
    # with a 1-day window, a provider-side back-adjustment (KLAC 10:1 split,
    # 2026-06-11) rescales history the nightly fetch never re-syncs, leaving
    # a permanent 10x discontinuity inside the scan that blocks EVERY night
    # for a month. 45 days re-syncs the whole scanned window nightly
    # (same number of yfinance requests; just more upserted rows).
    def __init__(self, lookback_days: int = 45):
        self.lookback_days = lookback_days

    def fetch(
        self,
        tickers: list[str],
        asof_date: date,
        store: Store,
        run_id: int,
    ) -> IngestResult:
        start = asof_date - timedelta(days=self.lookback_days)
        end = asof_date + timedelta(days=1)

        rows_inserted = 0
        network_error = False
        try:
            df = yf.download(
                tickers,
                start=start.isoformat(),
                end=end.isoformat(),
                progress=False,
                group_by="ticker",
                threads=True,
                auto_adjust=False,
                multi_level_index=False,
            )
            rows_inserted = self._insert_dataframe(tickers, df, store, run_id)
        except Exception as e:
            logger.warning("bulk yfinance download failed: {}; falling back per-ticker", e)
            per_ticker_failures = 0
            for t in tickers:
                try:
                    sub = yf.download(
                        t,
                        start=start.isoformat(),
                        end=end.isoformat(),
                        progress=False,
                        auto_adjust=False,
                multi_level_index=False,
                    )
                    rows_inserted += self._insert_single_ticker(t, sub, store, run_id)
                except Exception as inner:
                    per_ticker_failures += 1
                    logger.warning("yfinance per-ticker failed for {}: {}", t, inner)
            # Bulk threw AND every per-ticker fetch also threw → a real network/DNS
            # failure, not just an empty (no-data) response.
            network_error = per_ticker_failures == len(tickers)

        # A network/DNS outage (every fetch threw) previously still returned
        # status="ok", letting enough_sources_succeeded pass on a dead network and
        # masking a total price-ingest failure (the 6/1-6/3 freeze). Report "error"
        # only then — an empty (no-data) response is legitimately "ok".
        ok = not (rows_inserted == 0 and network_error)
        return IngestResult(
            source=self.name,
            rows_inserted=rows_inserted,
            status="ok" if ok else "error",
            error=None if ok else (
                "yfinance inserted 0 rows (all fetches failed — likely network/DNS)"
            ),
        )

    def _insert_dataframe(
        self,
        tickers: list[str],
        df: pd.DataFrame,
        store: Store,
        run_id: int,
    ) -> int:
        if df.empty:
            return 0
        if len(tickers) == 1:
            return self._insert_single_ticker(tickers[0], df, store, run_id)
        total = 0
        for t in tickers:
            try:
                sub = df[t].dropna(how="all")
            except KeyError:
                continue
            total += self._insert_single_ticker(t, sub, store, run_id)
        return total

    def _insert_single_ticker(
        self,
        ticker: str,
        df: pd.DataFrame,
        store: Store,
        run_id: int,
    ) -> int:
        if df is None or df.empty:
            return 0
        # Defensive flatten in case multi_level_index=False isn't honored
        # (older yfinance, group_by behavior, etc). Pick the level that
        # contains known OHLCV field names.
        if isinstance(df.columns, pd.MultiIndex):
            df = df.copy()
            known = {"open", "high", "low", "close", "adj close", "volume"}
            level0_hits = sum(
                1 for v in df.columns.get_level_values(0)
                if str(v).lower() in known
            )
            level1_hits = sum(
                1 for v in df.columns.get_level_values(1)
                if str(v).lower() in known
            )
            field_level = 0 if level0_hits >= level1_hits else 1
            df.columns = df.columns.get_level_values(field_level)
        cols = {c.lower().replace(" ", "_"): c for c in df.columns}
        rows = []
        for ts, r in df.iterrows():
            d = ts.date() if hasattr(ts, "date") else ts
            rows.append((
                ticker,
                d,
                float(r[cols.get("open", "Open")]) if cols.get("open", "Open") in r else None,
                float(r[cols.get("high", "High")]) if cols.get("high", "High") in r else None,
                float(r[cols.get("low", "Low")]) if cols.get("low", "Low") in r else None,
                float(r[cols.get("close", "Close")]) if cols.get("close", "Close") in r else None,
                (
                    float(r[cols.get("adj_close", "Adj Close")])
                    if cols.get("adj_close", "Adj Close") in r
                    else None
                ),
                int(r[cols.get("volume", "Volume")]) if cols.get("volume", "Volume") in r else None,
                self.name,
                run_id,
            ))
        if not rows:
            return 0
        store.conn.executemany(
            "INSERT OR REPLACE INTO prices "
            "(ticker, date, open, high, low, close, adj_close, volume, source, run_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        return len(rows)
