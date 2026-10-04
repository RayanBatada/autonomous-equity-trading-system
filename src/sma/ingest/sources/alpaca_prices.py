"""Alpaca daily bars ingestion.

Uses alpaca-py's StockHistoricalDataClient with the IEX feed (free with
paper account). Inserts rows with source='alpaca'.

Note: Alpaca does not provide an adjusted close, so adj_close is set equal
to close. Downstream code should prefer yfinance's adj_close when both are
available.
"""

import math
import re
from datetime import date, timedelta

from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from loguru import logger

from sma.ingest.sources.base import IngestResult
from sma.ingest.store import Store


class AlpacaPricesSource:
    name = "alpaca"

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        base_url: str,
        lookback_days: int = 1,
    ):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url
        self.lookback_days = lookback_days
        # Same default timeout as the broker client (flaw hunt 2026-10-01 A2):
        # a dead connection inside ingest would hold the writer lock forever.
        from sma.live.alpaca_client import with_default_timeout

        self._client = with_default_timeout(StockHistoricalDataClient(api_key, api_secret))

    def fetch(
        self,
        tickers: list[str],
        asof_date: date,
        store: Store,
        run_id: int,
    ) -> IngestResult:
        start = asof_date - timedelta(days=self.lookback_days)
        end = asof_date + timedelta(days=1)

        # Alpaca uses dot share-class notation (BRK.B) where the rest of the
        # system is yfinance-canonical (BRK-B). Translate on request and map
        # responses back so DB tickers stay canonical. Without this, ONE dashed
        # ticker fails the ENTIRE bulk request — the source was 100% dead from
        # 6/12 (BRK-B added to the universe) until the 2026-07-01 review.
        to_alpaca = {t: t.replace("-", ".") for t in tickers}
        from_alpaca = {v: k for k, v in to_alpaca.items()}

        def _request(symbols: list[str]):
            req = StockBarsRequest(
                symbol_or_symbols=symbols,
                timeframe=TimeFrame.Day,
                start=start.isoformat(),
                end=end.isoformat(),
                feed=DataFeed.IEX,  # free-tier paper accounts only allow IEX
            )
            return self._client.get_stock_bars(req)

        symbols = list(to_alpaca.values())
        try:
            try:
                resp = _request(symbols)
            except Exception as e:
                # Defense in depth: if Alpaca still names a rejected symbol,
                # drop the offender(s) and retry once rather than losing the
                # whole source for the night.
                named = set(re.findall(r"invalid symbol: ([A-Za-z0-9.\-]+)", str(e)))
                bad = [s for s in symbols if s in named]
                if not bad:
                    raise
                logger.warning("alpaca rejected {}; retrying without them", bad)
                resp = _request([s for s in symbols if s not in bad])
        except Exception as e:
            logger.warning("alpaca bars request failed: {}", e)
            return IngestResult(self.name, 0, "error", str(e))

        rows = []
        nan_dropped_by_ticker: dict[str, int] = {}
        data = getattr(resp, "data", None) or {}
        for alpaca_ticker, bars in data.items():
            ticker = from_alpaca.get(alpaca_ticker, alpaca_ticker)
            for bar in bars:
                # 2026-09-18 incident audit: yfinance_prices.py inserted NaN
                # closes unconditionally; apply the same defensive guard here
                # even though a NaN bar.close hasn't been observed from Alpaca
                # -- never insert a row whose close is NaN/None.
                close = bar.close
                if close is None or (isinstance(close, float) and math.isnan(close)):
                    nan_dropped_by_ticker[ticker] = nan_dropped_by_ticker.get(ticker, 0) + 1
                    continue
                ts = bar.timestamp
                d = ts.date() if hasattr(ts, "date") else ts
                rows.append((
                    ticker,
                    d,
                    float(bar.open),
                    float(bar.high),
                    float(bar.low),
                    float(close),
                    None,  # adj_close: Alpaca is UNADJUSTED — store NULL so it's
                    # never mistaken for split/div-adjusted (yfinance is canonical)
                    int(bar.volume),
                    self.name,
                    run_id,
                ))

        for ticker, n in nan_dropped_by_ticker.items():
            logger.warning(
                "alpaca: dropped {} row(s) with NaN/None close for {} (run_id={})",
                n, ticker, run_id,
            )

        if rows:
            store.conn.executemany(
                "INSERT OR REPLACE INTO prices "
                "(ticker, date, open, high, low, close, adj_close, volume, source, run_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
        return IngestResult(self.name, len(rows), "ok", None)
