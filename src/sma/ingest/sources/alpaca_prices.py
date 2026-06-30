"""Alpaca daily bars ingestion.

Uses alpaca-py's StockHistoricalDataClient with the IEX feed (free with
paper account). Inserts rows with source='alpaca'.

Note: Alpaca does not provide an adjusted close, so adj_close is set equal
to close. Downstream code should prefer yfinance's adj_close when both are
available.
"""

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
        self._client = StockHistoricalDataClient(api_key, api_secret)

    def fetch(
        self,
        tickers: list[str],
        asof_date: date,
        store: Store,
        run_id: int,
    ) -> IngestResult:
        start = asof_date - timedelta(days=self.lookback_days)
        end = asof_date + timedelta(days=1)

        req = StockBarsRequest(
            symbol_or_symbols=tickers,
            timeframe=TimeFrame.Day,
            start=start.isoformat(),
            end=end.isoformat(),
            feed=DataFeed.IEX,  # free-tier paper accounts only allow IEX, not SIP
        )
        try:
            resp = self._client.get_stock_bars(req)
        except Exception as e:
            logger.warning("alpaca bars request failed: {}", e)
            return IngestResult(self.name, 0, "error", str(e))

        rows = []
        data = getattr(resp, "data", None) or {}
        for ticker, bars in data.items():
            for bar in bars:
                ts = bar.timestamp
                d = ts.date() if hasattr(ts, "date") else ts
                rows.append((
                    ticker,
                    d,
                    float(bar.open),
                    float(bar.high),
                    float(bar.low),
                    float(bar.close),
                    None,  # adj_close: Alpaca is UNADJUSTED — store NULL so it's
                    # never mistaken for split/div-adjusted (yfinance is canonical)
                    int(bar.volume),
                    self.name,
                    run_id,
                ))

        if rows:
            store.conn.executemany(
                "INSERT OR REPLACE INTO prices "
                "(ticker, date, open, high, low, close, adj_close, volume, source, run_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
        return IngestResult(self.name, len(rows), "ok", None)
