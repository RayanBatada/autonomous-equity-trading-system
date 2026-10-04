"""1-minute IEX bars into `prices_intraday` (migration 11).

    python -m sma.ingest intraday [--asof-date D] [--tickers SPY,XLK,...]

Default ticker list is config `intraday.tickers` (SPY + the 11 SPDR sector
ETFs) plus, when `intraday.include_held`, every name the paper_fills ledger
currently holds. Feed is Alpaca IEX (free tier). IEX is one venue, so a
quiet minute simply has no bar: expect ~270-390 regular-hours bars per ETF
per day, not 390 (measured 2026-09-25: 4,115 rows for the 12 default ETFs).
IEX minute history starts 2020-07-27.

Rows are keyed (ticker, ts, source) with `ts` = bar START in UTC (naive), and
written with INSERT OR REPLACE, so rerunning a day is idempotent: the last
run's values and run_id win, row count does not change.

Scheduled as com.sma.ingest.intraday at 15:41 ET (bars through ~15:40 for the
close session). That job ships UNLOADED; see ops/launchd/README.md.
"""

from __future__ import annotations

import math
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import click
from loguru import logger

from sma.ingest.sources.base import IngestResult

ET = ZoneInfo("America/New_York")
SOURCE = "alpaca_iex"
LOG_SOURCE = "alpaca_intraday"
SENTINEL_LABEL = "com.sma.ingest.intraday"
_CHUNK = 50  # symbols per bars request


def held_tickers(conn) -> list[str]:
    """Names the paper_fills ledger nets long (same arithmetic as reconcile's
    ledger-vs-broker check). Read from the DB, never the broker."""
    rows = conn.execute(
        "SELECT ticker FROM paper_fills GROUP BY ticker "
        "HAVING SUM(CASE WHEN UPPER(side) = 'BUY' THEN filled_shares "
        "ELSE -filled_shares END) > 1e-9 ORDER BY ticker"
    ).fetchall()
    return [r[0] for r in rows]


def resolve_tickers(configured: list[str], held: list[str]) -> list[str]:
    """Configured list first (order kept), then held names not already in it."""
    out: list[str] = []
    for t in [*configured, *held]:
        t = t.strip().upper()
        if t and t not in out:
            out.append(t)
    return out


def _day_bounds_utc(asof: date) -> tuple[datetime, datetime]:
    start = datetime.combine(asof, datetime.min.time(), tzinfo=ET)
    return start.astimezone(UTC), (start + timedelta(days=1)).astimezone(UTC)


def fetch_intraday(
    *, tickers: list[str], asof_date: date, store, run_id: int, client
) -> IngestResult:
    """Fetch asof_date's 1-minute IEX bars (00:00-24:00 ET, so pre/post
    market prints IEX has are kept) and upsert them. `client` is an alpaca-py
    StockHistoricalDataClient (a mock in tests)."""
    from alpaca.data.enums import DataFeed
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    start, end = _day_bounds_utc(asof_date)
    to_alpaca = {t: t.replace("-", ".") for t in tickers}
    from_alpaca = {v: k for k, v in to_alpaca.items()}
    syms = list(to_alpaca.values())

    rows: list[tuple] = []
    dropped = 0
    for i in range(0, len(syms), _CHUNK):
        chunk = syms[i : i + _CHUNK]
        resp = client.get_stock_bars(
            StockBarsRequest(
                symbol_or_symbols=chunk,
                timeframe=TimeFrame.Minute,
                start=start,
                end=end,
                feed=DataFeed.IEX,
            )
        )
        data = getattr(resp, "data", None) or {}
        for sym, bars in data.items():
            ticker = from_alpaca.get(sym, sym)
            for bar in bars:
                close = bar.close
                if close is None or (isinstance(close, float) and math.isnan(close)):
                    dropped += 1
                    continue
                ts = bar.timestamp
                if ts.tzinfo is not None:
                    ts = ts.astimezone(UTC).replace(tzinfo=None)
                rows.append((
                    ticker, ts, float(bar.open), float(bar.high), float(bar.low),
                    float(close), int(bar.volume or 0), SOURCE, run_id,
                ))
    if dropped:
        logger.warning("intraday: dropped {} bar(s) with NaN/None close", dropped)
    if rows:
        store.conn.executemany(
            "INSERT OR REPLACE INTO prices_intraday "
            "(ticker, ts, open, high, low, close, volume, source, run_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
    return IngestResult(LOG_SOURCE, len(rows), "ok", None)


def _build_data_client(settings):
    from alpaca.data.historical import StockHistoricalDataClient

    from sma.live.alpaca_client import with_default_timeout

    s = settings.secrets
    return with_default_timeout(
        StockHistoricalDataClient(s.alpaca_api_key, s.alpaca_api_secret)
    )


def run_intraday(
    *, asof: date, tickers: list[str] | None, db: str, config: str, client=None
) -> dict:
    """Job body: writer_lock -> Store -> fetch -> ingest_log -> sentinel."""
    from sma.config import load_settings
    from sma.ingest.store import Store
    from sma.locks import writer_lock
    from sma.sentinels import write_sentinel

    settings = load_settings(config_path=Path(config))
    cfg = settings.intraday
    client = client if client is not None else _build_data_client(settings)

    with writer_lock(label="ingest_intraday"):
        store = Store(path=db).connect()
        try:
            if tickers:
                resolved = resolve_tickers(tickers, [])
            else:
                held = held_tickers(store.conn) if cfg.include_held else []
                resolved = resolve_tickers(cfg.tickers, held)
            run_id = store.allocate_run_id()
            store.log_run_start(run_id, LOG_SOURCE)
            try:
                result = fetch_intraday(
                    tickers=resolved, asof_date=asof, store=store, run_id=run_id, client=client
                )
            except Exception as e:
                store.log_run_end(run_id, LOG_SOURCE, 0, "error", str(e)[:500])
                raise
            store.log_run_end(run_id, LOG_SOURCE, result.rows_inserted, result.status, None)
            day_rows = store.conn.execute(
                "SELECT COUNT(*) FROM prices_intraday WHERE source = ? "
                "AND ts >= ? AND ts < ?",
                [SOURCE, *(t.replace(tzinfo=None) for t in _day_bounds_utc(asof))],
            ).fetchone()[0]
        finally:
            store.close()
        payload = {
            "label": SENTINEL_LABEL,
            "asof": asof.isoformat(),
            "completed_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "run_id": run_id,
            "tickers": resolved,
            "rows_fetched": result.rows_inserted,
            "rows_for_day": int(day_rows),
        }
        write_sentinel(label=SENTINEL_LABEL, asof=asof, payload=payload)
    return payload


@click.command("intraday")
@click.option("--asof-date", default=None, help="ISO date (default: today ET)")
@click.option("--tickers", default=None, help="Comma list; overrides config + held names")
@click.option("--db", default="data/sma.duckdb", type=click.Path())
@click.option("--config", default="config.yaml", type=click.Path(exists=True))
def intraday_cmd(asof_date, tickers, db, config):
    """Fetch one day of 1-minute IEX bars into prices_intraday (idempotent)."""
    asof = date.fromisoformat(asof_date) if asof_date else datetime.now(ET).date()
    tick_list = [t for t in (tickers or "").split(",") if t.strip()] or None
    payload = run_intraday(asof=asof, tickers=tick_list, db=db, config=config)
    click.echo(
        f"intraday[{asof.isoformat()}]: {payload['rows_fetched']} bars fetched for "
        f"{len(payload['tickers'])} tickers; {payload['rows_for_day']} rows on file for the day"
    )
