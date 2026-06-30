"""Earnings blackout helper: reject buys within N calendar days of upcoming earnings.

Phase 5 moved this module from sma.backtest.earnings_blackout to sma.risk so
both simulator and live trading consume the same code path.
"""

from datetime import date, timedelta
from pathlib import Path

import pandas as pd

BLACKOUT_DAYS = 3


def load_upcoming_earnings(
    db_path: Path,
    start_date: date,
    end_date: date,
) -> dict[str, list[date]]:
    """Return ticker -> sorted list of earnings report_dates in [start_date, end_date].

    Used by the simulator and live to enforce a blackout before each upcoming earnings.
    Returns an empty dict if the database doesn't exist or the table is empty.
    """
    if not db_path.exists():
        return {}
    import duckdb
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        df = con.execute("""
            SELECT ticker, report_date
            FROM earnings
            WHERE report_date BETWEEN $start AND $end
            ORDER BY ticker, report_date
        """, {"start": start_date, "end": end_date}).df()
    finally:
        con.close()
    if df.empty:
        return {}
    df["report_date"] = pd.to_datetime(df["report_date"]).dt.date
    return df.groupby("ticker")["report_date"].apply(list).to_dict()


def in_earnings_blackout(
    ticker: str,
    asof_date: date,
    earnings_by_ticker: dict[str, list[date]],
    blackout_days: int = BLACKOUT_DAYS,
) -> bool:
    """Return True if asof_date is within blackout_days calendar days BEFORE
    any of the ticker's upcoming earnings.

    The window is [asof_date, earnings_date]; a buy on the earnings date itself
    is also blocked. Days AFTER earnings are not affected.
    """
    if ticker not in earnings_by_ticker:
        return False
    for er in earnings_by_ticker[ticker]:
        if asof_date <= er <= asof_date + timedelta(days=blackout_days):
            return True
    return False
