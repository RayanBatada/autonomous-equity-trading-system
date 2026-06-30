"""Shared helpers: deduped full-history price con + PIT membership.

The prod DB stores pre-2023 as source='yfinance_hist' and 2023+ as 'yfinance'
(+ 'alpaca'). classify_regimes / per_feature_ic filter source='yfinance', so
they silently drop ALL 2018-2022 data. To span full history we build an
in-memory duckdb whose `prices` table is deduped (one row per ticker,date,
preferring yfinance > yfinance_hist > alpaca) and source forced to 'yfinance'
so the existing sma helpers (which hardcode that filter) work unchanged.
"""
import sys
from datetime import timedelta

import duckdb
import pandas as pd

sys.path.insert(0, "/Users/youruser/.sma-pit")

# Durable de-biased copy: prod prices + yfinance-backfilled decliners.
PROD_DB = "/Users/youruser/.sma-pit/sma-pit.duckdb"


def make_memcon(db_path=PROD_DB, source_priority=("yfinance", "yfinance_hist", "alpaca")):
    """In-memory duckdb with a deduped `prices` table (source='yfinance')."""
    mem = duckdb.connect(":memory:")
    mem.execute(f"ATTACH '{db_path}' AS prod (READ_ONLY)")
    case_prio = "CASE source " + " ".join(
        f"WHEN '{s}' THEN {i}" for i, s in enumerate(source_priority)
    ) + " ELSE 99 END"
    mem.execute(f"""
        CREATE TABLE prices AS
        SELECT ticker, date, open, high, low, close, adj_close, volume,
               'yfinance' AS source
        FROM (
            SELECT *, ROW_NUMBER() OVER (
                PARTITION BY ticker, date ORDER BY {case_prio}
            ) AS rn
            FROM prod.prices
        ) WHERE rn = 1
    """)
    mem.execute("DETACH prod")
    return mem


def load_prices_df(mem, start, end, pad_days=420):
    df = mem.execute(
        "SELECT * FROM prices WHERE date BETWEEN ? AND ?",
        [start - timedelta(days=pad_days), end],
    ).df()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df
