"""Read-only split-consistency audit of `prices` (2026-10-01).

    .venv/bin/python scripts/audit_split_consistency.py [--db data/sma.duckdb]
        [--threshold 0.20] [--solo-threshold 0.40] [--since YYYY-MM-DD]

Runs the rule in sma.ingest.split_audit twice: on yfinance close (what the
nightly ingest stores) and on the feature-served series (yfinance > alpaca >
yfinance_hist, adj_close IS NOT NULL: exactly what the model reads), each
against alpaca's raw close. Then two level checks that a return-based rule
cannot see:
  * latest common date: alpaca close / yfinance close should be ~1.0 (after
    the most recent split both are on today's scale). Off by >5% = stale scale.
  * yfinance_hist vs yfinance on overlapping dates: a split after the
    2026-06-12 yfinance_hist snapshot leaves the hist rows on the old scale.
    That only matters where yfinance has no row (hist is then served).
Plus where adj_close differs from close, per source.

Never writes. Opens the DB with read_only_connect.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import date

from sma.db_connect import read_only_connect
from sma.ingest.split_audit import (
    FEATURE_SERIES_SQL,
    YFINANCE_SERIES_SQL,
    find_split_inconsistencies,
)


def _print_flags(title: str, flags) -> list[str]:
    print(f"\n== {title}: {len(flags)} flagged day(s)")
    by_kind = defaultdict(list)
    for f in flags:
        by_kind[f.kind].append(f)
    for kind in sorted(by_kind):
        tag = "" if by_kind[kind][0].feature_affecting else "  (expected: alpaca is raw)"
        print(f"  [{kind}] {len(by_kind[kind])}{tag}")
        for f in by_kind[kind]:
            print(f"    {f.describe()}")
    bad = sorted({f.ticker for f in flags if f.feature_affecting})
    print(f"  feature-affecting tickers: {bad or 'none'}")
    return bad


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/sma.duckdb")
    ap.add_argument("--threshold", type=float, default=0.20)
    ap.add_argument("--solo-threshold", type=float, default=0.40)
    ap.add_argument("--since", default=None)
    ap.add_argument("--tickers", default=None)
    a = ap.parse_args(argv)
    since = date.fromisoformat(a.since) if a.since else None
    tickers = [t.strip().upper() for t in a.tickers.split(",")] if a.tickers else None

    con = read_only_connect(a.db)
    try:
        kw = dict(
            threshold=a.threshold, solo_threshold=a.solo_threshold, since=since, tickers=tickers
        )
        _print_flags(
            "yfinance close vs alpaca close",
            find_split_inconsistencies(con, left_sql=YFINANCE_SERIES_SQL, **kw),
        )
        _print_flags(
            "FEATURE series adj_close vs alpaca close",
            find_split_inconsistencies(con, left_sql=FEATURE_SERIES_SQL, **kw),
        )

        print("\n== level check: alpaca/yfinance close on each ticker's latest common date")
        rows = con.execute("""
            WITH j AS (
                SELECT y.ticker, y.date, a.close / y.close AS ratio,
                       ROW_NUMBER() OVER (PARTITION BY y.ticker ORDER BY y.date DESC) rn
                FROM prices y JOIN prices a
                  ON a.ticker = y.ticker AND a.date = y.date AND a.source = 'alpaca'
                WHERE y.source = 'yfinance' AND y.close > 0 AND a.close > 0
            )
            SELECT ticker, date, ratio FROM j WHERE rn = 1 AND abs(ratio - 1) > 0.05
            ORDER BY ticker
        """).fetchall()
        print(f"  off by >5%: {[(t, str(d), round(r, 3)) for t, d, r in rows] or 'none'}")

        print("\n== yfinance_hist vs yfinance scale on overlapping dates (median ratio)")
        rows = con.execute("""
            SELECT h.ticker, median(h.close / y.close) AS r, count(*) n
            FROM prices h
            JOIN prices y ON y.ticker = h.ticker AND y.date = h.date AND y.source = 'yfinance'
            WHERE h.source = 'yfinance_hist' AND h.close > 0 AND y.close > 0
            GROUP BY 1 HAVING abs(median(h.close / y.close) - 1) > 0.05 ORDER BY 1
        """).fetchall()
        print(
            f"  hist on a different scale (shadowed where yfinance exists): "
            f"{[(t, round(r, 3), n) for t, r, n in rows] or 'none'}"
        )
        served = con.execute("""
            SELECT h.ticker, min(h.date), max(h.date), count(*) FROM prices h
            WHERE h.source = 'yfinance_hist' AND h.adj_close IS NOT NULL
              AND NOT EXISTS (SELECT 1 FROM prices y WHERE y.ticker = h.ticker
                              AND y.date = h.date AND y.source = 'yfinance')
              AND NOT EXISTS (SELECT 1 FROM prices a WHERE a.ticker = h.ticker
                              AND a.date = h.date AND a.source = 'alpaca'
                              AND a.adj_close IS NOT NULL)
            GROUP BY 1
        """).fetchall()
        bad = {t for t, *_ in rows}
        print(
            f"  yfinance_hist rows actually served to features: "
            f"{sum(r[3] for r in served)} across {len(served)} tickers; "
            f"of those on a different scale: {sorted(t for t, *_ in served if t in bad) or 'none'}"
        )

        print("\n== adj_close vs close by source")
        for src, n, nn, diff, lo, hi in con.execute("""
            SELECT source, count(*), count(adj_close),
                   count(*) FILTER (WHERE adj_close IS NOT NULL
                                    AND abs(adj_close / close - 1) > 1e-6),
                   min(date) FILTER (WHERE adj_close IS NOT NULL),
                   max(date) FILTER (WHERE adj_close IS NOT NULL)
            FROM prices WHERE close > 0 GROUP BY 1 ORDER BY 1
        """).fetchall():
            print(
                f"  {src:14s} rows={n} adj_non_null={nn} adj!=close={diff} "
                f"adj_non_null_dates={lo}..{hi}"
            )
        served_alpaca = con.execute("""
            WITH f AS (
                SELECT ticker, date, source, ROW_NUMBER() OVER (
                    PARTITION BY ticker, date ORDER BY CASE source WHEN 'yfinance' THEN 0
                    WHEN 'alpaca' THEN 1 ELSE 2 END) rn
                FROM prices WHERE adj_close IS NOT NULL)
            SELECT ticker, min(date), max(date), count(*) FROM f
            WHERE rn = 1 AND source = 'alpaca' GROUP BY 1
        """).fetchall()
        print(f"  alpaca rows served to features (raw, unadjusted): {served_alpaca or 'none'}")
    finally:
        con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
