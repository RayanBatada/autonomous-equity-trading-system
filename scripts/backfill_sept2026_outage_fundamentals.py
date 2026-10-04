"""One-off: backfill `fundamentals` rows for trading days the 2026-09-08 /
2026-09-10 / 2026-09-14 network outages left with zero or partial
finnhub_fundamentals coverage (2026-08-28..today data-repair task).

Root cause (confirmed via ingest_log + quality logs):
  - 2026-09-08: the CLI's `ingest run --asof-date D --sources
    finnhub_fundamentals` deadline-budget guard (src/sma/ingest/runner.py,
    "deadline budget: skipping overlay source") ALWAYS skips overlay
    sources for a past --asof-date, because sma.schedule.deadline() computes
    a real (now-past) cutoff for any historical trading day and the guard
    has no bypass flag on the CLI. This makes `ingest run` structurally
    unable to backfill an overlay source for a past date -- only
    `backfill-news` (which builds its own IngestRunner without a deadline)
    routes around it, and there is no fundamentals equivalent.
  - 2026-09-10: same-night ingest ran very late (02:00 ET the next day,
    recovering from a stalled/offline host) and hit its OWN deadline budget
    before reaching finnhub_fundamentals (logged: "deadline budget: skipped
    to protect the trading pipeline").
  - 2026-09-14: total network/DNS outage (all sources 0 rows or hard error).
  - 2026-09-15: the recovery run's finnhub_fundamentals call landed only 14
    of 264 tickers (retry-sleep-budget exhaustion working through a 429
    backlog from the prior night), left partial.
  - 2026-09-16: the SAME class of outage recurred (Mac wifi dropped during
    the 22:30 ET ingest window) -- all 7 sources 0 rows or hard error,
    identical signature to 2026-09-14. Added to CANDIDATE_DATES below once
    confirmed live 2026-09-17.

Unlike prices/news, `fundamentals` has NO organic self-heal: the daily
finnhub_fundamentals source only ever writes rows for the CURRENT run's own
`asof_date`, no rolling lookback, so a night that misses it stays missing
forever without a deliberate backfill.

Approach: call FinnhubFundamentalsSource.fetch() directly (the source's own
documented `fetch(tickers, asof_date, store, run_id) -> IngestResult`
contract, per project CLAUDE.md), one call per (date, missing-tickers-only)
so we do not re-spend Finnhub quota on tickers that already have a row for
that date. This bypasses the CLI's deadline-budget guard (a scheduling
safety feature for the live evening pipeline, not a data-integrity rule)
without touching src/.

CAVEAT (surface this, don't bury it): Finnhub's company_basic_financials
endpoint has no historical "as of" parameter -- it always returns the
CURRENT snapshot. Every fundamentals row in this table (backfilled here or
written by a normal nightly run) is therefore "latest known fundamentals,
stamped with the asof_date it was fetched for", not a true point-in-time
historical value. This script does not change that existing limitation; it
only fills the (ticker, date) holes so downstream joins don't have gaps.

Idempotent: the missing (date, ticker) set is recomputed live from the DB
each run (INSERT OR REPLACE on the fundamentals PK), so a second run finds
nothing left to do. Every row this script writes carries the reserved
run_id 9000000000000004 (next unused id after the 2023-H1 gap backfill's
9000000000000003 -- checked `SELECT DISTINCT run_id FROM <table> WHERE
run_id >= 9e15` across every table first). `DELETE FROM fundamentals WHERE
run_id = 9000000000000004` (and `DELETE FROM earnings WHERE run_id =
9000000000000004` for the earnings-calendar rows fetch() also writes as a
side effect) cleanly reverses this script's writes without touching any
organic row.

Usage:
    ./.venv/bin/python scripts/backfill_sept2026_outage_fundamentals.py             # do it
    ./.venv/bin/python scripts/backfill_sept2026_outage_fundamentals.py --dry-run   # report only
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from loguru import logger

from sma.config import load_settings
from sma.db_connect import read_only_connect
from sma.ingest.ratelimit import per_minute_bucket
from sma.ingest.sources.finnhub_fundamentals import FinnhubFundamentalsSource
from sma.ingest.store import Store
from sma.ingest.universe import load_universe
from sma.locks import writer_lock

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "sma.duckdb"
UNIVERSE_PATH = Path(__file__).resolve().parents[1] / "src" / "sma" / "universe.yaml"
CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"

RESERVED_RUN_ID = 9000000000000004  # reserved: Sept-2026 outage fundamentals backfill

# Trading days in the audited outage window (2026-08-28..2026-09-16).
CANDIDATE_DATES = [
    date(2026, 8, 28), date(2026, 8, 31),
    date(2026, 9, 1), date(2026, 9, 2), date(2026, 9, 3), date(2026, 9, 4),
    date(2026, 9, 8), date(2026, 9, 9), date(2026, 9, 10), date(2026, 9, 11),
    date(2026, 9, 14), date(2026, 9, 15), date(2026, 9, 16),
]

# A date only needs backfilling if coverage is meaningfully short of the
# universe (a handful of missing tickers on an otherwise-healthy night is
# normal 429/flake noise, not an outage hole).
COVERAGE_FLOOR = 0.95


def _missing_by_date(con, universe: list[str]) -> dict[date, list[str]]:
    out: dict[date, list[str]] = {}
    for d in CANDIDATE_DATES:
        have = {
            r[0]
            for r in con.execute(
                "SELECT DISTINCT ticker FROM fundamentals WHERE asof_date = ? AND ticker = ANY(?)",
                [d, universe],
            ).fetchall()
        }
        missing = sorted(set(universe) - have)
        if len(have) / len(universe) < COVERAGE_FLOOR:
            out[d] = missing
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    universe = load_universe(UNIVERSE_PATH)

    ro = read_only_connect(DB_PATH)
    try:
        missing = _missing_by_date(ro, universe)
    finally:
        ro.close()

    if not missing:
        print("nothing to backfill; every candidate date already >= "
              f"{COVERAGE_FLOOR:.0%} fundamentals coverage")
        return

    for d, tickers in missing.items():
        print(f"{d}: {len(tickers)} missing tickers")

    if args.dry_run:
        print("--dry-run: no writes")
        return

    settings = load_settings(config_path=CONFIG_PATH)
    s = settings.secrets
    finnhub_rpm = settings.ingest.rate_limits.finnhub.requests_per_minute
    limiter = per_minute_bucket(finnhub_rpm)
    source = FinnhubFundamentalsSource(
        api_key=s.finnhub_api_key,
        rate_limiter=limiter,
        retry_sleep_budget_s=settings.ingest.retry_sleep_budget_s,
    )

    with writer_lock(label="backfill-sept2026-fundamentals", timeout_s=1800.0):
        store = Store(path=str(DB_PATH)).connect()
        try:
            for d, tickers in missing.items():
                logger.info("fundamentals backfill: {} ({} tickers)", d, len(tickers))
                result = source.fetch(
                    tickers=tickers, asof_date=d, store=store, run_id=RESERVED_RUN_ID
                )
                print(f"{d}: fetched {result.rows_inserted} rows "
                      f"(status={result.status})")
        finally:
            store.close()

    ro2 = read_only_connect(DB_PATH)
    try:
        for d in missing:
            n = ro2.execute(
                "SELECT COUNT(DISTINCT ticker) FROM fundamentals WHERE asof_date=? AND ticker=ANY(?)",
                [d, universe],
            ).fetchone()[0]
            print(f"{d}: fundamentals coverage now {n}/{len(universe)}")
    finally:
        ro2.close()


if __name__ == "__main__":
    main()
