"""Verify: AVB and EA's price-day gap vs AAPL is fully explained by real
delistings, not an ingest bug -- and that neither is still active in
universe.yaml.

## Why this script exists

2026-08-31 measurement: since 2026-01-01, AVB was missing 4 trading days and
EA was missing 14, vs AAPL's 165, in the deduped prices table (the
ROW_NUMBER/CASE priority query src/sma/model/__main__.py and
src/sma/model/predictor.py both use: PARTITION BY ticker,date ORDER BY
source priority yfinance=0, alpaca=1, filtered to adj_close IS NOT NULL --
alpaca rows always have adj_close NULL, so in practice only a 'yfinance' row
satisfies the filter). That read like a classic ingest gap, the kind
scripts/backfill_2023h1_gap.py fixes by re-fetching and inserting rows.

Diagnosis (per-source query + a live yfinance re-fetch + an external check
against SEC filings/press coverage as the "second source" the task asked
for before force-filling) found something different: both tickers stopped
trading for real, permanent reasons, and the "missing" days are entirely
AFTER each one's actual last trading day --

- EA (Electronic Arts): taken private in a $55B LBO (Saudi PIF 93.4% /
  Silver Lake 5.5% / Affinity Partners 1.1%). Stock stopped trading and was
  delisted from Nasdaq 2026-08-04; holders cashed out at $210/share. Sources:
  https://variety.com/2026/gaming/news/electronic-arts-close-55-billion-private-deal-1236824625/
  https://www.thewrap.com/industry-news/deals-ma/electronic-arts-goes-private-55-billion-buyout-closes/
- AVB (AvalonBay Communities): merger of equals with Equity Residential
  (~$69B combined EV), forming "Vivmark Residential". NYSE halted AVB before
  the open 2026-08-17 and delisted it; each AVB share converted to 2.793 EQR
  shares. Source:
  https://investors.avalonbay.com/news-events/press-releases/detail/445/equity-residential-and-avalonbay-communities-announce-shareholder-approvals-for-merger-to-create-vivmark-residential
  That 2.793 ratio is also the exact explanation for a second thing found
  during diagnosis: yfinance's AVB close for dates BEFORE the halt (rows
  with source='yfinance', still present in `prices` through 2026-08-24)
  reads ~35.8% (1/2.793) of alpaca's raw close for the same date. That is
  yfinance back-adjusting AVB's historical close by the merger exchange
  ratio -- the exact same mechanism (and the exact same "legitimate,
  don't-quarantine-it" case) as a stock split, per the CRWD 4:1 split
  comment in yfinance_prices.py's _quarantine_cross_source_outliers. It is
  NOT corrupt data. Do not "fix" it by rescaling or deleting those rows.

Neither ticker ever appears in `intended_orders` or `paper_fills` (checked
2026-08-31) -- the bot never held or attempted a position in either, so
there is no open-position cleanup needed.

## What this means for the "backfill"

There is nothing to backfill. Force-filling either ticker's post-delisting
dates with a price would be inventing trades for a security that was not
trading -- exactly the failure mode the task brief's own "check EA/AVB
weren't halted those dates via a second source before force-filling"
instruction exists to prevent. This script only VERIFIES the diagnosis
(read-only, no writer_lock needed, no run_id) and captures it in one place
so a future re-run confirms the gap is still fully explained by the
delisting and hasn't grown for some new, actually-buggy reason.

## What WAS done instead (see git log for this commit)

`src/sma/universe.yaml`: removed AVB and EA from the active `tickers:` map
(with the same facts recorded inline) so daily ingest stops chasing two
dead tickers and stops generating "possibly delisted" log noise. Left
`src/sma/universe_history.yaml` (the point-in-time training-membership file)
untouched -- it is a GENERATED file ("do not hand-edit; re-run it" via
scripts/gen_universe_history.py) that would need a fresh S&P 500 snapshot
through August/September 2026 to correctly date AVB/EA's removal; that is a
separate, larger job than this cleanup and doesn't cause bad training rows
either way since there is no post-delisting price data for the loader to
pick up regardless of what this file says.

## Update (2026-09-01): universe_history.yaml now carries the removed dates

The "doesn't cause bad training rows either way" claim above undersold the
risk: it's only true for the CURRENT retrain/autoresearch entry points,
which both build their training universe from `load_universe(universe.yaml)`
-- already AVB/EA-free since this commit. It is NOT true for any caller that
reconstructs a broader historical universe from universe_history.yaml's own
membership keys (past + present), and it left the file's own documentation
false ("removed is omitted for every entry -- these are all CURRENT
universe members"). `src/sma/universe_history.yaml`'s `members:` entries for
AVB and EA now carry a real `removed` date -- the day after each one's true
last trading day, pinned above and checked below -- so the file is honestly
PIT-correct: present while alive, absent after death, per
load_training_membership()'s documented [added, removed) convention. This
was a hand-edit despite the "GENERATED, do not hand-edit" header (re-running
scripts/gen_universe_history.py would need a fresh S&P 500 snapshot through
Aug/Sep 2026, and would currently just DROP both tickers from `members:`
outright, since it rebuilds that section from today's universe.yaml -- see
that script's own docstring). Empirically: with the production default
forward_horizon_days=30 and today's asof, this changes ZERO training rows
(the pre-existing forward-label-availability check already excludes both
tickers' post-delisting asofs on its own -- see build_training_set's
docstring); at forward_horizon_days=5 it removes exactly one phantom row
(AVB, asof 2026-08-17). The fix is defense-in-depth and file-correctness,
not a retrain-changing bugfix today.

Usage:
    ./.venv/bin/python scripts/verify_avb_ea_delisting.py
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

from sma.db_connect import read_only_connect
from sma.ingest.universe import load_training_membership, load_universe

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "sma.duckdb"
UNIVERSE_PATH = Path(__file__).resolve().parents[1] / "src" / "sma" / "universe.yaml"
HISTORY_PATH = Path(__file__).resolve().parents[1] / "src" / "sma" / "universe_history.yaml"

MEASUREMENT_START = "2026-01-01"

# (ticker, real last trading day, one-line reason)
DELISTED = [
    ("EA", date(2026, 8, 4), "taken private ($55B LBO), delisted from Nasdaq"),
    ("AVB", date(2026, 8, 14), "merged into Vivmark Residential, NYSE halt 2026-08-17"),
]


def _dedup_count(con, ticker: str, end: date) -> int:
    row = con.execute(
        """
        SELECT COUNT(*) FROM (
            SELECT *, ROW_NUMBER() OVER (
                PARTITION BY ticker, date
                ORDER BY CASE source WHEN 'yfinance' THEN 0
                                      WHEN 'alpaca' THEN 1
                                      ELSE 2 END
            ) AS rn
            FROM prices
            WHERE ticker = ? AND date >= ? AND date <= ? AND adj_close IS NOT NULL
        ) t WHERE rn = 1
        """,
        [ticker, MEASUREMENT_START, end.isoformat()],
    ).fetchone()
    return row[0]


def main() -> int:
    con = read_only_connect(DB_PATH)
    ok = True
    try:
        for ticker, last_day, reason in DELISTED:
            aapl_n = _dedup_count(con, "AAPL", last_day)
            ticker_n = _dedup_count(con, ticker, last_day)
            status = "OK" if ticker_n == aapl_n else "MISMATCH"
            if ticker_n != aapl_n:
                ok = False
            print(
                f"{ticker}: {ticker_n}/{aapl_n} trading days match AAPL through "
                f"its real last trading day {last_day} ({reason}) -- {status}"
            )
    finally:
        con.close()

    universe = load_universe(UNIVERSE_PATH)
    for ticker, _, _ in DELISTED:
        if ticker in universe:
            print(f"WARNING: {ticker} is still listed in {UNIVERSE_PATH} — remove it.")
            ok = False
        else:
            print(f"{ticker}: confirmed absent from the active universe ({UNIVERSE_PATH}).")

    membership = load_training_membership(HISTORY_PATH)
    for ticker, last_day, _reason in DELISTED:
        expected_removed = last_day + timedelta(days=1)
        removed = membership.get(ticker, (None, None))[1]
        if removed == expected_removed:
            print(
                f"{ticker}: {HISTORY_PATH.name} removed={removed} "
                f"(day after real last trade {last_day}) -- OK."
            )
        else:
            print(
                f"WARNING: {ticker} {HISTORY_PATH.name} removed={removed}, "
                f"expected {expected_removed} (day after real last trade "
                f"{last_day}) -- post-delisting rows are not PIT-excluded "
                "from training."
            )
            ok = False

    if ok:
        print(
            "\nAll checks passed: the AVB/EA day-count gap is fully explained by their "
            "real delistings. Nothing to backfill."
        )
    else:
        print(
            "\nSomething changed since this script was written — re-diagnose before "
            "assuming the delisting explanation still covers the full gap."
        )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
