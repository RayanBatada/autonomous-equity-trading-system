"""Verify: the no_dead_or_frozen_tickers quality check (src/sma/ingest/
quality.py) would have caught EA's real frozen-price pattern within 5
sessions of its 2026-08-04 delisting, and is silent today now that EA/AVB
are retired from universe.yaml.

## Why this script exists

See scripts/verify_avb_ea_delisting.py and commit 92474dc for the full
diagnosis: EA was delisted (LBO) 2026-08-04 and AVB merged away (into
Equity Residential) 2026-08-17, yet nothing noticed for weeks because
`all_tickers_have_price` / `enough_sources_succeeded` only ask "did *a* row
land today" -- neither looks at the SHAPE of a ticker's own price history.
Yahoo kept serving EA a FROZEN last-known quote ($209.6999969482422,
bit-identical) for several sessions after the real delisting, so a price
row DID land each day and the existing checks stayed green.

no_dead_or_frozen_tickers closes that gap: FROZEN flags a ticker whose last
`frozen_run` (default 5) deduped closes are bit-identical; STALE flags a
ticker with zero price rows (any source) in the last `stale_sessions`
(default 3) sessions while most of the universe has them. Both are
NON-BLOCKING (a single dead name must not freeze the whole book), paged via
notify_new_dead_or_frozen_tickers's own per-ticker dedup instead.

This script runs the check READ-ONLY (no writer_lock, no run_id) against the
live prod DB at two asof dates:

  1. 2026-08-10 -- the 5th trading session after EA's real 2026-08-04
     delisting (Tue 8/4, Wed 8/5, Thu 8/6, Fri 8/7, Mon 8/10). Confirmed
     against data/sma.duckdb: EA's deduped (yfinance) adj_close is exactly
     209.6999969482422 on all five of those dates -- a real bit-identical
     5-run. EA and AVB are added back into the universe passed to the check
     for this run only, since universe.yaml only removed them THIS MORNING
     (commit 92474dc) -- on 2026-08-10 they were still active names ingest
     was actually fetching. Expect: EA flagged FROZEN. AVB is NOT expected
     to flag here -- its own diagnosis found yfinance back-adjusting its
     close by the merger ratio (distinct values each day, not a frozen
     quote), and its real halt is still a week out at this date.
  2. Today, against universe.yaml as it stands now (EA/AVB already removed
     by commit 92474dc). Expect: silent -- no flags at all.

Usage:
    ./.venv/bin/python scripts/verify_no_dead_or_frozen_tickers.py
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

from sma.db_connect import read_only_connect
from sma.ingest.quality import find_dead_or_frozen_tickers
from sma.ingest.store import Store
from sma.ingest.universe import load_universe

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "sma.duckdb"
UNIVERSE_PATH = Path(__file__).resolve().parents[1] / "src" / "sma" / "universe.yaml"

# 5th trading session after EA's real 2026-08-04 delisting.
EA_CHECK_ASOF = date(2026, 8, 10)


def _print_flags(label: str, flags: list) -> None:
    print(f"--- {label} ---")
    if not flags:
        print("  (silent -- no flags)")
        return
    for f in flags:
        print(
            f"  [{f.kind.upper()}] {f.ticker}: last_price_date={f.last_price_date} "
            f"run_length={f.run_length}"
        )


def main() -> int:
    con = read_only_connect(DB_PATH)
    store = Store(DB_PATH)
    store.conn = con  # reuse the retry-tolerant read-only connection

    try:
        current_universe = load_universe(UNIVERSE_PATH)

        # Run 1: 2026-08-10, EA + AVB added back (as they were actually
        # active in the universe on that date, before commit 92474dc removed
        # them this morning).
        historical_universe = sorted(set(current_universe) | {"EA", "AVB"})
        historical_flags = find_dead_or_frozen_tickers(store, EA_CHECK_ASOF, historical_universe)
        _print_flags(f"asof={EA_CHECK_ASOF} (EA/AVB added back to universe)", historical_flags)
        ea_flagged = any(f.ticker == "EA" and f.kind == "frozen" for f in historical_flags)
        print(f"EA flagged FROZEN by {EA_CHECK_ASOF}: {ea_flagged}")

        # Run 2: today, current (EA/AVB-free) universe.
        today = date.today()
        today_flags = find_dead_or_frozen_tickers(store, today, current_universe)
        _print_flags(f"asof={today} (current universe, EA/AVB retired)", today_flags)
        silent_today = len(today_flags) == 0
        print(f"Silent today ({today}): {silent_today}")
    finally:
        con.close()

    ok = ea_flagged and silent_today
    if ok:
        print(
            "\nAll checks passed: no_dead_or_frozen_tickers would have caught EA within "
            "5 sessions of its 8/4 delisting, and is silent today now that EA/AVB are "
            "retired."
        )
    else:
        print("\nUnexpected result -- re-diagnose before trusting this check's coverage.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
