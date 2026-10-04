#!/usr/bin/env python
"""Generate src/sma/universe_history.yaml from PIT S&P 500 membership snapshots.

WHY: `src/sma/universe.yaml` is an April-2026 selection whose `added` dates are
operational file dates, not economic membership. Training therefore treated
every incumbent as if it had existed since 2018 — pure survivorship. The
breadth study (2026-08-17) measured that at +0.76pp of 30d top-15 excess
return overall (t +1.86, weekly, 301 dates) and +2.73pp on 2023+ (t +3.25).

This script emits the point-in-time membership map that
`sma.ingest.universe.load_training_membership` reads and
`sma.model.loader.build_training_set(membership=...)` applies, so a ticker
contributes training rows only from the date it became a plausible universe
member.

SOURCE: ~/.sma-pit/breadth-study-2026-08-17/data/sp500_snapshots.json —
104 monthly S&P 500 constituent lists (2017-12-31 .. 2026-07-31) parsed from
contemporaneous Wikipedia revisions, 697 distinct symbols. See that study's
scripts/01_scrape_pit_membership.py for provenance and PREREGISTRATION.md for
its known limits (monthly grid, Wikipedia lag of hours-to-days).

CONVENTIONS
  added   = the FIRST snapshot the symbol appears in. The real addition falls
            between the previous snapshot and this one, so this is the
            conservative edge: it never claims membership the name did not
            have. `null` = present in the first snapshot (member since before
            the data starts).
  removed = always null here. Every ticker in this file is in the universe
            TODAY, so none has left OUR universe — `removed` is reserved for
            the `former_tickers:` section (delisted / dropped names). A name
            that left the S&P but that we still trade (ENPH) keeps null.

Re-run:  ./.venv/bin/python scripts/gen_universe_history.py
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
UNIVERSE = REPO / "src" / "sma" / "universe.yaml"
OUT = REPO / "src" / "sma" / "universe_history.yaml"
SNAPSHOTS = (
    Path.home() / ".sma-pit" / "breadth-study-2026-08-17"
    / "data" / "sp500_snapshots.json"
)

# Training-history start (mirrors model.__main__.TRAIN_DEFAULT_START). Used as
# the documented fallback for tickers the S&P snapshots cannot date.
PROJECT_START = date(2018, 1, 1)

# Symbol changes where the INDEX SEAT was continuous: the company stayed in the
# S&P 500 and only its ticker changed, so the naive first-seen date would
# truncate real membership. Each maps successor -> predecessor symbol, verified
# against the snapshot in which the predecessor left and the successor arrived
# in the same month. NOT included: DD (DowDuPont -> DuPont, 2019-06) — the DD
# ticker was recycled from the pre-2017 DuPont and its price series is spliced,
# so it keeps the conservative 2019 date.
SYMBOL_CHANGES: dict[str, tuple[str, str]] = {
    "BKNG": ("PCLN", "Priceline -> Booking Holdings, 2018-02"),
    "WELL": ("HCN", "Health Care REIT -> Welltower, 2018-02"),
    "LIN": ("PX", "Praxair -> Linde plc, 2018-11"),
    "BKR": ("BHGE", "Baker Hughes a GE co -> Baker Hughes, 2019-10"),
    "TFC": ("BBT", "BB&T -> Truist Financial, 2019-12"),
    "RTX": ("UTX", "United Technologies -> Raytheon Technologies, 2020-04"),
    "WBD": ("DISCA", "Discovery -> Warner Bros. Discovery, 2022-04"),
    "META": ("FB", "Facebook -> Meta Platforms, 2022-06"),
    "ELV": ("ANTM", "Anthem -> Elevance Health, 2022-06"),
}

# Names we trade that the S&P snapshots cannot date, with why. Non-index
# equities and the benchmark/sector ETFs. They fall back to PROJECT_START,
# which is a no-op at the default training start (2018-01-01) and only bites
# if SMA_TRAIN_START is pushed earlier — where "we cannot date it" is the
# honest answer anyway.
UNDATABLE_REASON = "not an S&P 500 member in any snapshot"


def load_snapshots() -> tuple[list[date], dict[date, set[str]]]:
    raw = json.loads(SNAPSHOTS.read_text())
    snaps = {date.fromisoformat(d): set(v) for d, v in raw.items()}
    return sorted(snaps), snaps


def main() -> None:
    if not SNAPSHOTS.exists():
        raise SystemExit(f"missing PIT snapshots: {SNAPSHOTS}")
    snap_dates, snaps = load_snapshots()
    first_snap = snap_dates[0]

    first_seen: dict[str, date] = {}
    for d in snap_dates:
        for sym in snaps[d]:
            first_seen.setdefault(sym, d)

    universe = sorted(
        t.upper()
        for t in yaml.safe_load(UNIVERSE.read_text())["universe"]["tickers"]
    )

    lines: list[str] = []
    n_null = n_dated = n_alias = n_fallback = 0
    for t in universe:
        seen = first_seen.get(t)
        note = ""
        if t in SYMBOL_CHANGES:
            pred, why = SYMBOL_CHANGES[t]
            pred_seen = first_seen.get(pred)
            if pred_seen is not None and (seen is None or pred_seen < seen):
                seen, note = pred_seen, f"via {pred} ({why})"
                n_alias += 1
        if seen is None:
            added_txt = PROJECT_START.isoformat()
            note = note or UNDATABLE_REASON
            note += "; fallback = project training start"
            n_fallback += 1
        elif seen == first_snap:
            added_txt = "null"
            note = note or f"member at the first snapshot ({first_snap})"
            n_null += 1
        else:
            added_txt = seen.isoformat()
            n_dated += 1
        suffix = f"  # {note}" if note else ""
        # Every key is quoted: YAML 1.1 parses bare ON / NO / YES / OFF as
        # booleans, and ON (ON Semiconductor) is in the universe. universe.yaml
        # already carries the same scar tissue on that one line.
        lines.append(f'  "{t}": {{added: {added_txt}}}{suffix}')

    header = f"""\
# Point-in-time universe membership — the survivorship fix for TRAINING.
#
# GENERATED by scripts/gen_universe_history.py. Do not hand-edit; re-run it.
# Source: {len(snap_dates)} monthly S&P 500 constituent snapshots
# ({first_snap} .. {snap_dates[-1]}) from the breadth study 2026-08-17.
#
# Read by sma.ingest.universe.load_training_membership() and applied by
# sma.model.loader.build_training_set(membership=...): a listed ticker
# contributes training rows only while asof is in [added, removed), with a
# null bound unbounded. Without this file the loader gets {{}} and training
# treats today's universe as having always existed — which the breadth study
# measured at +0.76pp of 30d top-15 excess overall, +2.73pp on 2023+.
#
# `added` is the first snapshot the name appears in, so the real addition sits
# somewhere in the preceding month: the conservative edge, never claiming
# membership the name did not have. `removed` is omitted (null) for every
# entry here — these are all CURRENT universe members, so none has left OUR
# universe. ENPH is the one to know about: it left the S&P 500 in 2025-08 but
# we still trade it, so it keeps its 2021 `added` and no `removed`.
#
# Counts: {len(universe)} tickers — {n_null} members at the first snapshot,
# {n_dated} with a dated addition, {n_alias} resolved through a symbol change,
# {n_fallback} not datable from the snapshots (fallback {PROJECT_START}).
#
# KNOWN ASYMMETRY, stated plainly: S&P membership is a PROXY for "our process
# could have picked this name at date D", and it is a lossy one at both edges.
# A name that only recently joined the index (MRVL 2026-06, LITE 2026-03,
# HOOD 2025-09) loses nearly all of its training history even though it was a
# liquid large cap the whole time — while the 14 names that were NEVER index
# members (AFRM ARM RBLX RIVN SNOW SOFI ...) are the most hindsight-selected of
# all and get no restriction at all, because the snapshots cannot date them.
# The restriction is conservative where it applies and absent where it cannot.
# That is the same rule the breadth study's A2PIT arm measured, so the +0.76pp
# / +2.73pp numbers are the numbers for THIS rule, not for an idealized one.
#
# `former_tickers:` (names that left the universe entirely, with a real
# `removed` date) is read from the same file and is intentionally empty for
# now — populating it with the 195 churned-out S&P names is the next step.

members:
"""
    OUT.write_text(header + "\n".join(lines) + "\n\nformer_tickers: {}\n")
    print(f"wrote {OUT} — {len(universe)} tickers")
    print(
        f"  null-added (long-standing): {n_null}\n"
        f"  dated additions:            {n_dated}\n"
        f"  symbol-change aliases:      {n_alias}\n"
        f"  undatable fallbacks:        {n_fallback}"
    )


if __name__ == "__main__":
    main()
