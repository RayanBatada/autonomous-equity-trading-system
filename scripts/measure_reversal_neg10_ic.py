"""Standalone rank-IC scan for the reversal_neg_10d_z candidate feature.

Diagnosis 2026-06-25 #3: `reversal_neg_10d_z` = −ret_10d / vol_20d as a genuine
short-horizon reversion axis (the existing reversal_5d_z is +r5/v20 — a
momentum echo, not reversion). Before paying for a full retrain A/B, measure
the candidate's STANDALONE weekly cross-sectional Spearman IC against the
30-session forward DEMEANED return (the label the live model trains on).
Context columns: the existing reversal_5d_z axis and ret_60d (the dominant
momentum axis) measured identically.

Honest bar: the autoresearch promotion gate is +0.005 CV-IC on the FULL model;
a standalone feature IC materially below the existing features' ICs (or ~0)
means the retrain A/B is not worth the RAM/hours. PIT throughout: features use
data ≤ asof; the forward label obviously looks ahead (it is the target).

Run: .venv/bin/python scripts/measure_reversal_neg10_ic.py
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sma.ingest.universe import load_universe

SCRATCH = Path(
    "/private/tmp/claude-501/-Users-youruser-SecondBrain/"
    "9c290692-7caf-4a33-9129-9d332a7cd4cb/scratchpad/dispersion-ab"
)
DB = SCRATCH / "sma-research.duckdb"
START = date(2022, 6, 1)
END = date(2026, 7, 17)
FWD = 30  # sessions, matches the live label horizon


def main() -> None:
    universe = [t for t in load_universe("src/sma/universe.yaml")
                if not t.startswith("XL") and t not in {"SPY", "NANC"}]
    con = duckdb.connect(str(DB), read_only=True)
    px = con.execute(
        """
        SELECT ticker, date, adj_close FROM (
            SELECT *, ROW_NUMBER() OVER (
                PARTITION BY ticker, date
                ORDER BY CASE source WHEN 'yfinance' THEN 0
                                     WHEN 'yfinance_hist' THEN 1 ELSE 2 END
            ) AS rn
            FROM prices
            WHERE ticker = ANY($t) AND date BETWEEN $s AND $e
              AND adj_close IS NOT NULL
        ) q WHERE rn = 1
        """,
        {"t": universe, "s": START, "e": END},
    ).df()
    con.close()
    px["date"] = pd.to_datetime(px["date"]).dt.date
    piv = px.pivot_table(index="date", columns="ticker", values="adj_close").sort_index()
    rets = piv.pct_change()
    dates = list(piv.index)

    # Weekly asofs (every 5th session) with enough history + forward room.
    asof_idx = [i for i in range(60, len(dates) - FWD, 5)
                if dates[i] >= date(2023, 1, 1)]

    rows = []
    for i in asof_idx:
        r10 = piv.iloc[i] / piv.iloc[i - 10] - 1.0
        r5 = piv.iloc[i] / piv.iloc[i - 5] - 1.0
        r60 = piv.iloc[i] / piv.iloc[i - 60] - 1.0
        vol20 = rets.iloc[i - 19:i + 1].std()
        fwd = piv.iloc[i + FWD] / piv.iloc[i] - 1.0
        df = pd.DataFrame({
            "neg10z": -r10 / vol20.replace(0.0, float("nan")),
            "rev5z": r5 / vol20.replace(0.0, float("nan")),
            "ret60": r60,
            "fwd": fwd,
        }).dropna()
        if len(df) < 60:
            continue
        df["fwd_dm"] = df["fwd"] - df["fwd"].mean()  # demeaned label
        rows.append({
            "asof": dates[i],
            "n": len(df),
            "ic_neg10z": df["neg10z"].rank().corr(df["fwd_dm"].rank()),
            "ic_rev5z": df["rev5z"].rank().corr(df["fwd_dm"].rank()),
            "ic_ret60": df["ret60"].rank().corr(df["fwd_dm"].rank()),
        })

    out = pd.DataFrame(rows)
    print(f"{len(out)} weekly asofs, {out['n'].mean():.0f} names avg\n")
    for col in ("ic_neg10z", "ic_rev5z", "ic_ret60"):
        ics = out[col]
        t = ics.mean() / (ics.std() / len(ics) ** 0.5)
        print(f"{col:11}  mean IC {ics.mean():+.4f}   std {ics.std():.4f}   "
              f"t={t:+.2f}   |IC|>0.05 on {(ics.abs() > 0.05).mean():.0%} of weeks")
    # Year-by-year (regime stability).
    out["year"] = [d.year for d in out["asof"]]
    print("\nper-year mean IC:")
    print(out.groupby("year")[["ic_neg10z", "ic_rev5z", "ic_ret60"]]
          .mean().round(4).to_string())


if __name__ == "__main__":
    main()
