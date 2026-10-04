"""ARM A: per-feature regime-split IC, UNCLIPPED vs membership-CLIPPED universe.

Mirrors scripts/per_feature_ic.py `full` but at each asof restricts the
universe to bot-equities that were ACTUAL S&P members as-of that asof
(PIT membership from sp500_pit.csv). Removing look-ahead names (in the bot
universe but not yet in the index) tests whether the reversal-IC of the
momentum features is an artifact of look-ahead injection. NO model.
"""
import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, "/private/tmp/claude-501/-Users-youruser-SecondBrain/f0fae5bb-6213-478f-b88e-e8747330c663/scratchpad")
from membership import _norm, members_asof  # noqa: E402
from shared_pit import make_memcon  # noqa: E402

from sma.eval.regime import classify_regimes  # noqa: E402
from sma.features.builder import FEATURE_NAMES, build_features  # noqa: E402
from sma.ingest.universe import load_universe  # noqa: E402

ETFS = {"SPY", "NANC", "XLB", "XLC", "XLE", "XLF", "XLI", "XLK",
        "XLP", "XLRE", "XLU", "XLV", "XLY"}

# Full-history deduped con (relabels yfinance_hist->yfinance so 2018-2022 is
# NOT silently dropped by the source='yfinance' filter the sma helpers hardcode)
con = make_memcon()
start = date(2018, 6, 1)
end = con.execute("SELECT MAX(date) FROM prices").fetchone()[0]
n_asofs = 48
sessions = [r[0] for r in con.execute(
    "SELECT DISTINCT date FROM prices WHERE date BETWEEN ? AND ? ORDER BY date",
    [start, end - timedelta(days=35)]).fetchall()]
asofs = sessions[::max(1, len(sessions) // n_asofs)][:n_asofs]

universe_all = load_universe("/Users/youruser/code/stock-market-predictor-agents/src/sma/universe.yaml")
# equities only (drop ETFs); keep SPY in prices for features but not scored
equities = [t for t in universe_all if t not in ETFS]
universe = equities  # what classify_regimes / build_features score

prices = con.execute(
    "SELECT * FROM prices WHERE date BETWEEN ? AND ?",
    [start - timedelta(days=420), end]).df()
prices["date"] = pd.to_datetime(prices["date"]).dt.date


def realized_30d(asof):
    fwd = con.execute(
        "SELECT DISTINCT date FROM prices WHERE date > ? ORDER BY date LIMIT 31", [asof]
    ).fetchall()
    if len(fwd) < 31:
        return {}
    e, f = fwd[0][0], fwd[30][0]
    rows = con.execute("""
        SELECT ticker, MAX(CASE WHEN date=? THEN open*adj_close/NULLIF(close,0) END),
               MAX(CASE WHEN date=? THEN adj_close END)
        FROM prices WHERE source='yfinance' AND date IN (?,?) GROUP BY ticker
    """, [e, f, e, f]).fetchall()
    return {t: (fp / ep - 1) for t, ep, fp in rows if ep and fp and ep > 0}


regimes = classify_regimes(con, asofs, universe)

# Two accumulators: unclipped and clipped (membership-restricted)
def fresh():
    return {f: [] for f in FEATURE_NAMES}

acc = {
    ("unclipped", "trend"): fresh(), ("unclipped", "reversal"): fresh(),
    ("clipped", "trend"): fresh(), ("clipped", "reversal"): fresh(),
}
clip_stats = []  # (asof, n_uncl_common, n_clip_common)

for asof in asofs:
    reg = regimes.get(asof)
    if reg is None:
        continue
    feats = build_features(prices, universe, asof)
    if feats.empty:
        continue
    real = realized_30d(asof)
    members = members_asof(asof)

    common_all = [t for t in feats.index if t in real]
    common_clip = [t for t in common_all if _norm(t) in members]
    clip_stats.append((asof, reg[0], len(common_all), len(common_clip)))
    if len(common_all) < 20:
        continue

    for tag, common in (("unclipped", common_all), ("clipped", common_clip)):
        if len(common) < 20:
            continue
        rv = np.array([real[t] for t in common])
        bucket = acc[(tag, reg[0])]
        for f in FEATURE_NAMES:
            fv = feats.loc[common, f].to_numpy(dtype=float)
            if np.std(fv) < 1e-12:
                continue
            ic = spearmanr(fv, rv).correlation
            if not np.isnan(ic):
                bucket[f].append(ic)

nt = sum(1 for a in asofs if regimes.get(a, ("",))[0] == "trend")
nr = sum(1 for a in asofs if regimes.get(a, ("",))[0] == "reversal")
print(f"regime split: {nt} trend asofs, {nr} reversal asofs (of {len(asofs)})")
print("\nclip impact (asof, regime, n_unclipped, n_clipped, n_lookahead_removed):")
for asof, rg, na, nc in clip_stats:
    print(f"  {asof} {rg:8} {na:4d} {nc:4d}  -{na-nc}")

print(f"\n{'feature':26} {'rev_IC_uncl':>11} {'rev_IC_clip':>11} {'delta':>9} "
      f"{'trnd_uncl':>10} {'trnd_clip':>10}")
def m(b, f):
    return np.mean(b[f]) if b[f] else float("nan")
for f in FEATURE_NAMES:
    ru = m(acc[("unclipped", "reversal")], f)
    rc = m(acc[("clipped", "reversal")], f)
    tu = m(acc[("unclipped", "trend")], f)
    tc = m(acc[("clipped", "trend")], f)
    print(f"{f:26} {ru:+11.4f} {rc:+11.4f} {rc-ru:+9.4f} {tu:+10.4f} {tc:+10.4f}")

# Momentum-feature summary (the bet)
mom = ["ret_20d", "ret_60d", "rel_strength_spy_60d", "rel_strength_sector_etf_30d",
       "vol_adj_mom_60d"]
print("\nMOMENTUM features reversal-IC (unclipped -> clipped):")
for f in mom:
    ru = m(acc[("unclipped", "reversal")], f)
    rc = m(acc[("clipped", "reversal")], f)
    print(f"  {f:30} {ru:+.4f} -> {rc:+.4f}  ({rc-ru:+.4f})")
con.close()
