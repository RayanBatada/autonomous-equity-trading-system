"""Per-feature IC across the val regime split — the map for feature work.

For each feature, rank-IC vs realized 30d-forward returns, split into TREND
vs REVERSAL dates by the DATA-DRIVEN regime classifier (sma.eval.regime:
trend = the momentum factor made money that window, reversal = it inverted) —
NOT a calendar midpoint, which mislabeled trending months as reversals
(2026-06-25 diagnosis). A
feature that's +IC in BOTH halves is a regime-robust diversifier; one that
flips sign with momentum is just another momentum echo. This tells us what
to add/drop to de-concentrate the momentum bet (2026-06-13 IC analysis:
model IC inverts in reversal regimes — it's an undiversified momentum bet).
"""
import sys
from datetime import date, timedelta

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from sma.backtest.windows import window_dates
from sma.eval.regime import classify_regimes
from sma.features.builder import FEATURE_NAMES, build_features
from sma.ingest.universe import load_universe

con = duckdb.connect("data/sma.duckdb", read_only=True)
# `full` spans all history (2018+) so the regime split has MANY reversal dates
# (2018 Q4, 2020 COVID, 2022 bear, 2025 Feb/Apr/May). The default `val` window
# is structurally unusable for regime work — 2025-H2 was ~all trend (the
# data-driven classifier finds ~1 reversal asof there), so its "reversal" IC is
# a single-date coin flip and the old calendar split faked the reversal half.
if len(sys.argv) > 1 and sys.argv[1] == "full":
    start = date(2018, 6, 1)
    end = con.execute("SELECT MAX(date) FROM prices").fetchone()[0]
    n_asofs = 48
else:
    start, end = window_dates("val")
    n_asofs = 12
sessions = [r[0] for r in con.execute(
    "SELECT DISTINCT date FROM prices WHERE date BETWEEN ? AND ? ORDER BY date",
    [start, end - timedelta(days=35)]).fetchall()]
asofs = sessions[::max(1, len(sessions) // n_asofs)][:n_asofs]
universe = load_universe("src/sma/universe.yaml")

# Load all inputs build_features needs, once.
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
    return {t: (fp/ep - 1) for t, ep, fp in rows if ep and fp and ep > 0}

# accumulate per-feature IC per date, tagged by the DATA-DRIVEN regime
regimes = classify_regimes(con, asofs, universe)
trend_ic = {f: [] for f in FEATURE_NAMES}
rev_ic = {f: [] for f in FEATURE_NAMES}
for asof in asofs:
    reg = regimes.get(asof)
    if reg is None:
        continue
    feats = build_features(prices, universe, asof)
    if feats.empty:
        continue
    real = realized_30d(asof)
    common = [t for t in feats.index if t in real]
    if len(common) < 20:
        continue
    rv = np.array([real[t] for t in common])
    bucket = trend_ic if reg[0] == "trend" else rev_ic
    for f in FEATURE_NAMES:
        fv = feats.loc[common, f].to_numpy(dtype=float)
        if np.std(fv) < 1e-12:
            continue
        ic = spearmanr(fv, rv).correlation
        if not np.isnan(ic):
            bucket[f].append(ic)

_nt = sum(1 for a in asofs if regimes.get(a, ("",))[0] == "trend")
_nr = sum(1 for a in asofs if regimes.get(a, ("",))[0] == "reversal")
print(f"{'feature':28} {'trend_IC':>9} {'rev_IC':>9}  robust?")
print(f"  (data-driven regime split: {_nt} trend asofs vs {_nr} reversal asofs)")
for f in FEATURE_NAMES:
    t = np.mean(trend_ic[f]) if trend_ic[f] else float('nan')
    r = np.mean(rev_ic[f]) if rev_ic[f] else float('nan')
    robust = "ROBUST +" if (t > 0.01 and r > 0.01) else \
             "flips" if (np.sign(t) != np.sign(r)) else \
             "weak"
    print(f"{f:28} {t:+9.4f} {r:+9.4f}  {robust}")
con.close()
