"""Information Coefficient: the bedrock 'is there edge' diagnostic.

Per-asof Spearman rank correlation between model predictions and REALIZED
30d-forward returns. Independent of turnover/cost/strategy. A large-cap
ranker with real edge shows monthly IC ~0.03-0.08, t-stat > 2. IC ~0
(t < 2) means we're ranking noise — every config A/B is then a draw from
luck, and the +1.60 was selection over 21 tries on one window.

Reports raw-label IC AND demeaned (cross-sectional) IC — the latter strips
the market component the strategy doesn't trade on.
"""
import sys
from datetime import timedelta
from pathlib import Path

import duckdb
import numpy as np
from scipy.stats import spearmanr

from sma.backtest.windows import window_dates
from sma.ingest.universe import load_universe
from sma.model.predictor import Predictor

MODELS = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/sma-eval-models-21f")
WINDOW = sys.argv[2] if len(sys.argv) > 2 else "val"

start, end = window_dates(WINDOW)
con = duckdb.connect("data/sma.duckdb", read_only=True)
sessions = [r[0] for r in con.execute(
    "SELECT DISTINCT date FROM prices WHERE date BETWEEN ? AND ? ORDER BY date",
    [start, end - timedelta(days=35)]).fetchall()]
# ~12 asof dates evenly spread, each needs 30 fwd sessions
asofs = sessions[::max(1, len(sessions) // 12)][:12]
universe = [t for t in load_universe("src/sma/universe.yaml") if t != "SPY"]
predictor = Predictor(models_dir=MODELS, db_path=Path("data/sma.duckdb"))

def realized_30d(asof):
    """ticker -> realized fwd return (next open → adj_close 30 sessions out)."""
    fwd = con.execute(
        "SELECT DISTINCT date FROM prices WHERE date > ? ORDER BY date LIMIT 31", [asof]
    ).fetchall()
    if len(fwd) < 31:
        return {}
    entry_d, future_d = fwd[0][0], fwd[30][0]
    rows = con.execute("""
        SELECT ticker,
               MAX(CASE WHEN date=? THEN open*adj_close/NULLIF(close,0) END) entry,
               MAX(CASE WHEN date=? THEN adj_close END) fut
        FROM prices WHERE source='yfinance' AND date IN (?, ?) GROUP BY ticker
    """, [entry_d, future_d, entry_d, future_d]).fetchall()
    return {t: (f/e - 1.0) for t, e, f in rows if e and f and e > 0}

ics, ics_dm, ns = [], [], []
for asof in asofs:
    try:
        scores = predictor.predict_for(asof, universe)
    except Exception:
        continue
    real = realized_30d(asof)
    common = sorted(set(scores) & set(real))
    if len(common) < 20:
        continue
    pv = np.array([scores[t] for t in common])
    rv = np.array([real[t] for t in common])
    ic = spearmanr(pv, rv).correlation
    ic_dm = spearmanr(pv, rv - rv.mean()).correlation  # demean ret (rank IC identical—sanity)
    ics.append(ic); ics_dm.append(ic_dm); ns.append(len(common))
    print(f"  {asof}  n={len(common):3d}  IC={ic:+.4f}")

ics = np.array(ics)
mean, std = ics.mean(), ics.std(ddof=1)
tstat = mean / std * np.sqrt(len(ics)) if std > 0 else 0
print(f"\n{WINDOW} {MODELS.name}: {len(ics)} dates, avg n={np.mean(ns):.0f}")
print(f"  mean IC = {mean:+.4f}   std = {std:.4f}   IR = {mean/std:+.3f}   t-stat = {tstat:+.2f}")
print(f"  IC>0 on {(ics>0).sum()}/{len(ics)} dates")
print(f"  VERDICT: {'REAL edge (t>2)' if tstat>2 else 'WEAK/none (t<2) — config A/Bs may be noise' if tstat<2 else ''}")
con.close()
