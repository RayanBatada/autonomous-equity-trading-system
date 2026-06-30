"""Is multi-regime's val-IC win robust per-date, or one-date luck? (2026-06-16)
Before overriding the RMSE gate to deploy it, hold the same standard used for
demean: per-date IC, multi-regime vs 2023-only, on the val reversal half."""
from datetime import timedelta
from pathlib import Path

import duckdb
import numpy as np
from scipy.stats import spearmanr

from sma.backtest.windows import window_dates
from sma.ingest.universe import load_universe
from sma.model.predictor import Predictor

MR = Path("/tmp/sma-eval-models-multiregime")
BASE = Path("/tmp/sma-eval-models-21f")  # 2023-only
start, end = window_dates("val")
con = duckdb.connect("data/sma.duckdb", read_only=True)
sessions = [r[0] for r in con.execute(
    "SELECT DISTINCT date FROM prices WHERE date BETWEEN ? AND ? ORDER BY date",
    [start, end - timedelta(days=35)]).fetchall()]
asofs = sessions[::max(1, len(sessions)//12)][:12]
split = asofs[len(asofs)//2]
universe = [t for t in load_universe("src/sma/universe.yaml") if t != "SPY"]
pmr = Predictor(models_dir=MR, db_path=Path("data/sma.duckdb"))
pbase = Predictor(models_dir=BASE, db_path=Path("data/sma.duckdb"))

def realized(asof):
    fwd = con.execute("SELECT DISTINCT date FROM prices WHERE date > ? ORDER BY date LIMIT 31", [asof]).fetchall()
    if len(fwd) < 31: return {}
    e, f = fwd[0][0], fwd[30][0]
    rows = con.execute("SELECT ticker, MAX(CASE WHEN date=? THEN open*adj_close/NULLIF(close,0) END), MAX(CASE WHEN date=? THEN adj_close END) FROM prices WHERE source='yfinance' AND date IN (?,?) GROUP BY ticker", [e,f,e,f]).fetchall()
    return {t:(fp/ep-1) for t,ep,fp in rows if ep and fp and ep>0}

print(f"{'date':12} {'base_IC':>8} {'mr_IC':>8} {'mr wins':>8}")
wins=tot=0
for asof in asofs:
    if asof < split: continue  # reversal half
    real = realized(asof)
    smr = pmr.predict_for(asof, universe); sb = pbase.predict_for(asof, universe)
    common = sorted(set(smr)&set(sb)&set(real))
    if len(common)<20: continue
    rv = np.array([real[t] for t in common])
    icmr = spearmanr([smr[t] for t in common], rv).correlation
    icb = spearmanr([sb[t] for t in common], rv).correlation
    w = icmr>icb; wins+=w; tot+=1
    print(f"{str(asof):12} {icb:+8.4f} {icmr:+8.4f} {'  yes' if w else '   no':>8}")
print(f"\nmulti-regime beats 2023-only on {wins}/{tot} reversal dates "
      f"({'ROBUST -> deploy' if wins>tot*0.6 else 'FRAGILE -> keep gate decision'})")
con.close()
