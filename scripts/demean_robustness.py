"""Is the demean-label advantage robust or one-date luck? Per-date IC of raw
vs demean on the val reversal half (2026-06-13). If demean beats raw on MOST
reversal dates, the ship is solid; if the edge is one outlier, it's fragile.
Uses cached sets (/tmp/sma-exp), trains 2 quick models, no rebuild."""
from pathlib import Path

import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr

from sma.model.trainer import DEFAULT_HYPERPARAMS

C = Path("/tmp/sma-exp")
Xtr = pd.read_parquet(C/"train_x.parquet"); ytr = pd.read_parquet(C/"train_y.parquet")["y"]
atr = pd.read_parquet(C/"train_a.parquet")["a"]
Xv = pd.read_parquet(C/"val_x.parquet"); yv = pd.read_parquet(C/"val_y.parquet")["y"]
av = pd.read_parquet(C/"val_a.parquet")["a"]

raw = xgb.XGBRegressor(**DEFAULT_HYPERPARAMS); raw.fit(Xtr, ytr)
dm = xgb.XGBRegressor(**DEFAULT_HYPERPARAMS)
dm.fit(Xtr, ytr - ytr.groupby(atr.to_numpy()).transform("mean"))

df = pd.DataFrame({"a": av.to_numpy(), "raw": raw.predict(Xv),
                   "dm": dm.predict(Xv), "y": yv.to_numpy()})
dates = sorted(df["a"].unique()); split = dates[len(dates)//2]
print(f"{'date':12} {'raw_IC':>8} {'dm_IC':>8} {'dm wins?':>9}")
wins = 0; n = 0
for d in dates:
    if d < split: continue  # reversal half only
    g = df[df["a"] == d]
    if len(g) < 20: continue
    ric = spearmanr(g["raw"], g["y"]).correlation
    dic = spearmanr(g["dm"], g["y"]).correlation
    w = dic > ric; wins += w; n += 1
    print(f"{str(d):12} {ric:+8.4f} {dic:+8.4f} {'  yes' if w else '   no':>9}")
print(f"\ndemean beats raw on {wins}/{n} reversal dates "
      f"({'ROBUST' if wins > n*0.6 else 'FRAGILE — one-date driven' if wins < n*0.45 else 'mixed'})")
