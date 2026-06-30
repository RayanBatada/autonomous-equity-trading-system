"""Combine the two winners: rank objective ON demeaned labels, + variants to
tame the rank objective's reversal amplification (2026-06-13). Cached sets."""
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr

from sma.model.trainer import DEFAULT_HYPERPARAMS

C = Path("/tmp/sma-exp")
Xtr = pd.read_parquet(C/"train_x.parquet"); ytr = pd.read_parquet(C/"train_y.parquet")["y"]
atr = pd.read_parquet(C/"train_a.parquet")["a"]
Xv = pd.read_parquet(C/"val_x.parquet"); yv = pd.read_parquet(C/"val_y.parquet")["y"]
av = pd.read_parquet(C/"val_a.parquet")["a"]

def regime_ic(pred):
    df = pd.DataFrame({"a": av.to_numpy(), "p": pred, "y": yv.to_numpy()})
    dates = sorted(df["a"].unique()); split = dates[len(dates)//2]
    tr, rv = [], []
    for d, g in df.groupby("a"):
        if len(g) < 20 or g["p"].std() < 1e-12: continue
        ic = spearmanr(g["p"], g["y"]).correlation
        (tr if d < split else rv).append(ic)
    return np.nanmean(tr), np.nanmean(rv)

def show(name, pred):
    t, r = regime_ic(pred)
    avg = (t + r) / 2
    flag = "ROBUST both+" if t > 0 and r > 0 else f"avg={avg:+.4f}"
    print(f"{name:30} trend={t:+.4f}  rev={r:+.4f}  {flag}")

ytr_dm = (ytr - ytr.groupby(atr.to_numpy()).transform("mean"))
order = np.argsort(atr.to_numpy(), kind="stable")
Xs = Xtr.iloc[order]; as_ = atr.iloc[order]
ys_raw = ytr.iloc[order]; ys_dm = ytr_dm.iloc[order]
groups = as_.value_counts(sort=False).reindex(pd.Index(as_.unique())).tolist()

def rank_fit(y, **over):
    m = xgb.XGBRanker(**{**DEFAULT_HYPERPARAMS, "objective": "rank:pairwise", **over})
    m.fit(Xs, y, group=groups)
    return m.predict(Xv)

show("rank raw", rank_fit(ys_raw))
show("rank demean", rank_fit(ys_dm))
show("rank demean depth3", rank_fit(ys_dm, max_depth=3))
show("rank demean depth3 reg", rank_fit(ys_dm, max_depth=3, reg_lambda=5.0, subsample=0.7))
show("rank raw ndcg", rank_fit(ys_raw, objective="rank:ndcg"))
