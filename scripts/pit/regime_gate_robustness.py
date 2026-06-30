"""Robustness / honesty checks for the gate.

1. Per-held-out-year OOS AUC and ON-OFF spread for each signal -> is any edge
   present in MOST years or driven by one (e.g. 2022)?
2. Significance of the ON-minus-OFF forward-factor-return spread accounting for
   the forward-30d window overlap (monthly asofs overlap ~50%): block bootstrap.
3. Is the gate's gain just a couple of lucky non-independent months?
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path.home()/".sma-pit"
panel = pd.read_parquet(OUT/"regime_panel.parquet")
panel["year"]=panel["date"].dt.year
SIGNALS=["fac_21","fac_63","dispersion","breadth50","breadth200","spy_vol","spy_above_ma200","spy_ma200_dist"]

def auc(score,label):
    score=np.asarray(score,float); label=np.asarray(label)
    m=np.isfinite(score)&np.isfinite(label); score,label=score[m],label[m]
    pos=(label==1).sum(); neg=(label==0).sum()
    if pos==0 or neg==0: return np.nan
    r=pd.Series(score).rank().values
    return (r[label==1].sum()-pos*(pos+1)/2)/(pos*neg)

# ---- Per-year OOS (expanding) AUC matrix, monthly ----
sub=panel[panel.freq=="M"].copy()
years=sorted(sub.year.unique())
print("PER-YEAR OOS AUC (expanding train), monthly. Want CONSISTENTLY >0.5.\n")
mat={}
for sig in SIGNALS:
    row={}
    for ty in years:
        train=sub[sub.year<ty].dropna(subset=[sig])
        test=sub[sub.year==ty].dropna(subset=[sig])
        if len(train)<24 or len(test)<4: continue
        dirn=np.sign(np.corrcoef(train[sig],train["target_ret"])[0,1]) or 1.0
        row[ty]=round(auc(dirn*test[sig],test["trend"]),2)
    mat[sig]=row
mdf=pd.DataFrame(mat).T
mdf["n_yrs>0.5"]=(mdf>0.5).sum(axis=1)
mdf["mean"]=mdf[[c for c in mdf.columns if isinstance(c,(int,np.integer))]].mean(axis=1).round(3)
print(mdf.to_string())

# ---- Block bootstrap of ON-OFF spread (best signals), monthly expanding ----
def oos_on_off(sig):
    rows=[]
    for ty in years:
        train=sub[sub.year<ty].dropna(subset=[sig]); test=sub[sub.year==ty].dropna(subset=[sig])
        if len(train)<24: continue
        dirn=np.sign(np.corrcoef(train[sig],train["target_ret"])[0,1]) or 1.0
        th=train[sig].median()
        for _,r in test.iterrows():
            rows.append({"on": dirn*r[sig]>=dirn*th, "tr":r["target_ret"]})
    return pd.DataFrame(rows)

print("\nBLOCK-BOOTSTRAP of ON-minus-OFF fwd-factor-return spread (overlap-aware, 6-month blocks)")
print("p = P(bootstrap spread <= 0). Want small p AND positive spread.\n")
rng=np.random.default_rng(0)
for sig in ["breadth50","spy_vol","spy_above_ma200","dispersion","breadth200","fac_63"]:
    d=oos_on_off(sig)
    if len(d)<12: continue
    obs=d[d.on].tr.mean()-d[~d.on].tr.mean()
    # block bootstrap over time (preserve overlap structure)
    n=len(d); bl=6; boots=[]
    arr_on=d.on.values; arr_tr=d.tr.values
    for _ in range(2000):
        idx=[]
        while len(idx)<n:
            s=rng.integers(0,n-1); idx+=list(range(s,min(s+bl,n)))
        idx=np.array(idx[:n])
        on=arr_on[idx]; tr=arr_tr[idx]
        if on.sum()==0 or (~on).sum()==0: continue
        boots.append(tr[on].mean()-tr[~on].mean())
    boots=np.array(boots)
    p=(boots<=0).mean()
    print(f"  {sig:16s} obs_spread={obs:+.4f}  boot_mean={boots.mean():+.4f}  p(<=0)={p:.3f}  frac_off={(~d.on).mean():.2f}")

# ---- How concentrated is the gate gain? top months driving it ----
print("\nCONCENTRATION: for breadth50 gate (exp_off=0), months where gate was OFF and their target_ret")
d=oos_on_off("breadth50")
off=d[~d.on].sort_values("tr")
print("  worst 5 avoided (gate off, negative=good to avoid):", off.tr.head(5).round(3).tolist())
print("  but also avoided these POSITIVE months:", off.tr.tail(5).round(3).tolist())
print(f"  gate-off months: {len(off)}, of which negative (correctly avoided): {(off.tr<0).mean():.2f}")
