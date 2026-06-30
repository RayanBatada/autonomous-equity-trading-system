"""OOS evaluation + gate-effect estimate for the regime-momentum gate.

Honest protocol: every signal's DIRECTION and THRESHOLD are learned on TRAIN
years only, then applied to held-out TEST years. Two schemes:
  - leave-one-year-out (k-fold by year)
  - expanding window (train on all prior years, test next)
Metrics: pooled OOS AUC, hit-rate, regime-conditional forward factor return,
and a gated equity curve (return + max drawdown) vs always-on.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path.home() / ".sma-pit"
panel = pd.read_parquet(OUT / "regime_panel.parquet")
panel["year"] = panel["date"].dt.year

SIGNALS = ["fac_21","fac_63","dispersion","breadth50","breadth200",
           "spy_vol","spy_above_ma200","spy_ma200_dist"]

def auc(score, label):
    # rank-based AUC; score oriented so higher => predict trend=1
    label = np.asarray(label); score = np.asarray(score)
    m = np.isfinite(score) & np.isfinite(label)
    score, label = score[m], label[m]
    pos = score[label==1]; neg = score[label==0]
    if len(pos)==0 or len(neg)==0: return np.nan
    # average ranks for ties
    s = pd.Series(score); r = s.rank().values
    rp = r[label==1].sum()
    return (rp - len(pos)*(len(pos)+1)/2) / (len(pos)*len(neg))

def eval_signal_oos(df, sig, scheme):
    """Return pooled OOS predictions (signal value, learned direction applied,
    predicted trend-on bool, actual trend, target_ret)."""
    years = sorted(df.year.unique())
    rows = []
    for ty in years:
        if scheme == "loyo":
            train = df[df.year != ty]
        else:  # expanding
            train = df[df.year < ty]
            if len(train) < 24:  # need a warmup
                continue
        test = df[df.year == ty]
        tr = train.dropna(subset=[sig])
        if len(tr) < 12: continue
        # learn direction: corr of signal with target_ret on train
        dirn = np.sign(np.corrcoef(tr[sig], tr["target_ret"])[0,1])
        if dirn == 0: dirn = 1.0
        thresh = tr[sig].median()  # train median threshold
        for _, r in test.dropna(subset=[sig]).iterrows():
            oriented = dirn * r[sig]
            on = oriented >= dirn * thresh  # signal in "trend" direction
            rows.append({"date": r["date"], "year": ty, "sig": r[sig],
                         "oriented": oriented, "trend_on": bool(on),
                         "trend": r["trend"], "target_ret": r["target_ret"]})
    return pd.DataFrame(rows)

def summarize(oos, name):
    if len(oos) == 0:
        return None
    a = auc(oos["oriented"], oos["trend"])
    hit = (oos["trend_on"] == (oos["trend"]==1)).mean()
    on = oos[oos["trend_on"]]; off = oos[~oos["trend_on"]]
    ret_on = on["target_ret"].mean() if len(on) else np.nan
    ret_off = off["target_ret"].mean() if len(off) else np.nan
    spread = ret_on - ret_off
    return {"signal": name, "n": len(oos), "AUC": round(a,3), "hit_rate": round(hit,3),
            "fwd_ret_when_ON": round(ret_on,4), "fwd_ret_when_OFF": round(ret_off,4),
            "ON_minus_OFF": round(spread,4), "frac_ON": round(oos["trend_on"].mean(),3)}

def max_dd(equity):
    peak = np.maximum.accumulate(equity)
    return float((equity/peak - 1.0).min())

print("="*90)
for freq in ["M","W"]:
    sub = panel[panel.freq==freq].copy()
    print(f"\n############### FREQ={freq}  (n={len(sub)}, trend base rate={sub.trend.mean():.3f}) ###############")
    for scheme in ["loyo","expanding"]:
        print(f"\n----- scheme={scheme} -----")
        res = []
        for sig in SIGNALS:
            oos = eval_signal_oos(sub, sig, scheme)
            s = summarize(oos, sig)
            if s: res.append(s)
        rdf = pd.DataFrame(res).sort_values("AUC", ascending=False)
        print(rdf.to_string(index=False))

# ---- Combined multi-signal logistic model, expanding, monthly --------------
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

print("\n" + "="*90)
print("COMBINED logistic (all signals), expanding OOS, MONTHLY")
sub = panel[panel.freq=="M"].dropna(subset=SIGNALS).copy().reset_index(drop=True)
years = sorted(sub.year.unique())
comb_rows = []
for ty in years:
    train = sub[sub.year < ty]
    if len(train) < 24: continue
    test = sub[sub.year == ty]
    sc = StandardScaler().fit(train[SIGNALS])
    lr = LogisticRegression(max_iter=1000, C=0.5).fit(sc.transform(train[SIGNALS]), train["trend"])
    p = lr.predict_proba(sc.transform(test[SIGNALS]))[:,1]
    for (_, r), pp in zip(test.iterrows(), p):
        comb_rows.append({"date": r["date"], "year": ty, "p": pp,
                          "trend": r["trend"], "target_ret": r["target_ret"]})
comb = pd.DataFrame(comb_rows)
if len(comb):
    a = auc(comb["p"], comb["trend"])
    on = comb[comb.p>=0.5]; off = comb[comb.p<0.5]
    print(f"  n={len(comb)} AUC={a:.3f} hit={(((comb.p>=0.5).astype(int))==comb.trend).mean():.3f}")
    print(f"  fwd_ret ON(p>=.5)={on.target_ret.mean():.4f}  OFF={off.target_ret.mean():.4f}  spread={on.target_ret.mean()-off.target_ret.mean():.4f}")
    comb.to_parquet(OUT/"combined_oos.parquet")

# ---- GATE EFFECT: monthly equity curve, best single signal + combined ------
print("\n" + "="*90)
print("GATE EFFECT (monthly, expanding OOS). exposure in {0, 0.5, 1}.")
print("always-on = factor strategy with exposure=1 every month.\n")

def gate_curve(oos_df, exposure_off):
    """oos_df has date, trend_on, target_ret. Returns dict of stats vs always-on."""
    d = oos_df.sort_values("date")
    exp = np.where(d["trend_on"], 1.0, exposure_off)
    gated_r = exp * d["target_ret"].values
    base_r = d["target_ret"].values
    g_eq = np.cumprod(1 + gated_r); b_eq = np.cumprod(1 + base_r)
    return {
        "n_months": len(d),
        "always_on_total": round(b_eq[-1]-1, 4), "always_on_maxDD": round(max_dd(b_eq),4),
        "always_on_sharpe": round(base_r.mean()/base_r.std()*np.sqrt(12),3),
        "gated_total": round(g_eq[-1]-1,4), "gated_maxDD": round(max_dd(g_eq),4),
        "gated_sharpe": round(gated_r.mean()/gated_r.std()*np.sqrt(12),3),
        "frac_time_gated": round((~d["trend_on"]).mean(),3),
    }

subM = panel[panel.freq=="M"].copy()
# pick best signal by expanding-OOS AUC (computed above visually); evaluate all gates
gate_summ = []
for sig in SIGNALS:
    oos = eval_signal_oos(subM, sig, "expanding")
    if len(oos) < 12: continue
    for goff in [0.0, 0.5]:
        st = gate_curve(oos, goff); st["signal"]=sig; st["exp_off"]=goff
        gate_summ.append(st)
# combined model gate
if len(comb):
    comb2 = comb.copy(); comb2["trend_on"] = comb2["p"]>=0.5
    for goff in [0.0, 0.5]:
        st = gate_curve(comb2[["date","trend_on","target_ret"]], goff)
        st["signal"]="COMBINED_logit"; st["exp_off"]=goff
        gate_summ.append(st)
gdf = pd.DataFrame(gate_summ)
cols = ["signal","exp_off","n_months","always_on_total","gated_total","always_on_maxDD","gated_maxDD","always_on_sharpe","gated_sharpe","frac_time_gated"]
print(gdf[cols].to_string(index=False))
gdf.to_parquet(OUT/"gate_effects.parquet")
print("\nsaved: regime_panel.parquet, combined_oos.parquet, gate_effects.parquet")
