"""Regime-momentum gate viability test.

Question: does any CONTEMPORANEOUS (data<=t), look-ahead-free signal predict the
FORWARD momentum-factor regime (sign of forward 30d top-minus-bottom-momentum
return) out-of-sample? If yes, does gating momentum exposure by it improve OOS
return/drawdown vs always-on?

Prod DB is opened READ ONLY. Nothing is written back.
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

OUT = Path.home() / ".sma-pit"
OUT.mkdir(exist_ok=True)
DB = "/Users/youruser/code/stock-market-predictor-agents/data/sma.duckdb"

con = duckdb.connect(DB, read_only=True)

# ---- 1. Build a continuous coalesced adj_close panel -----------------------
# priority: yfinance_hist (2018-2022) -> yfinance (2023+) -> alpaca fallback
print("loading prices...")
df = con.execute("""
    WITH ranked AS (
      SELECT ticker, date, adj_close,
             ROW_NUMBER() OVER (PARTITION BY ticker, date
               ORDER BY CASE source WHEN 'yfinance_hist' THEN 0
                                    WHEN 'yfinance' THEN 1
                                    WHEN 'alpaca' THEN 2 ELSE 3 END) rn
      FROM prices
      WHERE source IN ('yfinance_hist','yfinance','alpaca')
        AND adj_close IS NOT NULL AND adj_close > 0
    )
    SELECT ticker, date, adj_close FROM ranked WHERE rn = 1
""").fetchdf()
con.close()
df["date"] = pd.to_datetime(df["date"])
px = df.pivot(index="date", columns="ticker", values="adj_close").sort_index()
print("panel:", px.shape, px.index.min().date(), px.index.max().date())

SPY = px["SPY"].dropna()
stocks = px.drop(columns=["SPY"])  # universe excludes SPY for the factor

# daily simple returns
rets = stocks.pct_change()
spy_rets = SPY.pct_change()

dates = px.index
# trading-day index helpers
def shift_sessions(i, n):
    return i - n

# ---- 2. Daily momentum factor return series --------------------------------
# At each day d, form buckets by past-60d return measured at d-1 (look-ahead
# free), realize the next 1-day return (d-1 -> d). Factor_d = top tercile mean
# 1d return - bottom tercile mean 1d return. This is the canonical tradeable
# momentum-factor daily P&L (a portfolio formed yesterday earns today's move).
LB = 60  # momentum formation lookback (sessions), matches classify_regimes past_sessions
QTILE = 3

past60 = stocks / stocks.shift(LB) - 1.0  # past-60d return as of each day
# one-day forward (next day) return aligned so factor_d uses formation at d-1
fwd1 = rets  # return realized on day d (close d-1 -> close d)

factor = pd.Series(index=dates, dtype=float)
arr_signal = past60.shift(1).values  # formation known at d-1
arr_fwd = fwd1.values
cols = stocks.columns
for i in range(len(dates)):
    s = arr_signal[i]; f = arr_fwd[i]
    m = np.isfinite(s) & np.isfinite(f)
    if m.sum() < QTILE * 2:
        continue
    sv = s[m]; fv = f[m]
    order = np.argsort(sv)
    k = len(order) // QTILE
    if k == 0:
        continue
    bot = order[:k]; top = order[-k:]
    factor.iloc[i] = fv[top].mean() - fv[bot].mean()

factor = factor.dropna()
print("daily factor mean (ann):", factor.mean()*252, "sharpe:", factor.mean()/factor.std()*np.sqrt(252))

# ---- 3. Target: forward 30d momentum-factor regime at each asof ------------
# Reproduces classify_regimes: top-minus-bottom (formed on past-60d at asof)
# realized over forward 30 sessions. Sign => trend(+)/reversal(-). LABEL ONLY.
FWD = 30
def fwd_factor_return(asof_i):
    if asof_i + FWD >= len(dates) or asof_i - LB < 0:
        return None
    s = past60.iloc[asof_i].values            # formation at asof (data<=t, ok for label)
    p_entry = stocks.iloc[asof_i].values
    p_exit = stocks.iloc[asof_i + FWD].values
    fwdret = p_exit / p_entry - 1.0
    m = np.isfinite(s) & np.isfinite(fwdret)
    if m.sum() < QTILE * 2:
        return None
    sv = s[m]; fv = fwdret[m]
    order = np.argsort(sv); k = len(order)//QTILE
    if k == 0:
        return None
    return fv[order[-k:]].mean() - fv[order[:k]].mean()

# ---- 4. Contemporaneous signals (data <= t only) ---------------------------
ma50 = stocks.rolling(50).mean()
ma200 = stocks.rolling(200).mean()
spy_ma200 = SPY.rolling(200).mean()
ret20 = stocks / stocks.shift(20) - 1.0
factor_cum = factor.cumsum()  # for trailing factor returns

def signals_at(asof_i):
    d = dates[asof_i]
    out = {}
    # (a) trailing realized momentum-FACTOR return (factor autocorrelation)
    fwin = factor.loc[:d]
    out["fac_21"] = fwin.iloc[-21:].sum() if len(fwin) >= 21 else np.nan
    out["fac_63"] = fwin.iloc[-63:].sum() if len(fwin) >= 63 else np.nan
    # (b) cross-sectional dispersion of trailing 20d returns
    r = ret20.iloc[asof_i].values
    out["dispersion"] = np.nanstd(r[np.isfinite(r)]) if np.isfinite(r).sum() > 10 else np.nan
    # (c) breadth: % above own 50d / 200d MA
    p = stocks.iloc[asof_i].values
    m50 = ma50.iloc[asof_i].values; m200 = ma200.iloc[asof_i].values
    v50 = np.isfinite(p) & np.isfinite(m50)
    v200 = np.isfinite(p) & np.isfinite(m200)
    out["breadth50"] = (p[v50] > m50[v50]).mean() if v50.sum() > 10 else np.nan
    out["breadth200"] = (p[v200] > m200[v200]).mean() if v200.sum() > 10 else np.nan
    # (d) trailing market realized vol (SPY 20d, annualized)
    sr = spy_rets.loc[:d].iloc[-20:]
    out["spy_vol"] = sr.std() * np.sqrt(252) if sr.notna().sum() >= 15 else np.nan
    # (e) SPY trend: above/below 200d MA (and signed distance)
    sp = SPY.loc[d] if d in SPY.index else np.nan
    spm = spy_ma200.loc[d] if d in spy_ma200.index else np.nan
    out["spy_above_ma200"] = float(sp > spm) if np.isfinite(sp) and np.isfinite(spm) else np.nan
    out["spy_ma200_dist"] = (sp/spm - 1.0) if np.isfinite(sp) and np.isfinite(spm) else np.nan
    return out

# ---- 5. Build the asof panel (monthly AND weekly) --------------------------
def build_asofs(freq):
    # last trading day of each month / each week
    s = pd.Series(dates, index=dates)
    if freq == "M":
        grp = s.groupby([dates.year, dates.month]).max()
    else:
        iso = dates.isocalendar()
        grp = s.groupby([iso.year.values, iso.week.values]).max()
    return sorted(set(grp.values))

records = []
for freq in ["M", "W"]:
    for asof in build_asofs(freq):
        asof_i = dates.get_loc(asof)
        if asof_i - 200 < 0:  # need 200d MA history
            continue
        tgt = fwd_factor_return(asof_i)
        if tgt is None:
            continue
        sig = signals_at(asof_i)
        rec = {"date": asof, "freq": freq, "asof_i": asof_i,
               "target_ret": tgt, "trend": int(tgt >= 0)}
        rec.update(sig)
        records.append(rec)

panel = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
panel.to_parquet(OUT / "regime_panel.parquet")
print("\nPanel built:", panel.shape)
print("monthly asofs:", (panel.freq=="M").sum(), "weekly:", (panel.freq=="W").sum())
print("trend base rate (monthly):", panel[panel.freq=="M"].trend.mean())
print("trend base rate (weekly):", panel[panel.freq=="W"].trend.mean())
print(panel[panel.freq=="M"][["date","target_ret","trend","fac_21","fac_63","dispersion","breadth50","breadth200","spy_vol","spy_above_ma200"]].head(8).to_string())
