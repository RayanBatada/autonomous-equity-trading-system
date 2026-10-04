"""DE-CONCENTRATION test: FULL (21 feats) vs INFORMED demean model, regime IC.

Adapted from arm_b.py. For each held-out asof, build the FULL feature panel for
every weekly train_asof STRICTLY BEFORE the asof (purged by FWD+1 sessions so no
forward-30d label crosses the asof), train a demean XGBRegressor (prod
DEFAULT_HYPERPARAMS, cross-sectional-demeaned labels), predict the universe at
the asof, Spearman IC vs realized 30d-fwd return.

Two arms share ONE feature cache + identical label/universe logic, differing
ONLY by which feature COLUMNS the model sees:

  FULL     : all 21 FEATURE_NAMES (the production momentum-concentrated set).
  INFORMED : de-concentrated subset from the corrected full-history regime-IC
             map (arm_a_full.log). KEEP the both-regime-robust vol diversifiers
             + ONE long-horizon momentum (drop the collinear twins) + a minimal
             trend core + the orthogonal non-momentum side-data. DROP the
             short-horizon momentum flippers, the harmful sector-relative
             features, the collinear momentum twins, and the FAKE reversal_5d_z.

Universe = sma-pit.duckdb durable de-biased copy (prod prices + recovered
decliners), deduped to source='yfinance'. Prod sma.duckdb is NEVER touched
(a live `sma.ingest run` holds its write lock anyway).

HONEST LIMITATION: the 4 side-data features (politician_flow_30d,
days_to_next_earnings, news_count_7d_log, earnings_surprise_last) are built from
PRICE data only here, so they are CONSTANT (no variance) for BOTH arms -> they
contribute nothing. The INFORMED arm is therefore effectively a 6-live-feature
model {vol_20d, vol_60d, ret_60d, rsi_14, dist_from_52w_high, dollar_volume_20d}
and the test really measures: does dropping the redundant/harmful MOMENTUM
features (keeping the vol diversifiers + a minimal momentum core) help reversal
IC without killing trend IC. The side-data diversification is NOT exercised.
"""
import sys
import time
from datetime import date

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

SCRATCH = "/private/tmp/claude-501/-Users-youruser-SecondBrain/f0fae5bb-6213-478f-b88e-e8747330c663/scratchpad"
sys.path.insert(0, SCRATCH)
import shared_pit  # noqa: E402

shared_pit.PROD_DB = "/Users/youruser/.sma-pit/sma-pit.duckdb"
from shared_pit import load_prices_df, make_memcon  # noqa: E402

from sma.features.builder import FEATURE_NAMES, build_features  # noqa: E402
from sma.ingest.universe import load_universe  # noqa: E402
from sma.model.trainer import DEFAULT_HYPERPARAMS, train_xgb  # noqa: E402

ETFS = {"SPY", "NANC", "XLB", "XLC", "XLE", "XLF", "XLI", "XLK",
        "XLP", "XLRE", "XLU", "XLV", "XLY"}

INFORMED = [
    "vol_20d", "vol_60d",            # robust + in BOTH regimes (genuine diversifiers)
    "ret_60d",                       # ONE long-horizon mom (drop collinear spy_60d/vol_adj)
    "rsi_14", "dist_from_52w_high",  # minimal trend core (non-collinear w/ ret_60d)
    "dollar_volume_20d",             # liquidity control
    "earnings_surprise_last", "days_to_next_earnings",
    "politician_flow_30d", "news_count_7d_log",  # orthogonal side-data
]
ARMS = {"FULL": list(FEATURE_NAMES), "INFORMED": INFORMED}

TRAIN_START = date(2018, 6, 1)
STRIDE = 5
FWD = 30
PURGE = FWD + 1

REVERSAL_ASOFS = [date(2020, 3, 16), date(2022, 6, 15), date(2025, 4, 15)]
TREND_ASOFS = [date(2024, 1, 16), date(2021, 7, 7)]

mem = make_memcon(db_path=shared_pit.PROD_DB)
uni_all = load_universe(
    "/Users/youruser/code/stock-market-predictor-agents/src/sma/universe.yaml")
universe = [t for t in uni_all if t not in ETFS]

end = mem.execute("SELECT MAX(date) FROM prices").fetchone()[0]
prices = load_prices_df(mem, TRAIN_START, end)
sessions = sorted(prices["date"].unique())
sess_idx = {d: i for i, d in enumerate(sessions)}

px = prices[prices["ticker"].isin(universe)].copy()
px = px[(px["close"] > 0) & (px["open"] > 0)]
px["adj_open"] = px["open"] * px["adj_close"] / px["close"]
ac = px.pivot_table(index="date", values="adj_close", columns="ticker").reindex(sessions)
ao = px.pivot_table(index="date", values="adj_open", columns="ticker").reindex(sessions)


def fwd_label(asof):
    i = sess_idx.get(asof)
    if i is None or i + 1 >= len(sessions) or i + FWD >= len(sessions):
        return pd.Series(dtype=float)
    entry = ao.iloc[i + 1]
    exit_ = ac.iloc[i + FWD]
    out = exit_ / entry - 1.0
    return out[(entry > 0) & (exit_ > 0)].dropna()


_feat_cache: dict = {}


def feats_at(asof):
    if asof not in _feat_cache:
        _feat_cache[asof] = build_features(prices, universe, asof)
    return _feat_cache[asof]


def build_xy(train_asofs):
    rows, labels, afs = [], [], []
    for a in train_asofs:
        feats = feats_at(a)
        if feats.empty:
            continue
        lab = fwd_label(a)
        idx = [t for t in feats.index if t in lab.index]
        if len(idx) < 5:
            continue
        yv = lab.loc[idx].to_numpy(dtype=float)
        yv = yv - yv.mean()  # cross-sectional demean within date
        for j, t in enumerate(idx):
            rows.append({c: feats.loc[t, c] for c in FEATURE_NAMES})
            labels.append(yv[j])
            afs.append(a)
    if not rows:
        return None
    X = pd.DataFrame(rows)[FEATURE_NAMES]
    return X, pd.Series(labels), pd.Series(afs)


def predict_ic(model, asof, cols):
    feats = feats_at(asof)
    lab = fwd_label(asof)
    idx = [t for t in feats.index if t in lab.index]
    if len(idx) < 10:
        return np.nan, len(idx)
    preds = model.predict(feats.loc[idx, cols])
    rv = lab.loc[idx].to_numpy(dtype=float)
    if np.std(preds) < 1e-12:
        return np.nan, len(idx)
    return float(spearmanr(preds, rv).correlation), len(idx)


def run_asof(asof):
    i = sess_idx[asof]
    train_end = sessions[i - PURGE]
    train_asofs = [d for d in sessions if TRAIN_START <= d <= train_end][::STRIDE]
    built = build_xy(train_asofs)
    if built is None:
        return []
    Xfull, y, af = built
    out = []
    for arm, cols in ARMS.items():
        t0 = time.time()
        Xa = Xfull[cols]
        model = train_xgb(Xa, y, hyperparams=DEFAULT_HYPERPARAMS.copy(),
                          asof_dates=af, objective="reg")
        ic, n = predict_ic(model, asof, cols)
        out.append(dict(asof=asof, arm=arm, ic=ic, n_pred=n, n_train=len(Xa),
                        n_train_asofs=len(train_asofs), train_end=train_end,
                        n_live_cols=int((Xa.std() > 1e-12).sum()),
                        secs=round(time.time() - t0, 1)))
    return out


# ---- MANY-ASOF walk-forward: monthly grid 2019-2025, data-driven regime,
#      paired delta (same dates -> date-level noise cancels = the powerful test) ----
from sma.eval.regime import classify_regimes  # noqa: E402

grid = []
for yr in range(2019, 2026):
    for mo in range(1, 13):
        cand = [d for d in sessions
                if d >= date(yr, mo, 1) and sess_idx[d] + FWD < len(sessions)]
        if cand:
            grid.append(cand[0])
grid = sorted(set(grid))
regimes = classify_regimes(mem, grid, universe)  # {asof: (label, factor_ret)}

results = []
for asof in grid:
    reg = regimes.get(asof)
    if reg is None:
        continue
    label = "REVERSAL" if reg[0] == "reversal" else "TREND"
    by_arm = {}
    for r in run_asof(asof):
        r["regime"] = label
        results.append(r)
        by_arm[r["arm"]] = r["ic"]
    if "FULL" in by_arm and "INFORMED" in by_arm:
        print(f"{label:8} {asof} FULL={by_arm['FULL']:+.4f} "
              f"INFORMED={by_arm['INFORMED']:+.4f} "
              f"delta={by_arm['INFORMED'] - by_arm['FULL']:+.4f}", flush=True)

df = pd.DataFrame(results)
df.to_csv(f"{SCRATCH}/feature_deconc_results.csv", index=False)

print("\n========== mean IC by regime x arm ==========")
print(df.pivot_table(index="regime", columns="arm", values="ic", aggfunc="mean").round(4))

wide = df.pivot_table(index="asof", columns="arm", values="ic").dropna(
    subset=["FULL", "INFORMED"])
d = (wide["INFORMED"] - wide["FULL"]).to_numpy()
n = len(d)
mean_d, sd = float(d.mean()), float(d.std(ddof=1))
tstat = mean_d / (sd / np.sqrt(n)) if sd > 0 else float("nan")
print(f"\n===== PAIRED DELTA (INFORMED - FULL), n={n} asofs =====")
print(f"mean delta = {mean_d:+.4f}  std = {sd:.4f}  t = {tstat:+.2f}  "
      f"%positive = {(d > 0).mean():.0%}   (gate +0.005)")
for label in ("REVERSAL", "TREND"):
    sub = df[df.regime == label].pivot_table(
        index="asof", columns="arm", values="ic").dropna(subset=["FULL", "INFORMED"])
    if len(sub) > 1:
        dd = (sub["INFORMED"] - sub["FULL"]).to_numpy()
        tt = dd.mean() / (dd.std(ddof=1) / np.sqrt(len(dd))) if dd.std() > 0 else float("nan")
        print(f"  {label}: n={len(dd)}  mean delta={dd.mean():+.4f}  "
              f"t={tt:+.2f}  %pos={(dd > 0).mean():.0%}")
mem.close()
