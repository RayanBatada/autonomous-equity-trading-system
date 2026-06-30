"""Label/objective experiment (2026-06-13): the lean-vs-full result showed
feature selection doesn't fix the regime inversion — the model bakes in
trend-regime weights by fitting RAW returns via RMSE on one regime. Test the
two built-but-untested levers that target exactly that:
  - demean-label: train on cross-sectional (alpha) return, strip market/beta
  - rank objective: XGBRanker per-asof pairwise — rank within date, not RMSE
Regime-split rank-IC vs realized RAW returns. Caches built sets to parquet.
"""
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from loguru import logger
from scipy.stats import spearmanr

from sma.ingest.universe import load_universe
from sma.model.__main__ import (
    _load_earnings_calendar,
    _load_news_for_count_feature,
    _load_politician_trades,
    _load_prices_for_range,
)
from sma.model.loader import build_training_set
from sma.model.trainer import DEFAULT_HYPERPARAMS

DB = Path("data/sma.duckdb")
CACHE = Path("/tmp/sma-exp")
CACHE.mkdir(exist_ok=True)
universe = load_universe("src/sma/universe.yaml")

def _build(start, end, tag):
    fx, fy, fa = CACHE/f"{tag}_x.parquet", CACHE/f"{tag}_y.parquet", CACHE/f"{tag}_a.parquet"
    if fx.exists():
        return (pd.read_parquet(fx), pd.read_parquet(fy)["y"], pd.read_parquet(fa)["a"])
    p = _load_prices_for_range(DB, universe, start - timedelta(days=550), end)
    X, y, a = build_training_set(
        p, universe, train_start=start, train_end=end, forward_horizon_days=30,
        politician_trades=_load_politician_trades(DB),
        earnings=_load_earnings_calendar(DB), news=_load_news_for_count_feature(DB))
    X = X.astype(float)
    X.to_parquet(fx); pd.DataFrame({"y": y}).to_parquet(fy); pd.DataFrame({"a": a}).to_parquet(fa)
    return X, y, a

logger.info("building/loading train + val sets...")
Xtr, ytr, atr = _build(date(2023,1,1), date(2025,6,30), "train")
Xv, yv, av = _build(date(2025,7,1), date(2025,12,1), "val")
logger.info(f"train={len(Xtr)} val={len(Xv)}")

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
    flag = "ROBUST both+" if t>0 and r>0 else "better-rev" if r>-0.030 else "regime-exposed"
    print(f"{name:22} trend_IC={t:+.4f}  rev_IC={r:+.4f}  -> {flag}")

# baseline reg
m = xgb.XGBRegressor(**DEFAULT_HYPERPARAMS); m.fit(Xtr, ytr)
show("reg raw-label", m.predict(Xv))
# demeaned label
ytr_dm = ytr - ytr.groupby(atr.to_numpy()).transform("mean")
m = xgb.XGBRegressor(**DEFAULT_HYPERPARAMS); m.fit(Xtr, ytr_dm)
show("reg demean-label", m.predict(Xv))
# ranking objective (sort by asof, per-date groups)
order = np.argsort(atr.to_numpy(), kind="stable")
Xs, ys, as_ = Xtr.iloc[order], ytr.iloc[order], atr.iloc[order]
groups = as_.value_counts(sort=False).reindex(pd.Index(as_.unique())).tolist()
r = xgb.XGBRanker(**{**DEFAULT_HYPERPARAMS, "objective": "rank:pairwise"})
r.fit(Xs, ys, group=groups)
show("rank objective", r.predict(Xv))
