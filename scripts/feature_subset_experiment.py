"""Full vs lean feature set: does de-concentrating the momentum bet hold IC
in BOTH regimes? (2026-06-13 — the real model-improvement question.)

Lean = data-robust signals (+IC both regimes per per_feature_ic.py) + the
one genuine diversifier (reversal_5d_z) + theory-orthogonal fundamentals/
flow. Drops short-horizon momentum, raw vol, gap, sector-RS (flips), and the
harmful sector-ETF feature. Trains both at 2025-06-30, measures rank-IC per
asof on val, split by the DATA-DRIVEN regime classifier (sma.eval.regime, NOT
a calendar midpoint — 2026-06-25 fix). Winner = higher IC in BOTH regimes.
"""
from datetime import date

import duckdb
import numpy as np
import pandas as pd
import xgboost as xgb
from loguru import logger
from scipy.stats import spearmanr

from sma.eval.regime import classify_regimes
from sma.ingest.universe import load_universe
from sma.model.__main__ import (
    _load_earnings_calendar,
    _load_news_for_count_feature,
    _load_politician_trades,
    _load_prices_for_range,
)
from sma.model.loader import build_training_set
from sma.model.trainer import DEFAULT_HYPERPARAMS

LEAN = [
    "ret_60d", "rsi_14", "dollar_volume_20d", "rel_strength_spy_60d",
    "dist_from_52w_high", "vol_adj_mom_60d",           # robust (+IC both)
    "reversal_5d_z",                                    # diversifier (+ in reversal)
    "earnings_surprise_last", "days_to_next_earnings",  # orthogonal fundamentals
    "politician_flow_30d", "news_count_7d_log",
]
DB = "data/sma.duckdb"
universe = load_universe("src/sma/universe.yaml")

def _set(start, end):
    from datetime import timedelta
    from pathlib import Path
    # Load ~1.5y of price history BEFORE the asof window so features
    # (ret_60d, dist_from_52w_high, ...) resolve; asof range stays [start, end].
    p = _load_prices_for_range(Path(DB), universe, start - timedelta(days=550), end)
    return build_training_set(
        p, universe, train_start=start, train_end=end, forward_horizon_days=30,
        politician_trades=_load_politician_trades(Path(DB)),
        earnings=_load_earnings_calendar(Path(DB)),
        news=_load_news_for_count_feature(Path(DB)),
    )

logger.info("building train set (2023-01..2025-06-30)...")
Xtr, ytr, _ = _set(date(2023, 1, 1), date(2025, 6, 30))
logger.info("building val set (2025-07-01..2025-12-01)...")
Xv, yv, av = _set(date(2025, 7, 1), date(2025, 12, 1))
Xtr = Xtr.astype(float); Xv = Xv.astype(float)
logger.info(f"train={len(Xtr)} val={len(Xv)}")

def train_predict(cols):
    m = xgb.XGBRegressor(**DEFAULT_HYPERPARAMS)
    m.fit(Xtr[cols], ytr)
    return m.predict(Xv[cols])

def regime_ic(pred):
    df = pd.DataFrame(
        {"asof": pd.to_datetime(av.to_numpy()).date, "pred": pred, "y": yv.to_numpy()}
    )
    dates = sorted(df["asof"].unique())
    con = duckdb.connect(DB, read_only=True)
    regimes = classify_regimes(con, dates, universe)
    con.close()
    tr, rv = [], []
    for d, g in df.groupby("asof"):
        reg = regimes.get(d)
        if reg is None or len(g) < 20 or g["pred"].std() < 1e-12:
            continue
        ic = spearmanr(g["pred"], g["y"]).correlation
        (tr if reg[0] == "trend" else rv).append(ic)
    return (
        np.mean(tr) if tr else float("nan"),
        np.mean(rv) if rv else float("nan"),
        len(tr),
        len(rv),
    )

for name, cols in [("FULL (21f)", list(Xtr.columns)), ("LEAN (11f)", LEAN)]:
    t, r, nt, nr = regime_ic(train_predict(cols))
    flag = "ROBUST (both +)" if t > 0 and r > 0 else "regime-exposed"
    print(f"{name:12} trend_IC={t:+.4f}({nt}d)  rev_IC={r:+.4f}({nr}d)  -> {flag}")
