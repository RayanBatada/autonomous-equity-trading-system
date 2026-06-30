"""True walk-forward validation of the shipped config (2026-06-12).

Trains boundary models every ~21 sessions through the val window (cached in
~/.sma-wf-models), then runs the standard eval — Predictor resolves the
right boundary model per asof. This is the honest version of the +1.60
single-model number.
"""
from pathlib import Path

import duckdb

from sma.backtest.__main__ import _get_universe
from sma.backtest.strategies.xgb_top_k import XGBoostTopKStrategy
from sma.backtest.windows import window_dates
from sma.eval import evaluate_strategy
from sma.eval.walkforward import ensure_boundary_models, retrain_boundaries
from sma.model.predictor import Predictor
from sma.risk.rails import RiskRails

CACHE = Path.home() / ".sma-wf-models"
start, end = window_dates("val")
con = duckdb.connect("data/sma.duckdb", read_only=True)
sessions = [r[0] for r in con.execute(
    "SELECT DISTINCT date FROM prices WHERE date BETWEEN ? AND ? ORDER BY date",
    [start, end]).fetchall()]
con.close()

boundaries = retrain_boundaries(sessions, every=21)
print(f"val {start}..{end}: {len(sessions)} sessions, {len(boundaries)} boundaries")
n = ensure_boundary_models(boundaries, models_dir=CACHE)
print(f"trained {n} new boundary models")

universe = _get_universe()
s = XGBoostTopKStrategy(
    predictor=Predictor(models_dir=CACHE, db_path=Path("data/sma.duckdb")),
    universe=universe, k=15, hold_rank=30, sector_neutralize=1.0,
)
rails = RiskRails(stop_loss_pct=0.0, max_position_pct=s.target_weight, min_hold_days=7)
r = evaluate_strategy(strategy=s, window="val", universe=universe,
                      rails=rails, membership="current")
print(f"WALK-FORWARD val: sharpe {r.sharpe:.3f}  ret {r.total_return:.2%}  "
      f"maxDD {r.max_drawdown:.2%}  hit {r.hit_rate:.1%}  trades {r.num_trades}  "
      f"hold {r.avg_holding_days:.1f}d")
