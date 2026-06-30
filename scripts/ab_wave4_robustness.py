"""Wave 4: robustness of the wave-3 winner (raw λ=1 hr=30 mh=7).

Best-of-21-configs on one window = selection bias. Two probes:
(1) neighbors — a robust optimum sits on a plateau, an overfit one on a
spike; (2) the same config on the TRAIN window (different period; the model
saw this data in training so absolute numbers are optimistic, but a NEGATIVE
result would kill the config)."""
from pathlib import Path

from sma.backtest.__main__ import _get_universe
from sma.backtest.strategies.xgb_top_k import XGBoostTopKStrategy
from sma.eval import evaluate_strategy
from sma.model.predictor import Predictor
from sma.risk.rails import RiskRails

RAW = Path("/tmp/sma-eval-models")
GRID = [
    ("val λ=1.0 hr=30 mh=7  *", "val", 1.0, 30, 7),
    ("val λ=0.75 hr=30 mh=7",   "val", 0.75, 30, 7),
    ("val λ=1.0 hr=25 mh=7",    "val", 1.0, 25, 7),
    ("val λ=1.0 hr=35 mh=7",    "val", 1.0, 35, 7),
    ("val λ=1.0 hr=30 mh=5",    "val", 1.0, 30, 5),
    ("val λ=1.0 hr=30 mh=10",   "val", 1.0, 30, 10),
    ("TRAIN λ=1.0 hr=30 mh=7",  "train", 1.0, 30, 7),
]
universe = _get_universe()
print(f"{'config':26} {'sharpe':>8} {'ret':>8} {'maxDD':>8} {'hit':>6} {'trades':>7} {'hold_d':>7}")
for name, window, lam, hr, mh in GRID:
    s = XGBoostTopKStrategy(
        predictor=Predictor(models_dir=RAW, db_path=Path("data/sma.duckdb")),
        universe=universe, k=15, hold_rank=hr, sector_neutralize=lam,
    )
    rails = RiskRails(stop_loss_pct=0.0, max_position_pct=s.target_weight, min_hold_days=mh)
    r = evaluate_strategy(
        strategy=s, window=window, universe=universe, rails=rails, membership="current",
    )
    print(f"{name:26} {r.sharpe:8.3f} {r.total_return:8.2%} {r.max_drawdown:8.2%} "
          f"{r.hit_rate:6.1%} {r.num_trades:7d} {r.avg_holding_days:7.1f}")
