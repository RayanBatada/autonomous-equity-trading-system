"""Wave 2: turnover policy grid + demeaned-label A/B (2026-06-11 strategy review).

The binding constraint is turnover (avg hold 3.1d on a 30d signal). Grid
hold_rank (hysteresis buffer) x min_hold_days against the raw-label model,
then re-run the leaders on the demeaned-label model. Val window only.
"""
from pathlib import Path

from sma.backtest.__main__ import _get_universe
from sma.backtest.strategies.xgb_top_k import XGBoostTopKStrategy
from sma.eval import evaluate_strategy
from sma.model.predictor import Predictor
from sma.risk.rails import RiskRails

RAW = Path("/tmp/sma-eval-models")
DEMEANED = Path("/tmp/sma-eval-models-demeaned")

GRID = [
    ("raw  base (hr=15 mh=1)", RAW, 15, 1),
    ("raw  hr=30 mh=1", RAW, 30, 1),
    ("raw  hr=30 mh=7", RAW, 30, 7),
    ("raw  hr=45 mh=14", RAW, 45, 14),
    ("raw  hr=30 mh=21", RAW, 30, 21),
    ("dm   base (hr=15 mh=1)", DEMEANED, 15, 1),
    ("dm   hr=30 mh=7", DEMEANED, 30, 7),
    ("dm   hr=45 mh=14", DEMEANED, 45, 14),
]

universe = _get_universe()
print(f"{'config':26} {'sharpe':>8} {'ret':>8} {'maxDD':>8} {'hit':>6} {'trades':>7} {'hold_d':>7}")
for name, mdir, hold_rank, min_hold in GRID:
    s = XGBoostTopKStrategy(
        predictor=Predictor(models_dir=mdir, db_path=Path("data/sma.duckdb")),
        universe=universe, k=15, hold_rank=hold_rank,
    )
    rails = RiskRails(
        stop_loss_pct=0.0, max_position_pct=s.target_weight,
        min_hold_days=min_hold,
    )
    r = evaluate_strategy(
        strategy=s, window="val", universe=universe,
        rails=rails, membership="current",
    )
    print(f"{name:26} {r.sharpe:8.3f} {r.total_return:8.2%} {r.max_drawdown:8.2%} "
          f"{r.hit_rate:6.1%} {r.num_trades:7d} {r.avg_holding_days:7.1f}")
