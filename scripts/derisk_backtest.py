"""Does drawdown-scaled de-risking help? (2026-06-15) Val window (has the
Sept-Nov momentum crash). Same model/config; vary drawdown_derisk_slope.
Winner reduces max-drawdown without gutting return."""
from pathlib import Path

from sma.backtest.__main__ import _get_universe
from sma.backtest.strategies.xgb_top_k import XGBoostTopKStrategy
from sma.eval import evaluate_strategy
from sma.model.predictor import Predictor
from sma.risk.rails import RiskRails

MODELS = Path("/tmp/sma-eval-models-21f")
universe = _get_universe()
print(f"{'derisk slope':14} {'sharpe':>8} {'ret':>8} {'maxDD':>8} {'trades':>7} {'hold_d':>7}")
for slope in [0.0, 1.5, 3.0, 5.0]:
    s = XGBoostTopKStrategy(
        predictor=Predictor(models_dir=MODELS, db_path=Path("data/sma.duckdb")),
        universe=universe, k=15, hold_rank=30, sector_neutralize=1.0,
    )
    rails = RiskRails(
        stop_loss_pct=0.0, max_position_pct=s.target_weight, min_hold_days=7,
        cash_floor_pct=0.05, drawdown_derisk_start=0.05, drawdown_derisk_slope=slope,
    )
    r = evaluate_strategy(strategy=s, window="val", universe=universe,
                          rails=rails, membership="current")
    tag = "OFF" if slope == 0 else f"{slope}"
    print(f"{tag:14} {r.sharpe:8.3f} {r.total_return:8.2%} {r.max_drawdown:8.2%} "
          f"{r.num_trades:7d} {r.avg_holding_days:7.1f}")
