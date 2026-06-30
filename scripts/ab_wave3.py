"""Wave 3: combine the wave-1 winners (sector-neutral + hysteresis + breadth),
raw vs demeaned label. Val window only."""
from pathlib import Path

from sma.backtest.__main__ import _get_universe
from sma.backtest.strategies.xgb_top_k import XGBoostTopKStrategy
from sma.eval import evaluate_strategy
from sma.model.predictor import Predictor
from sma.risk.rails import RiskRails

RAW = Path("/tmp/sma-eval-models")
DM = Path("/tmp/sma-eval-models-demeaned")

GRID = [
    ("raw λ=1 hr=30",          RAW, dict(k=15, hold_rank=30, sector_neutralize=1.0), 1),
    ("raw λ=1 hr=30 mh=7",     RAW, dict(k=15, hold_rank=30, sector_neutralize=1.0), 7),
    ("raw λ=1 k=20 hr=40",     RAW, dict(k=20, hold_rank=40, sector_neutralize=1.0), 1),
    ("raw λ=1 k=20 hr=40 mh=7",RAW, dict(k=20, hold_rank=40, sector_neutralize=1.0), 7),
    ("dm  λ=1 hr=30",          DM,  dict(k=15, hold_rank=30, sector_neutralize=1.0), 1),
    ("dm  λ=1 k=20 hr=40 mh=7",DM,  dict(k=20, hold_rank=40, sector_neutralize=1.0), 7),
    ("dm  λ=0.5 k=20 hr=40",   DM,  dict(k=20, hold_rank=40, sector_neutralize=0.5), 1),
]

universe = _get_universe()
print(f"{'config':27} {'sharpe':>8} {'ret':>8} {'maxDD':>8} {'hit':>6} {'trades':>7} {'hold_d':>7}")
for name, mdir, kw, min_hold in GRID:
    s = XGBoostTopKStrategy(
        predictor=Predictor(models_dir=mdir, db_path=Path("data/sma.duckdb")),
        universe=universe, **kw,
    )
    rails = RiskRails(
        stop_loss_pct=0.0, max_position_pct=s.target_weight, min_hold_days=min_hold,
    )
    r = evaluate_strategy(
        strategy=s, window="val", universe=universe, rails=rails, membership="current",
    )
    print(f"{name:27} {r.sharpe:8.3f} {r.total_return:8.2%} {r.max_drawdown:8.2%} "
          f"{r.hit_rate:6.1%} {r.num_trades:7d} {r.avg_holding_days:7.1f}")
