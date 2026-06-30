"""Re-baseline strategy configs on the CORRECTED eval (2026-06-11).

Every historical A/B (horizon, hysteresis, sector-neutralize, tilt) was
measured on the inverse-selected tail book (eval rails capped 5% vs 10%
targets). This re-runs the matrix on the fixed eval so config decisions
rest on real measurements. Read-only DB; val window only (TEST is sacred).
"""
from pathlib import Path

from sma.backtest.__main__ import _get_universe
from sma.backtest.risk import eval_rails_for
from sma.backtest.strategies.xgb_top_k import XGBoostTopKStrategy
from sma.eval import evaluate_strategy
from sma.model.predictor import Predictor

# Historical walk-forward models were lost in the 2026-06-11 artifacts
# incident; evals use a model retrained at the val-window start instead
# (train: python -m sma.model train --asof 2025-06-30 --no-cv
#         --models-dir /tmp/sma-eval-models).
EVAL_MODELS = Path("/tmp/sma-eval-models")

CONFIGS = [
    ("baseline k=15 tilt", {}),
    ("tilt OFF (flat)", {"weight_tilt": False}),
    ("hysteresis hold_rank=25", {"hold_rank": 25}),
    ("sector-neutral λ=1", {"sector_neutralize": 1.0}),
    ("k=10", {"k": 10}),
    ("k=20", {"k": 20}),
]

universe = _get_universe()
print(f"{'config':28} {'sharpe':>8} {'ret':>8} {'maxDD':>8} {'hit':>6} {'trades':>7} {'hold_d':>7}")
for name, kw in CONFIGS:
    predictor = Predictor(
        models_dir=EVAL_MODELS, db_path=Path("data/sma.duckdb"),
    )
    k = kw.pop("k", 15)
    s = XGBoostTopKStrategy(
        predictor=predictor, universe=universe, k=k,
        hold_rank=kw.pop("hold_rank", None), **kw,
    )
    r = evaluate_strategy(
        strategy=s, window="val", universe=universe,
        rails=eval_rails_for(s), membership="current",
    )
    print(f"{name:28} {r.sharpe:8.3f} {r.total_return:8.2%} {r.max_drawdown:8.2%} "
          f"{r.hit_rate:6.1%} {r.num_trades:7d} {r.avg_holding_days:7.1f}")
