"""Sector-cap re-validation: does loosening max_sector_pct beyond 0.35 pay,
AFTER the reversal-regime hit?

The model's top ranks cluster in semis/IT; the 35% sector cap blocks piling in.
This holds the live strategy config fixed (k=15, hold_rank=30, sector_neutralize=1.0,
min_hold=7) and varies ONLY the sector cap, on the corrected surface:
parity-fixed sim + rails matched to live + walk-forward + trend/reversal regime
split. Loosening the cap = more sector concentration = more momentum-crash risk,
so a win must show in RISK-ADJUSTED return and NOT blow up the reversal half.

VAL only. Reuses an existing wf cache (RAW default; demean via SMA_REVAL_CACHE).
Isolated on the DB copy.
"""

from __future__ import annotations

import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

SCRATCH = Path(
    "/private/tmp/claude-501/-Users-youruser-SecondBrain/"
    "f0fae5bb-6213-478f-b88e-e8747330c663/scratchpad"
)
DB = SCRATCH / "sma-eval.duckdb"
CACHE = Path(os.environ.get("SMA_REVAL_CACHE", str(Path.home() / ".sma-wf-models")))
LABEL = os.environ.get("SMA_REVAL_LABEL", "RAW")
WORKERS = int(os.environ.get("SMA_REVAL_WORKERS", "2"))

TREND = {"2025-07", "2025-08"}
REVERSAL = {"2025-09", "2025-10", "2025-11"}
LIVE = dict(k=15, hold_rank=30, sector_neutralize=1.0, min_hold=7)
CAPS = [0.35, 0.45, 0.55, 0.65, 1.00]  # 0.35 = current live; 1.00 = no cap


def run_one(cap):
    from sma.backtest.__main__ import _get_universe
    from sma.backtest.strategies.xgb_top_k import XGBoostTopKStrategy
    from sma.eval import evaluate_strategy
    from sma.model.predictor import Predictor
    from sma.risk.rails import RiskRails

    u = _get_universe()
    s = XGBoostTopKStrategy(
        predictor=Predictor(models_dir=CACHE, db_path=DB),
        universe=u,
        k=LIVE["k"],
        hold_rank=LIVE["hold_rank"],
        sector_neutralize=LIVE["sector_neutralize"],
        target_weight_per_position=0.10,
    )
    rails = RiskRails(
        stop_loss_pct=0.0,
        cash_floor_pct=0.05,
        max_sector_pct=cap,
        max_position_pct=0.10,
        max_drawdown_pct=0.15,
        min_hold_days=LIVE["min_hold"],
    )
    r = evaluate_strategy(
        strategy=s, window="val", universe=u, rails=rails,
        membership="current", db_path=DB,
    )
    return {
        "cap": cap, "sharpe": r.sharpe, "ret": r.total_return, "maxdd": r.max_drawdown,
        "hit": r.hit_rate, "trades": r.num_trades, "hold": r.avg_holding_days,
        "mret": list(r.monthly_returns), "mper": list(r.monthly_periods),
    }


def regime(res, months):
    rets = [m for p, m in zip(res["mper"], res["mret"], strict=True) if p in months]
    if not rets:
        return float("nan"), float("nan")
    total = 1.0
    for m in rets:
        total *= 1.0 + m
    total -= 1.0
    if len(rets) > 1:
        mean = sum(rets) / len(rets)
        sd = math.sqrt(sum((x - mean) ** 2 for x in rets) / (len(rets) - 1))
        shp = (mean / sd * math.sqrt(12)) if sd > 0 else float("nan")
    else:
        shp = float("nan")
    return total, shp


def main():
    print(f"=== {LABEL} sector-cap sweep (cache {CACHE}, {WORKERS} workers) ===", flush=True)
    print("live config fixed (k15 hr30 sn1.0 mh7); only max_sector_pct varies\n", flush=True)
    results = {}
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(run_one, c): c for c in CAPS}
        for fut in as_completed(futs):
            c = futs[fut]
            try:
                results[c] = fut.result()
                print(f"  done cap={c}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"  FAIL cap={c}: {e}", flush=True)

    print(
        f"\n{'sector_cap':12} {'sharpe':>7} {'ret':>8} {'maxDD':>8} {'hit':>6} "
        f"{'trd':>5} {'hold':>5}  | {'TREND r/s':>14}  {'REVERSAL r/s':>16}",
        flush=True,
    )
    print("-" * 116, flush=True)
    for c in CAPS:
        r = results.get(c)
        if not r:
            print(f"{c:<12}  (failed)", flush=True)
            continue
        tr_ret, tr_shp = regime(r, TREND)
        rv_ret, rv_shp = regime(r, REVERSAL)
        tag = "  <- LIVE" if c == 0.35 else ""
        print(
            f"{c:<12} {r['sharpe']:7.3f} {r['ret']:8.2%} {r['maxdd']:8.2%} "
            f"{r['hit']:6.1%} {r['trades']:5d} {r['hold']:5.1f}  | "
            f"{tr_ret:6.2%}/{tr_shp:5.2f}  {rv_ret:7.2%}/{rv_shp:6.2f}{tag}",
            flush=True,
        )


if __name__ == "__main__":
    main()
