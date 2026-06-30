"""Parallel strategy-config re-validation on the corrected simulator.

Runs a one-knob-at-a-time sweep AROUND the live config concurrently (each eval
is an independent ~5-min walk-forward run). Reports aggregate metrics + a
trend-vs-reversal regime split, because a single-window aggregate is regime-luck.

Model surface is selected by env:
  SMA_REVAL_CACHE  -- models_dir (default ~/.sma-wf-models, the RAW wf cache)
  SMA_REVAL_LABEL  -- a label for the printout (default "RAW")
  SMA_REVAL_WORKERS-- process pool size (default 4)

Corrected surface: parity-fixed sim (HEAD) + rails matched to live
(sector 0.35 / cash-floor 0.05 / pos 0.10 / no-stop / mh7) + walk-forward
boundary models. VAL ONLY -- TEST stays sacred. Isolated on a DB copy.
"""

from __future__ import annotations

import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

SCRATCH = Path(
    "/private/tmp/claude-501/-Users-rayanbatada-SecondBrain/"
    "f0fae5bb-6213-478f-b88e-e8747330c663/scratchpad"
)
DB = SCRATCH / "sma-eval.duckdb"
CACHE = Path(os.environ.get("SMA_REVAL_CACHE", str(Path.home() / ".sma-wf-models")))
LABEL = os.environ.get("SMA_REVAL_LABEL", "RAW")
WORKERS = int(os.environ.get("SMA_REVAL_WORKERS", "4"))

TREND = {"2025-07", "2025-08"}
REVERSAL = {"2025-09", "2025-10", "2025-11"}

LIVE = dict(k=15, hold_rank=30, sector_neutralize=1.0, min_hold=7)
SWEEP = [
    ("LIVE k15 hr30 sn1.0 mh7", dict()),
    ("k=10", dict(k=10)),
    ("k=12", dict(k=12)),
    ("k=20", dict(k=20)),
    ("hold_rank=20", dict(hold_rank=20)),
    ("hold_rank=40", dict(hold_rank=40)),
    ("sector_neutralize=0.5", dict(sector_neutralize=0.5)),
    ("sector_neutralize=0.0", dict(sector_neutralize=0.0)),
    ("min_hold=5", dict(min_hold=5)),
    ("min_hold=10", dict(min_hold=10)),
]


def run_one(item):
    """Worker: build strategy+rails from a config dict, run one val eval."""
    name, override = item
    from sma.backtest.__main__ import _get_universe
    from sma.backtest.strategies.xgb_top_k import XGBoostTopKStrategy
    from sma.eval import evaluate_strategy
    from sma.model.predictor import Predictor
    from sma.risk.rails import RiskRails

    cfg = {**LIVE, **override}
    u = _get_universe()
    s = XGBoostTopKStrategy(
        predictor=Predictor(models_dir=CACHE, db_path=DB),
        universe=u,
        k=cfg["k"],
        hold_rank=cfg["hold_rank"],
        sector_neutralize=cfg["sector_neutralize"],
        target_weight_per_position=0.10,
    )
    rails = RiskRails(
        stop_loss_pct=0.0,
        cash_floor_pct=0.05,
        max_sector_pct=0.35,
        max_position_pct=0.10,
        max_drawdown_pct=0.15,
        min_hold_days=cfg["min_hold"],
    )
    r = evaluate_strategy(
        strategy=s, window="val", universe=u, rails=rails,
        membership="current", db_path=DB,
    )
    return {
        "name": name,
        "sharpe": r.sharpe,
        "ret": r.total_return,
        "maxdd": r.max_drawdown,
        "hit": r.hit_rate,
        "trades": r.num_trades,
        "hold": r.avg_holding_days,
        "monthly_returns": list(r.monthly_returns),
        "monthly_periods": list(r.monthly_periods),
    }


def regime(res, months):
    rets = [
        m
        for p, m in zip(res["monthly_periods"], res["monthly_returns"], strict=True)
        if p in months
    ]
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
    print(f"=== {LABEL} walk-forward re-validation (cache {CACHE}, {WORKERS} workers) ===", flush=True)
    results = {}
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(run_one, item): item[0] for item in SWEEP}
        for fut in as_completed(futs):
            name = futs[fut]
            try:
                res = fut.result()
                results[name] = res
                print(f"  done: {name}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"  FAILED: {name}: {e}", flush=True)

    print(
        f"\n{'config':26} {'sharpe':>7} {'ret':>8} {'maxDD':>8} {'hit':>6} "
        f"{'trd':>5} {'hold':>5}  | {'TREND r/s':>14}  {'REVERSAL r/s':>16}",
        flush=True,
    )
    print("-" * 122, flush=True)
    for name, _ in SWEEP:
        r = results.get(name)
        if not r:
            print(f"{name:26}  (failed)", flush=True)
            continue
        tr_ret, tr_shp = regime(r, TREND)
        rv_ret, rv_shp = regime(r, REVERSAL)
        print(
            f"{name:26} {r['sharpe']:7.3f} {r['ret']:8.2%} {r['maxdd']:8.2%} "
            f"{r['hit']:6.1%} {r['trades']:5d} {r['hold']:5.1f}  | "
            f"{tr_ret:6.2%}/{tr_shp:5.2f}  {rv_ret:7.2%}/{rv_shp:6.2f}",
            flush=True,
        )

    live = results.get("LIVE k15 hr30 sn1.0 mh7")
    if live:
        print("\nLIVE config monthly returns:", flush=True)
        for p, m in zip(live["monthly_periods"], live["monthly_returns"], strict=True):
            tag = "trend" if p in TREND else "rev" if p in REVERSAL else "tail"
            print(f"  {p} [{tag:5}] {m:+.2%}", flush=True)


if __name__ == "__main__":
    main()
