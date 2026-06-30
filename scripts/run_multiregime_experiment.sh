#!/bin/zsh
# Deferred: backfill 2018-2022 history, then train a multi-regime eval model
# and measure if it reduces the reversal-regime IC inversion. Runs after the
# live pipeline (post-20:45) to avoid writer-lock contention.
cd /Users/youruser/code/Stock-Market-Predictor-Agents
echo "=== backfill 2018-2022 ($(date)) ==="
uv run python -u scripts/backfill_history_2018.py 2>&1 | grep -vE "Warning|warn" | tail -3
echo "=== train multi-regime eval model (TRAIN_START via env) ==="
SMA_TRAIN_START=2018-01-01 uv run python -m sma.model train --asof 2025-06-30 --no-cv --demean-labels --models-dir /tmp/sma-eval-models-multiregime 2>&1 | grep -iE "saved|deployed|rows" | tail -2
echo "=== regime-split IC: multi-regime vs 2023-only ==="
uv run python -u scripts/measure_ic.py /tmp/sma-eval-models-multiregime val 2>&1 | grep -vE "Warning|warn|INFO" | tail -4
echo "=== de-risk backtest (does drawdown-scaling help?) ==="
uv run python -u scripts/derisk_backtest.py 2>&1 | grep -vE "Warning|warn|INFO"
echo "=== DONE ($(date)) ==="
