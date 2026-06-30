# Autonomous Equity Trading System

**A machine-learning equity trading system.** A fully-scheduled daily
pipeline that ingests market and alternative data, ranks a ~270-stock universe with a
gradient-boosted model on a 30-day forward horizon, layers LLM-agent research theses on
top, constructs a risk-controlled portfolio, executes through a broker API, and
reconciles fills the next day — with self-healing reliability and a rigorous,
look-ahead-free validation framework.

> Built solo over ~2 months — **~18,500 lines of Python, 1,177 automated tests, ~20
> subsystems** — trading a live paper account on a daily schedule.

---

## What it does — the daily cycle

| Stage | What happens |
|---|---|
| **Ingest** (18:30 ET) | Pull daily prices + alternative data from 7 sources into DuckDB; 6 SQL quality checks; self-healing backfill of any days missed during an outage |
| **Predict** (19:30 ET) | Score every stock with the deployed XGBoost model — a cross-sectional ranking of expected 30-day-forward relative return |
| **Research agents** | A 3-agent LLM pipeline (researcher → analyst → strategist) writes per-stock theses that can veto a buy on a bearish view |
| **Decide** (20:00 ET) | Construct a target portfolio under risk rails (sector caps, cash floor, min-hold, earnings blackout) and submit broker orders |
| **Reconcile** (next day) | Verify fills vs intended orders, detect ledger drift, alert on partial fills / missed sells / large equity moves |

The model **retrains weekly behind a statistical deploy gate**, and an **autonomous
research loop** searches the config space — both gated on out-of-sample rank-IC.

## Architecture

- **`ingest/`** — 7 data sources, 6 quality checks, DuckDB single-writer store, self-healing gap backfill
- **`features/`** — 21 engineered features (momentum, volatility, RSI, distance-from-high, liquidity, earnings surprise, news attention, political flow)
- **`model/`** — XGBoost ranker, walk-forward cross-validation, cross-sectionally demeaned multi-regime labels, IC-gated promotion, versioned persistence
- **`strategy/` + `risk/`** — portfolio construction with sector caps, cash floor, min-hold, earnings blackout, stop-loss; one shared risk module for backtest/live parity
- **`agents/`** — 3-agent LLM thesis pipeline (Claude Haiku) with cost guardrails and a full audit trail
- **`live/`** — broker (Alpaca) execution, DB-mirrored order state, reconciliation, ledger-drift detection
- **`autoresearch/`** — deterministic config search with IC-gated auto-promotion
- **`backtest/`** — look-ahead-free simulator (decide D, fill D+1), overfit detector, metrics
- **`eval/`** — data-driven regime classifier, information-coefficient analysis
- **ops** — `launchd` scheduling, hourly watchdog, monitoring/alerting, daily backups, lineage gating, sentinels

## Machine-learning methodology

- **Cross-sectional ranking** of 30-day-forward relative returns (a factor/ranking problem, not price forecasting)
- **Walk-forward cross-validation** — performance measured only on data strictly after the training window (no look-ahead)
- **Information Coefficient (IC)** — rank correlation of predictions vs realized returns — as the bedrock metric instead of noisy P&L
- **Demeaned, multi-regime labels** trained back to 2018
- **Statistical deploy gate** — a model/config is promoted only if it beats the incumbent on held-out CV rank-IC by a fixed margin

## Validation & rigor

- An **8-pass adversarial code audit** of the full codebase (all 5 critical issues fixed)
- Discovered and corrected an **evaluation-harness contamination bug** that had invalidated earlier backtest numbers — then re-ran the analysis honestly
- A **multi-agent diagnostic** that decomposed the strategy's apparent outperformance and showed it was market **beta + variance**, not demonstrable alpha — an evidence-based decision to keep validating rather than over-deploy
- Rigorous experiments ruling out survivorship bias, regime-timing, and feature de-concentration as cheap wins
- **1,177 automated tests** gating every change

## Results (honest)

Trading a live paper account since 2026-04-30. Net positive vs the benchmark over the
window, but rigorous attribution shows the returns are driven by **market beta**, not
yet by demonstrable skill — the underlying signal is a real but thin momentum factor
(IC ~ 0.03-0.05). The system is judged on **IC accumulated over time, not short-run
P&L**. The headline is not a return number; it is having built both the trading system
*and* the validation framework honest enough to assess it correctly.

## Tech stack

Python · XGBoost · pandas · DuckDB · Alpaca API · Anthropic Claude (LLM agents) ·
`launchd` (scheduling) · pytest · uv

## Setup

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"
cp .env.example .env   # then fill in your API keys
```

## Disclaimer

Research and educational project. All trading shown is **paper trading**. Nothing here
is investment advice.
