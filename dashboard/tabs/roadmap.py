"""Roadmap tab: at-a-glance project progress, models, and next steps.

Pulls live state from:
  - models_artifacts/*.json — model metadata
  - data/sma.duckdb (read-only) — predictions coverage
  - git log — recent commits

Plus static content (phase plan, model rationale, next steps) embedded
inline so the dashboard stays useful even without vault access.
"""

import json
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from dashboard.data import DB_PATH
from sma.db_connect import read_only_connect

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models_artifacts"

PHASE_PLAN: list[dict] = [
    {
        "phase": "0",
        "name": "Data ingestion + DuckDB storage",
        "status": "✅ shipped",
        "what": "7 sources (yfinance, alpaca, finnhub_news, alpaca_news, "
                "finnhub_fundamentals, newsapi, edgar) + quality checks + launchd cron",
        "key_metric": "81 tickers, 30 days news span, 100% coverage",
        "plain_english": (
            "**Pull the raw data we need to make any decision.** Every weekday at "
            "18:30 ET, the laptop wakes up, downloads the latest stock prices, "
            "company news headlines, earnings calendars, and SEC filings for our "
            "81-ticker watchlist, and writes everything to a single local database "
            "file (DuckDB). Without this layer, nothing else has anything to look "
            "at. Quality checks run after each download to catch obvious problems "
            "(a ticker with no price, news that didn't grow, etc.) so we don't "
            "make trades on broken data."
        ),
    },
    {
        "phase": "1",
        "name": "Backtest harness + baselines + overfit detector",
        "status": "✅ shipped",
        "what": "Single-scalar `evaluate_strategy()`, 3 baselines, 5 overfit checks, "
                "test-window access gated by promotion flag (autoresearch-compat)",
        "key_metric": "5/5 promotion gate items pass; SPY ground-truth within 0.7%",
        "plain_english": (
            "**A way to ask 'how would this strategy have done last year?' "
            "honestly.** The backtest harness simulates a strategy day-by-day "
                        "against historical prices, recording every trade and the resulting "
            "P&L. To make sure we don't fool ourselves, we built three trivial "
            "baseline strategies (buy-and-hold SPY, equal-weight everything, "
            "random picks) so we can ask 'is our smart strategy actually beating "
            "dumb ones?'. The overfit detector runs five sanity checks (is the "
            "test result implausibly close to validation? did we look at test "
            "data while tuning? etc.). The held-out test window is locked behind "
            "a flag so a tuner can't accidentally optimize against it."
        ),
    },
    {
        "phase": "1.5",
        "name": "Streamlit dashboard",
        "status": "✅ shipped",
        "what": "10 tabs: Coverage / Prices / News / Backtest / Overfit / Model / "
                "Quality / Theses / Paper / Roadmap (this tab)",
        "key_metric": "localhost:8765, persistent launchd daemon",
        "plain_english": (
            "**A web page that lets you see what's going on without writing SQL.** "
            "Each tab is a different lens: 'Coverage' shows what data we have, "
            "'Prices' draws charts, 'Backtest' lets you run a strategy against "
            "history with a click, 'Theses' shows what the LLM agents wrote, "
            "'Paper' shows what's been traded, and this Roadmap tab summarizes "
            "where the project stands. It runs as a always-on background "
            "service so you can open it any time without restarting anything."
        ),
    },
    {
        "phase": "2",
        "name": "Quant model (XGBoost)",
        "status": "✅ shipped",
        "what": "12 lookahead-clean features, walk-forward CV, weekly retrain. "
                "Strategy: xgb_top_k (k=20, 5%/position, equal-weight)",
        "key_metric": "Test Sharpe 0.225 → 1.076 with rails on, return 5.34%",
        "plain_english": (
            "**The actual prediction engine.** Each Friday night, the laptop "
            "trains an XGBoost model that takes 12 numbers about each stock "
            "(recent returns, volatility, momentum indicators, distance from "
            "52-week highs, etc.) and predicts the next 30-day return. Once a "
            "day, the model scores all 81 tickers; the strategy 'xgb_top_k' "
            "buys the 20 highest-scoring names, equal-weighted at 5% each. We "
            "use walk-forward training (no model ever sees future data) and "
            "hold out a separate test window, so what we see in backtests "
            "matches what we'd get in real life."
        ),
    },
    {
        "phase": "0.5",
        "name": "Data quality pass (Alpaca News + backfills + 2 quality checks)",
        "status": "✅ shipped",
        "what": "Tiingo→Alpaca pivot, news backfill 30d, earnings 8→248 rows, "
                "`news_per_ticker_minimum` + `earnings_coverage` checks",
        "key_metric": "5/5 quality gate clean (yfinance earnings closes Finnhub-quota gap)",
        "plain_english": (
            "**A retroactive cleanup of the data layer.** Phase 0 left some gaps: "
            "we'd planned to use Tiingo for news but it turned out to require a "
            "paid plan, so we swapped to Alpaca News (free with our existing "
            "key). We also backfilled 30 days of historical news so the LLM "
            "agents have something to read on day one. Earnings data went from "
            "8 records to 248 by adding a yfinance-based fallback to cover "
            "tickers Finnhub's free-tier quota was missing. Two new quality "
            "checks ('every ticker must have at least one news article' + "
            "'every ticker must have a recent earnings record') turn silent "
            "data gaps into loud failures."
        ),
    },
    {
        "phase": "3",
        "name": "Strategy + risk layer",
        "status": "✅ shipped",
        "what": "Stop-loss (disabled per diagnostic), cash floor 5%, real GICS sectors, "
                "earnings blackout 3 days, fixed avg_holding_days",
        "key_metric": "Test Sharpe 0.225→1.076, hit rate 32.7%→66.7%",
        "plain_english": (
            "**Hard guardrails the strategy can't override.** Even if the model "
            "screams 'BUY ALL THE NVDA', the risk layer says no: cash floor "
            "keeps at least 5% in cash, the earnings blackout refuses to open "
            "new positions in the 3 days before a company reports, the sector "
            "cap prevents more than 25% in any one industry. The stop-loss "
            "rail (sell anything down 8% from cost basis) was actually "
            "*disabled* in this phase after we discovered it was destroying "
            "returns by ejecting positions that subsequently recovered. "
            "Adding these guardrails almost 5x'd the test-window Sharpe and "
            "doubled the win rate (32% → 66%)."
        ),
    },
    {
        "phase": "4",
        "name": "LLM research agents (Claude Haiku 4.5)",
        "status": "✅ shipped",
        "what": "3-agent pipeline (researcher → analyst → strategist), "
                "weekly+event-triggered cadence, asymmetric C-rules in xgb_top_k, "
                "cost tracker + daily budget guardrail",
        "key_metric": "Test-window Sharpe +1.123 with theses vs +0.874 baseline (Δ +0.248)",
        "plain_english": (
            "**An LLM reading the news on top of the math.** A 3-stage pipeline: "
            "the 'researcher' agent reads each ticker's recent news and "
            "summarizes it; the 'analyst' agent takes that summary plus 30 "
            "days of price stats and writes a bull/bear case; the 'strategist' "
            "agent decides whether to override the model's signal. Asymmetric "
            "rules: a bearish thesis can VETO a buy the model wanted, a "
            "bullish thesis nudges the model's score up by 5%. Cost is "
            "carefully bounded: ~$1.50/month steady-state, with a hard daily "
            "spend ceiling. The system holds about a quarter Sharpe more with "
            "theses on than off, on real held-out data."
        ),
    },
    {
        "phase": "5",
        "name": "Paper trading via Alpaca",
        "status": "🔵 in progress (live submit path validated, awaiting first OPG fill)",
        "what": "Live module: orders.translate, alpaca_client, preflight, decide, "
                "stop_loss, reconcile, status CLI. 3 launchd jobs (decide/stop-loss/reconcile). "
                "Schema v4: intended_orders, paper_fills, account_snapshots",
        "key_metric": "534 tests passing; 8 codex findings + 4 dry-run bugs all fixed",
        "plain_english": (
            "**Doing it for real, but with fake money.** Alpaca offers a 'paper "
            "trading' account that simulates real submission against real "
            "market data — you place orders, they fill at real prices, P&L "
            "moves up and down — but no actual money changes hands ($100k "
            "starting balance, can be reset). Each weekday at 18:35 ET, the "
            "system runs the strategy + risk rails + cash floor, translates "
            "decisions into share counts, and submits orders to Alpaca that "
            "fill at the next morning's open auction. A reconcile job at "
            "16:30 ET pulls the actual fills back, snapshots the account, "
            "and detects drift between what we intended vs what happened. "
            "The goal: 2+ weeks of clean parity (paper P&L within 30bp/day "
            "of the simulator) before considering real money."
        ),
    },
    {
        "phase": "6",
        "name": "Auto-research loop (Karpathy autoresearch pattern)",
        "status": "📐 spec v2 locked, awaiting Phase 5 paper-trading proof",
        "what": "Sonnet rewrites `src/sma/strategy/active.py` (post-processor tilt). "
                "Walk-forward CV across 5 val sub-windows with monotonicity gate. "
                "Human-in-loop promotion path",
        "key_metric": "~$24/night Sonnet upper bound; promotion = mono≥3 + Δ≥0.1 Sharpe",
        "plain_english": (
            "**The thing that improves the strategy automatically while you "
            "sleep.** Pattern from Andrej Karpathy's autoresearch repo: run a "
            "tight loop where Claude Sonnet reads one specific code file "
            "(active.py), proposes an edit, the harness runs the new version "
            "against historical data, logs the result, then asks the LLM to "
            "try again. Hundreds of iterations per night. Crucial guardrail: "
            "trading strategies will overfit to validation data given enough "
            "tries, so we split the val window into 5 sub-windows and require "
            "an improvement to hold across at least 3 of them before "
            "promotion. The held-out test window stays locked. A human "
            "reviews any candidate before it goes to paper trading."
        ),
    },
    {
        "phase": "7",
        "name": "Real money, tiny size",
        "status": "⏸ not started",
        "what": "Requires paper-trading proof (realized Sharpe within 0.5 of "
                "backtest Sharpe, no risk rail tripped by a bug)",
        "key_metric": "TBD (≥2-week paper window required first)",
        "plain_english": (
            "**The thing this whole project is building toward.** Once the "
            "strategy has been paper-trading cleanly for at least two weeks "
            "(realized Sharpe within 0.5 of backtest, no risk rail tripped "
            "by a bug), we promote it to real capital — small size at first, "
            "scaled up only on continued evidence. Phase 7 is the only point "
            "where actual dollars are at risk; everything before it is "
            "validation. Not started yet."
        ),
    },
]

NEXT_STEPS: list[dict] = [
    {
        "step": "Tonight 19:05 ET",
        "what": "Auto-fire canary AAPL order. Backgrounded process at PID 25218 "
                "(caffeinate-d) sleeps until 19:05 then runs `decide --canary AAPL`. "
                "Result logs to /tmp/canary-fire-2026-04-29.log",
    },
    {
        "step": "Thu 2026-04-30 ~09:35 ET",
        "what": "Run reconcile to verify the AAPL fill in `paper_fills`. Cleanup "
                "the position via Alpaca dashboard.",
    },
    {
        "step": "After smoke succeeds",
        "what": "`bash scripts/install_launchd.sh` to load all 7 plists "
                "(ingest + agents + model retrain/predict + Phase 5 decide/stop-loss/reconcile)",
    },
    {
        "step": "5-day --dry-run period",
        "what": "Manual `--dry-run` each evening; eyeball the proposed orders. "
                "After 5 clean evenings, flip to auto-fire.",
    },
    {
        "step": "2-week paper window",
        "what": "Watch parity criterion: |paper - sim| < 30bp/day on ≥70% of days, "
                "<100bp cumulative. If parity holds, promote toward Phase 7.",
    },
    {
        "step": "Phase 6 implementation",
        "what": "After paper proves out, run `superpowers:writing-plans` against "
                "the locked Phase 6 spec → build `src/sma/strategy/active.py` + "
                "`src/sma/autoresearch/` + first overnight Sonnet smoke",
    },
]

MODEL_ARCHITECTURE = """
**Architecture: XGBoost regressor predicting 30-day forward return.**

- **12 features** (lookahead-clean, proven by adversarial test):
  - Returns: `ret_1d`, `ret_5d`, `ret_20d`, `ret_60d`
  - Vol: `vol_20d`, `vol_60d`
  - Momentum: `rsi_14`
  - Volume: `volume_z_20d`, `dollar_volume_20d`
  - Relative: `rel_strength_spy_60d`
  - Gap: `gap_open`
  - Distance from 52-week high: `dist_from_52w_high`
- **Training**: walk-forward CV with 6-combination hyperparameter grid
  (`max_depth ∈ {3, 5, 7}`, `learning_rate ∈ {0.05, 0.1}`), 5 folds.
  Picked: `max_depth=3, learning_rate=0.05, n_estimators=300`.
- **30-day purge gap** between training and validation: ensures the
  forward-return label window doesn't bleed into the eval set.
- **Strategy**: `XGBoostTopKStrategy` picks top 20 names by predicted return,
  equal-weighted at 5% each.

**Why this architecture?** Phase 2 spec rationale:
1. **Tabular features → XGBoost** is the well-established baseline for
   cross-sectional return prediction. Beats simple OLS by capturing
   non-linear factor interactions.
2. **Walk-forward CV** prevents lookahead bias that single train/val splits
   miss. Each model is trained only on data ≤ its train_end_date.
3. **Top-K equal-weight** is the simplest strategy that converts a return
   prediction into a portfolio. Doesn't over-fit on quirky position sizing
   that more complex schemes (mean-variance, Kelly) would introduce.
4. **Weekly retrain** balances staleness (longer = stale model) vs noise
   (daily retrain = high variance from small training windows).

**LLM theses (Phase 4) plug in as asymmetric C-rules** on top of the quant
score: bearish thesis vetoes a buy; bullish thesis tilts the score by ×1.05;
strong-bearish on a held position triggers a `thesis_exit` sell.
"""


def _load_model_metadata() -> pd.DataFrame:
    if not MODELS_DIR.exists():
        return pd.DataFrame()
    rows = []
    for json_path in sorted(MODELS_DIR.glob("*.json")):
        try:
            meta = json.loads(json_path.read_text())
        except Exception:
            continue
        rows.append({
            "model_id": json_path.stem,
            "train_end": meta.get("train_end_date", ""),
            "train_rows": meta.get("train_rows", 0),
            "rmse": meta.get("rmse_holdout"),
            "max_depth": meta.get("hyperparams", {}).get("max_depth"),
            "lr": meta.get("hyperparams", {}).get("learning_rate"),
            "n_est": meta.get("hyperparams", {}).get("n_estimators"),
        })
    return pd.DataFrame(rows)


def _recent_commits(n: int = 15) -> list[dict]:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(PROJECT_ROOT), "log", f"-n{n}",
             "--pretty=format:%h%x09%an%x09%ar%x09%s"],
            text=True, timeout=5,
        )
    except Exception as e:
        return [{"sha": "?", "author": "?", "rel": "?",
                 "message": f"git log failed: {e}"}]
    rows = []
    for line in out.splitlines():
        parts = line.split("\t", 3)
        if len(parts) == 4:
            rows.append({"sha": parts[0], "author": parts[1],
                         "rel": parts[2], "message": parts[3]})
    return rows


def _predictions_coverage() -> tuple[int, int, int, str | None]:
    if not DB_PATH.exists():
        return (0, 0, 0, None)
    con = read_only_connect(DB_PATH)
    try:
        tables = {t for (t,) in con.execute("SHOW TABLES").fetchall()}
        if "predictions" not in tables:
            return (0, 0, 0, None)
        row = con.execute("""
            SELECT COUNT(*) AS rows,
                   COUNT(DISTINCT asof_date) AS dates,
                   COUNT(DISTINCT model_id) AS models,
                   MAX(asof_date)::TEXT AS latest
            FROM predictions
        """).fetchone()
        return (row[0] or 0, row[1] or 0, row[2] or 0, row[3])
    finally:
        con.close()


def render() -> None:
    st.header("Roadmap")
    st.caption("At-a-glance project progress, models, and next steps. "
               "Pulls live from `models_artifacts/`, `git`, and DuckDB; "
               "static content (phase plan, model architecture, next steps) "
               "embedded inline.")

    # ── Phase plan ────────────────────────────────────────────
    st.subheader("Phase plan")
    st.caption("Quick technical view in the table; plain-English explainer for "
               "each phase below.")
    phase_df = pd.DataFrame(
        [{k: v for k, v in p.items() if k != "plain_english"} for p in PHASE_PLAN]
    )
    st.dataframe(
        phase_df, hide_index=True, width="stretch",
        column_config={
            "phase": st.column_config.TextColumn("Phase", width="small"),
            "name": st.column_config.TextColumn("Name", width="medium"),
            "status": st.column_config.TextColumn("Status", width="medium"),
            "what": st.column_config.TextColumn("What it does (technical)", width="large"),
            "key_metric": st.column_config.TextColumn("Key metric"),
        },
    )

    # ── Plain-English explainers ─────────────────────────────
    st.markdown("### Phase plan in plain English")
    st.caption("Same phases, written so a non-engineer can follow what each "
               "step actually accomplishes and why it matters.")
    for p in PHASE_PLAN:
        label = f"**Phase {p['phase']}** — {p['name']}  ·  {p['status']}"
        with st.expander(label, expanded=False):
            st.markdown(p["plain_english"])
            st.caption(f"**Technical:** {p['what']}")
            st.caption(f"**Key metric:** {p['key_metric']}")

    # ── Next steps ────────────────────────────────────────────
    st.subheader("Next steps")
    for i, step in enumerate(NEXT_STEPS, 1):
        st.markdown(f"**{i}. {step['step']}** — {step['what']}")

    # ── Models ────────────────────────────────────────────────
    st.subheader("Models")
    pred_rows, pred_dates, pred_models, pred_latest = _predictions_coverage()
    cols = st.columns(4)
    cols[0].metric("Model artifacts on disk",
                   len(list(MODELS_DIR.glob("*.pkl"))) if MODELS_DIR.exists() else 0)
    cols[1].metric("Predictions table rows", f"{pred_rows:,}")
    cols[2].metric("Distinct asof dates", pred_dates)
    cols[3].metric("Latest prediction", pred_latest or "—")

    with st.expander("Why this model? (architecture + rationale)", expanded=False):
        st.markdown(MODEL_ARCHITECTURE)

    st.markdown("**All weekly walk-forward models** (latest on top):")
    model_df = _load_model_metadata()
    if model_df.empty:
        st.info("No model metadata found in `models_artifacts/`.")
    else:
        model_df = model_df.sort_values("train_end", ascending=False)
        st.dataframe(model_df, hide_index=True, width="stretch")

    # ── Recent commits ────────────────────────────────────────
    st.subheader("Recent commits (`git log`)")
    commits = _recent_commits(15)
    commit_df = pd.DataFrame(commits)
    if not commit_df.empty:
        st.dataframe(commit_df, hide_index=True, width="stretch")

    # ── Spec docs ─────────────────────────────────────────────
    st.subheader("Specs and reviews (in vault)")
    st.markdown("""
The canonical spec docs live in the obsidian vault under
`1-Projects/Stock-Market-Predictor-Agents/specs/` and `references/`.

**Phase specs:**
- `2026-04-24-system-architecture.md` — full system architecture
- `2026-04-24-phase-0-data-ingestion.md` — Phase 0 spec
- `2026-04-25-phase-1-backtest-harness.md` — Phase 1 spec (autoresearch-compat)
- `2026-04-26-phase-0.5-data-quality-spec.md` — Phase 0.5 spec
- `2026-04-26-phase-2-quant-model.md` — Phase 2 spec
- `2026-04-26-phase-4-llm-research-agents.md` — Phase 4 spec
- `2026-04-28-phase-5-alpaca-paper-trading.md` — Phase 5 spec (v3)
- `2026-04-28-phase-6-autoresearch.md` — Phase 6 spec (v2, locked)

**Codex reviews:**
- `2026-04-28-codex-adversarial-review.md` — Phase 4 (caught the look-ahead leak)
- `2026-04-28-codex-phase-5-adversarial-review.md` — Phase 5 round 0 (4 HIGH)
- `2026-04-29-codex-fix-bundle-review.md` — Phase 5 rounds 1+2 (3 HIGH + 1 LOW)

**Reference:**
- `references/karpathy-autoresearch.md` — the Phase 6 pattern source

**Session handoff:**
- `where-we-left-off.md` — continuously-updated session log
""")
    st.caption(
        "Date last refreshed: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
