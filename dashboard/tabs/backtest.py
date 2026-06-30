"""Backtest tab: run a baseline strategy and visualise results."""

from collections import defaultdict
from pathlib import Path

import pandas as pd
import plotly.express as pex
import plotly.graph_objects as go
import streamlit as st

from sma.backtest.strategies.buy_and_hold_spy import BuyAndHoldSPYStrategy
from sma.backtest.strategies.equal_weight import EqualWeightStrategy
from sma.backtest.strategies.random_long import RandomLongStrategy
from sma.backtest.strategies.xgb_top_k import XGBoostTopKStrategy
from sma.eval import evaluate_strategy
from sma.ingest.universe import load_universe
from sma.model.persistence import latest_model_for_date
from sma.model.predictor import Predictor

_UNIVERSE_PATH = Path("src/sma/universe.yaml")
_MODELS_DIR = Path("models_artifacts")
_DB_PATH = Path("data/sma.duckdb")

STRATEGY_ORDER = ["buy_and_hold_spy", "equal_weight", "random_long", "xgb_top_k"]

STRATEGY_LABELS = {
    "buy_and_hold_spy": "Buy and Hold SPY",
    "equal_weight": "Equal Weight",
    "random_long": "Random Long",
    "xgb_top_k": "XGBoost Top-20",
}

# Distinct colors for comparison equity curves
_COMPARISON_COLORS = {
    "buy_and_hold_spy": "#1f77b4",
    "equal_weight": "#ff7f0e",
    "random_long": "#9467bd",
    "xgb_top_k": "#2ca02c",
}


def _build_strategy(name: str, universe: list[str], seed: int):
    if name == "buy_and_hold_spy":
        return BuyAndHoldSPYStrategy()
    if name == "equal_weight":
        return EqualWeightStrategy(universe=universe)
    if name == "random_long":
        return RandomLongStrategy(universe=universe, seed=seed)
    if name == "xgb_top_k":
        predictor = Predictor(models_dir=_MODELS_DIR, db_path=_DB_PATH)
        return XGBoostTopKStrategy(predictor=predictor, universe=universe)
    raise ValueError(f"Unknown strategy: {name}")


def _today_for_check():
    from datetime import date
    return date.today()


def _final_holdings(trades: list[dict]) -> dict[str, float]:
    """Walk the trade log and return tickers still held at the end."""
    pos: dict[str, float] = defaultdict(float)
    for t in trades:
        if t["action"] == "buy":
            pos[t["ticker"]] += t["shares"]
        else:
            pos[t["ticker"]] -= t["shares"]
    return {k: v for k, v in pos.items() if v > 0}


def _render_single(strategy_key: str, window: str, seed: int, universe: list[str]) -> None:
    """Run and display a single-strategy backtest."""
    if strategy_key == "xgb_top_k":
        try:
            latest_model_for_date(_MODELS_DIR, _today_for_check())
        except FileNotFoundError:
            st.warning(
                "No trained XGBoost model found. Run "
                "`uv run python -m sma.model train --asof YYYY-MM-DD` first, "
                "then `backfill-predictions` for the window you want to test."
            )
            return

    with st.spinner("Running backtest..."):
        try:
            strategy = _build_strategy(strategy_key, universe, int(seed))
            result = evaluate_strategy(
                strategy=strategy,
                window=window,
                universe=universe,
                seed=int(seed) if strategy_key == "random_long" else None,
            )
        except Exception as exc:
            st.error(f"Backtest failed: {exc}")
            return

    # Metrics row
    st.subheader("Metrics")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Sharpe", f"{result.sharpe:.3f}")
    m2.metric("Sortino", f"{result.sortino:.3f}")
    m3.metric("Calmar", f"{result.calmar:.3f}")
    m4.metric("Max Drawdown", f"{result.max_drawdown:.1%}")

    m5, m6, m7, m8, m9 = st.columns(5)
    m5.metric("Total Return", f"{result.total_return:.1%}")
    m6.metric("Ann. Return", f"{result.annualized_return:.1%}")
    m7.metric("Hit Rate", f"{result.hit_rate:.1%}")
    m8.metric("Num Trades", str(result.num_trades))
    m9.metric("Avg Hold (days)", f"{result.avg_holding_days:.1f}")

    # Equity curve from daily_returns
    if result.daily_returns:
        st.subheader("Equity curve")
        cumulative = []
        equity = 1.0
        for r in result.daily_returns:
            equity *= (1.0 + r)
            cumulative.append(equity - 1.0)

        fig_eq = pex.line(
            x=list(range(len(cumulative))),
            y=cumulative,
            labels={"x": "Trading day", "y": "Cumulative return"},
        )
        fig_eq.update_layout(height=350)
        st.plotly_chart(fig_eq, width="stretch")

    # Monthly returns bar (calendar month labels on x-axis)
    if result.monthly_returns:
        st.subheader("Monthly returns")
        x_labels = result.monthly_periods or [f"M{i+1}" for i in range(len(result.monthly_returns))]
        fig_m = pex.bar(
            x=x_labels,
            y=result.monthly_returns,
            labels={"x": "Month", "y": "Return"},
            color=[r >= 0 for r in result.monthly_returns],
            color_discrete_map={True: "#2ca02c", False: "#d62728"},
        )
        fig_m.update_layout(showlegend=False, height=300, xaxis={"type": "category"})
        st.plotly_chart(fig_m, width="stretch")

    # Trade log + final holdings (computed from trades)
    if result.trades:
        st.subheader("Trades")
        trades_df = pd.DataFrame(result.trades)
        trades_df = trades_df[["date", "ticker", "action", "shares", "price", "value"]]
        trades_df["price"] = trades_df["price"].round(2)
        trades_df["value"] = trades_df["value"].round(2)

        held = _final_holdings(result.trades)
        if held:
            held_str = ", ".join(f"{t} ({s:g} sh)" for t, s in sorted(held.items()))
            st.caption(f"Held at end of window: {held_str}")
        else:
            st.caption("No open positions at end of window.")

        st.dataframe(trades_df, width="stretch", hide_index=True)
    else:
        st.info("No trades executed in this run.")

    # Provenance
    st.subheader("Provenance")
    st.markdown(
        f"Code commit: `{result.code_commit[:8]}` | "
        f"Data run_id: `{result.data_run_id}` | "
        f"Seed: `{result.seed}` | "
        f"Window: `{result.window}` | "
        f"Start: `{result.start_date}` | "
        f"End: `{result.end_date}`"
    )


def _render_comparison(window: str, seed: int, universe: list[str]) -> None:
    """Run all strategies and display a side-by-side comparison."""

    # Check whether xgb model exists; if not, skip it with a warning.
    xgb_available = True
    try:
        latest_model_for_date(_MODELS_DIR, _today_for_check())
    except FileNotFoundError:
        xgb_available = False
        st.warning(
            "No trained XGBoost model found. XGBoost Top-20 will be excluded from comparison. "
            "Run `uv run python -m sma.model train --asof YYYY-MM-DD` to add it."
        )

    strategies_to_run = [k for k in STRATEGY_ORDER if k != "xgb_top_k" or xgb_available]

    results = {}
    with st.spinner("Running all strategies..."):
        for key in strategies_to_run:
            try:
                strat = _build_strategy(key, universe, int(seed))
                result = evaluate_strategy(
                    strategy=strat,
                    window=window,
                    universe=universe,
                    seed=int(seed) if key == "random_long" else None,
                )
                results[key] = result
            except Exception as exc:
                st.warning(f"Strategy '{STRATEGY_LABELS[key]}' failed: {exc}")

    if not results:
        st.error("All strategies failed. Check logs for details.")
        return

    # Comparison metrics table
    st.subheader("Metrics comparison")
    rows = []
    for key in STRATEGY_ORDER:
        if key not in results:
            continue
        r = results[key]
        rows.append(
            {
                "Strategy": STRATEGY_LABELS[key],
                "Sharpe": round(r.sharpe, 3),
                "Sortino": round(r.sortino, 3),
                "Total Return": r.total_return,
                "Ann. Return": r.annualized_return,
                "Max Drawdown": r.max_drawdown,
                "Hit Rate": r.hit_rate,
                "Num Trades": r.num_trades,
            }
        )

    cmp_df = pd.DataFrame(rows)
    styled = cmp_df.style.format(
        {
            "Total Return": "{:.1%}",
            "Ann. Return": "{:.1%}",
            "Max Drawdown": "{:.1%}",
            "Hit Rate": "{:.1%}",
        }
    )
    st.dataframe(styled, width="stretch", hide_index=True)

    st.divider()

    # Overlaid equity curves
    st.subheader("Equity curves")
    fig = go.Figure()
    for key in STRATEGY_ORDER:
        if key not in results:
            continue
        r = results[key]
        if not r.daily_returns:
            continue
        cumulative = []
        equity = 1.0
        for ret in r.daily_returns:
            equity *= (1.0 + ret)
            cumulative.append(equity - 1.0)

        fig.add_trace(
            go.Scatter(
                x=list(range(len(cumulative))),
                y=cumulative,
                mode="lines",
                name=STRATEGY_LABELS[key],
                line={"color": _COMPARISON_COLORS[key]},
            )
        )

    fig.update_layout(
        height=420,
        xaxis_title="Trading day",
        yaxis_title="Cumulative return",
        legend={"orientation": "v", "x": 1.01, "y": 1},
        yaxis_tickformat=".0%",
    )
    st.plotly_chart(fig, width="stretch")


def render() -> None:
    st.header("Backtest")

    mode = st.radio(
        "Mode",
        ["Single strategy", "Compare all strategies"],
        horizontal=True,
        key="backtest_mode",
    )

    col1, col2, col3 = st.columns(3)

    if mode == "Single strategy":
        with col1:
            strategy_key = st.selectbox(
                "Strategy",
                STRATEGY_ORDER,
                format_func=lambda k: STRATEGY_LABELS[k],
            )
    with col2:
        window = st.selectbox("Window", ["train", "val"])
    with col3:
        seed = st.number_input("Seed", value=0, min_value=0, step=1)

    if not _UNIVERSE_PATH.exists():
        st.error(f"Universe file not found at {_UNIVERSE_PATH}.")
        return

    universe = load_universe(_UNIVERSE_PATH)

    if mode == "Single strategy":
        if not st.button("Run", key="backtest_run_single"):
            st.caption("Select a strategy and click Run.")
            return
        _render_single(strategy_key, window, int(seed), universe)
    else:
        if not st.button("Run comparison", key="backtest_run_compare"):
            st.caption("Click Run comparison to evaluate all strategies on the same window.")
            return
        _render_comparison(window, int(seed), universe)
