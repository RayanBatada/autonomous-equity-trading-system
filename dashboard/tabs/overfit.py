"""Overfit tab: train vs val comparison with five overfit checks."""

from pathlib import Path

import streamlit as st

from sma.backtest.overfit import detect_overfit
from sma.backtest.strategies.buy_and_hold_spy import BuyAndHoldSPYStrategy
from sma.backtest.strategies.equal_weight import EqualWeightStrategy
from sma.backtest.strategies.random_long import RandomLongStrategy
from sma.backtest.strategies.xgb_top_k import XGBoostTopKStrategy
from sma.eval import evaluate_strategy
from sma.ingest.universe import load_universe
from sma.model.predictor import Predictor

_UNIVERSE_PATH = Path("src/sma/universe.yaml")
_MODELS_DIR = Path("models_artifacts")
_DB_PATH = Path("data/sma.duckdb")

STRATEGY_LABELS = {
    "buy_and_hold_spy": "Buy and Hold SPY",
    "equal_weight": "Equal Weight",
    "random_long": "Random Long",
    "xgb_top_k": "XGBoost Top-20",
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


def render() -> None:
    st.header("Overfit detector")
    st.caption("Runs both train and val windows to evaluate strategy generalization.")

    col1, col2 = st.columns(2)
    with col1:
        strategy_key = st.selectbox(
            "Strategy",
            list(STRATEGY_LABELS.keys()),
            format_func=lambda k: STRATEGY_LABELS[k],
            key="overfit_strategy",
        )
    with col2:
        seed = st.number_input("Seed", value=0, min_value=0, step=1, key="overfit_seed")

    if not st.button("Run overfit detector"):
        st.caption("Select a strategy and click Run.")
        return

    if not _UNIVERSE_PATH.exists():
        st.error(f"Universe file not found at {_UNIVERSE_PATH}.")
        return

    universe = load_universe(_UNIVERSE_PATH)

    with st.spinner("Running train + val backtests (this may take 30-60 seconds)..."):
        try:
            train_strat = _build_strategy(strategy_key, universe, int(seed))
            train_result = evaluate_strategy(
                membership="current",  # display tab: current-universe accepted
                strategy=train_strat,
                window="train",
                universe=universe,
                seed=int(seed) if strategy_key == "random_long" else None,
            )

            val_strat = _build_strategy(strategy_key, universe, int(seed))
            val_result = evaluate_strategy(
                membership="current",  # display tab: current-universe accepted
                strategy=val_strat,
                window="val",
                universe=universe,
                seed=int(seed) if strategy_key == "random_long" else None,
            )
        except Exception as exc:
            st.error(f"Overfit detection failed: {exc}")
            return

    report = detect_overfit(train_result, val_result)

    # Aggregate verdict banner
    if report.passed:
        st.success("No overfit flags detected.")
    else:
        n_flags = sum(1 for c in report.checks if not c.passed)
        st.warning(f"{n_flags} overfit flag(s) detected.")

    # Side-by-side metric comparison
    st.subheader("Train vs Val")
    col_t, col_v = st.columns(2)

    with col_t:
        st.markdown("**Train**")
        st.metric("Sharpe", f"{train_result.sharpe:.3f}")
        st.metric("Total Return", f"{train_result.total_return:.1%}")
        st.metric("Num Trades", str(train_result.num_trades))

    with col_v:
        st.markdown("**Val**")
        st.metric("Sharpe", f"{val_result.sharpe:.3f}")
        st.metric("Total Return", f"{val_result.total_return:.1%}")
        st.metric("Num Trades", str(val_result.num_trades))

    # Per-check results table
    st.subheader("Check results")
    check_rows = []
    for c in report.checks:
        check_rows.append(
            {
                "Check": c.name,
                "Status": "PASS" if c.passed else "FLAG",
                "Detail": c.detail,
            }
        )

    import pandas as pd
    check_df = pd.DataFrame(check_rows)
    st.dataframe(check_df, width="stretch", hide_index=True)
