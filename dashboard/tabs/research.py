"""Research tab: on-demand ticker analysis from the dashboard.

Wraps `sma.research.report.build_report` so the user can punch in a ticker
and see the same markdown report they'd get from the CLI, without leaving
the browser. No --refresh button: a paid LLM call shouldn't be one click
away. For a fresh thesis, run `python -m sma.research ticker NVDA --refresh`
from the terminal.
"""

from datetime import date as date_cls
from pathlib import Path

import streamlit as st

from dashboard.data import DB_PATH
from sma.ingest.store import Store
from sma.ingest.universe import load_universe
from sma.research.report import build_report
from sma.sectors import sector_for


def render() -> None:
    st.title("Ticker research")
    st.markdown(
        "Type a ticker (in-universe or otherwise) and get a unified report: "
        "latest XGBoost prediction, 30d price context, recent news, latest "
        "LLM thesis, recent politician disclosures. **Read-only** — no API "
        "calls are made from here. For a fresh LLM thesis, run "
        "`python -m sma.research ticker <TICKER> --refresh` from the "
        "terminal (paid Sonnet call)."
    )

    # Use st.form so DB queries don't fire on every keystroke — only on
    # explicit submit.
    with st.form("research_form", clear_on_submit=False):
        col_a, col_b = st.columns([2, 1])
        with col_a:
            raw_ticker = st.text_input(
                "Ticker", placeholder="e.g. NVDA", key="research_ticker"
            )
        with col_b:
            asof = st.date_input(
                "As of", value=date_cls.today(), key="research_asof"
            )
        submitted = st.form_submit_button("Generate report")

    if not submitted or not raw_ticker:
        st.info("Enter a ticker symbol and click **Generate report**.")
        return

    ticker = raw_ticker.strip().upper()

    universe_path = Path("src/sma/universe.yaml")
    if not universe_path.exists():
        st.error(f"Universe file not found at {universe_path}.")
        return

    universe_list = load_universe(universe_path)

    if not DB_PATH.exists():
        st.error(f"DuckDB not found at {DB_PATH}.")
        return

    store = Store(path=DB_PATH).connect(read_only=True)
    try:
        report = build_report(
            store=store,
            ticker=ticker,
            asof=asof,
            universe=universe_list,
            sector=sector_for(ticker),
            position=None,  # dashboard intentionally doesn't hit Alpaca
        )
    finally:
        store.close()

    st.markdown(report.to_markdown())
