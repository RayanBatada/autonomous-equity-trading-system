"""Streamlit entry point for the SMA dashboard."""

import sys
from pathlib import Path

# Streamlit puts dashboard/ on sys.path, not the project root. Add the root
# so `from dashboard.tabs import ...` and `from dashboard import data` resolve.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st  # noqa: E402

from dashboard.tabs import (  # noqa: E402
    autoresearch,
    backtest,
    coverage,
    features,
    model,
    news,
    overfit,
    paper,
    pipeline,
    politicians,
    prices,
    quality,
    research,
    roadmap,
    system,
    theses,
)

st.set_page_config(page_title="SMA dashboard", layout="wide")
st.title("Stock Market Predictor Agents")

tabs = st.tabs(
    ["Roadmap", "System", "Pipeline", "Research", "Autoresearch", "Coverage",
     "Prices", "News", "Politicians", "Features", "Backtest", "Overfit",
     "Model", "Quality", "Theses", "Paper"]
)
with tabs[0]:
    roadmap.render()
with tabs[1]:
    system.render()
with tabs[2]:
    pipeline.render()
with tabs[3]:
    research.render()
with tabs[4]:
    autoresearch.render()
with tabs[5]:
    coverage.render()
with tabs[6]:
    prices.render()
with tabs[7]:
    news.render()
with tabs[8]:
    politicians.render()
with tabs[9]:
    features.render()
with tabs[10]:
    backtest.render()
with tabs[11]:
    overfit.render()
with tabs[12]:
    model.render()
with tabs[13]:
    quality.render()
with tabs[14]:
    theses.render()
with tabs[15]:
    paper.render()
