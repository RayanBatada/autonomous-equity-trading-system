"""News tab: filterable article table."""

from datetime import date, timedelta

import streamlit as st

from dashboard import data


def render() -> None:
    st.header("News")

    all_tickers = data.news_tickers()
    all_sources = data.distinct_news_sources()

    col1, col2 = st.columns(2)
    with col1:
        selected_tickers = st.multiselect(
            "Tickers (leave blank for all)", all_tickers, default=[]
        )
    with col2:
        selected_sources = st.multiselect(
            "Sources (leave blank for all)", all_sources, default=[]
        )

    today = date.today()
    col3, col4 = st.columns(2)
    with col3:
        date_start = st.date_input("From", value=today - timedelta(days=7))
    with col4:
        date_end = st.date_input("To", value=today)

    if date_start > date_end:
        st.error("Start date must be before end date.")
        return

    tickers_filter = selected_tickers if selected_tickers else None
    sources_filter = selected_sources if selected_sources else None

    df, total = data.news_filtered(
        tickers=tickers_filter,
        sources=sources_filter,
        date_start=str(date_start),
        date_end=str(date_end),
        limit=200,
    )

    st.caption(f"{total} total matches (showing up to 200)")

    if df.empty:
        st.info("No articles match the current filters.")
        return

    # Render as a table with headline as link
    for _, row in df.iterrows():
        pub = str(row["published_at"])[:10]
        ticker = row["ticker"]
        src = row["source"]
        headline = row["headline"]
        url = row["url"]
        st.markdown(f"**{pub}** | {ticker} | {src} | [{headline}]({url})")
