"""Prices tab: OHLCV candlestick chart and adj_close comparison."""


import plotly.graph_objects as go
import streamlit as st

from dashboard import data


def render() -> None:
    tickers = data.list_tickers()
    if not tickers:
        st.warning("No tickers found in prices table.")
        return

    col1, col2 = st.columns([2, 1])
    with col1:
        default_idx = tickers.index("SPY") if "SPY" in tickers else 0
        ticker = st.selectbox("Ticker", tickers, index=default_idx)
    with col2:
        source = st.selectbox("Source", ["yfinance", "alpaca"], index=0)

    df = data.ohlcv(ticker, source)

    if df.empty:
        st.warning(f"No data for {ticker} / {source}.")
        return

    min_date = df["date"].min().date()
    max_date = df["date"].max().date()

    col3, col4 = st.columns(2)
    with col3:
        date_start = st.date_input("From", value=min_date, min_value=min_date, max_value=max_date)
    with col4:
        date_end = st.date_input("To", value=max_date, min_value=min_date, max_value=max_date)

    if date_start > date_end:
        st.error("Start date must be before end date.")
        return

    mask = (df["date"].dt.date >= date_start) & (df["date"].dt.date <= date_end)
    filtered = df[mask]

    if filtered.empty:
        st.warning("No data for selected date range.")
        return

    # Candlestick chart
    fig_candle = go.Figure(
        data=[
            go.Candlestick(
                x=filtered["date"],
                open=filtered["open"],
                high=filtered["high"],
                low=filtered["low"],
                close=filtered["close"],
                name=ticker,
            )
        ]
    )
    fig_candle.update_layout(
        title=f"{ticker} OHLC ({source})",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        xaxis_rangeslider_visible=False,
        height=400,
    )
    st.plotly_chart(fig_candle, width="stretch")

    # Close vs adj_close overlay
    fig_adj = go.Figure()
    fig_adj.add_trace(
        go.Scatter(x=filtered["date"], y=filtered["close"], name="Close", line={"color": "#1f77b4"})
    )
    fig_adj.add_trace(
        go.Scatter(
            x=filtered["date"], y=filtered["adj_close"],
            name="Adj Close", line={"color": "#ff7f0e", "dash": "dot"},
        )
    )
    fig_adj.update_layout(
        title=f"{ticker} Close vs Adj Close",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        height=300,
    )
    st.plotly_chart(fig_adj, width="stretch")

    # Last 30 rows table
    st.subheader("Last 30 rows")
    cols = ["date", "open", "high", "low", "close", "adj_close", "volume"]
    last30 = filtered.tail(30)[cols].copy()
    last30["date"] = last30["date"].dt.strftime("%Y-%m-%d")
    st.dataframe(last30, width="stretch", hide_index=True)

    # News for this ticker
    st.subheader(f"News for {ticker}")
    news_df = data.news_for_ticker(ticker, limit=20)
    if news_df.empty:
        st.info(f"No news articles found for {ticker}.")
    else:
        # Render headlines as links
        for _, row in news_df.iterrows():
            pub = str(row["published_at"])[:10]
            src = row["source"]
            headline = row["headline"]
            url = row["url"]
            st.markdown(f"**{pub}** | {src} | [{headline}]({url})")
