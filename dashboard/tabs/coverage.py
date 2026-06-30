"""Coverage tab: data inventory and ingest run history."""

import plotly.express as px
import streamlit as st

from dashboard import data


def render() -> None:
    st.header("Data inventory")

    summary = data.coverage_summary()
    st.dataframe(summary, width="stretch", hide_index=True)

    st.subheader("Recent ingest runs")

    chart_df = data.ingest_log_for_chart(limit=30)

    if chart_df.empty:
        st.info("No ingest runs found.")
    else:
        chart_df["started_at"] = chart_df["started_at"].astype(str).str[:19]

        # Color mapping: green = ok, yellow = rate_limited, red = failed
        color_map = {"ok": "#2ca02c", "rate_limited": "#ff7f0e", "failed": "#d62728"}

        fig = px.bar(
            chart_df,
            x="started_at",
            y="rows_inserted",
            color="status",
            color_discrete_map=color_map,
            barmode="group",
            labels={
                "started_at": "Started at",
                "rows_inserted": "Rows inserted",
                "source": "Source",
                "status": "Status",
            },
            hover_data=["source"],
        )
        fig.update_layout(xaxis_tickangle=-45, height=350)
        st.plotly_chart(fig, width="stretch")

    st.subheader("Last 20 ingest log entries")
    log = data.ingest_log(limit=20)
    st.dataframe(log, width="stretch", hide_index=True)
