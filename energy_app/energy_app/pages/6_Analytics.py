import streamlit as st
import plotly.graph_objects as go
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(page_title="Analytics — Energy AI", layout="wide", initial_sidebar_state="expanded")

from components.ui import inject_css, page_header, sidebar_info
from components.charts import histogram_chart, line_chart, submetering_area_chart, CHART_LAYOUT
from ml.preprocessing import load_data
from ml.model import train_all_models

inject_css()
sidebar_info()

if "df" not in st.session_state:
    st.session_state.df = load_data()
if "model_data" not in st.session_state:
    st.session_state.model_data = train_all_models(st.session_state.df)

df = st.session_state.df

page_header("Analytics", "Trends, distributions, and sub-metering breakdown")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Trends", "Distributions", "Sub-metering", "Correlations"])

with tab1:
    st.markdown("<br>", unsafe_allow_html=True)
    sample_size = st.slider("Sample size for trend chart", 500, 5000, 2000, 500)
    sample = df["Global_active_power"].sample(sample_size, random_state=42).sort_index().reset_index(drop=True)

    fig = line_chart(
        x=list(range(len(sample))),
        y=sample.tolist(),
        title=f"Active Power Trend — {sample_size:,} Sample Points",
        xlabel="Sample Index",
        ylabel="kW",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Rolling average
    rolling = sample.rolling(window=50).mean()
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=list(range(len(sample))), y=sample, mode="lines",
                              line=dict(color="rgba(0,212,255,0.3)", width=1), name="Raw"))
    fig2.add_trace(go.Scatter(x=list(range(len(rolling))), y=rolling, mode="lines",
                              line=dict(color="#FF6B35", width=2), name="Rolling Avg (50)"))
    fig2.update_layout(**CHART_LAYOUT, title=dict(text="Active Power with Rolling Average",
                       font=dict(family="Syne, sans-serif", size=14)),
                       legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#CDD6E0")))
    st.plotly_chart(fig2, use_container_width=True)

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        fig = histogram_chart(df["Global_active_power"], "Active Power Distribution", "#00D4FF")
        st.plotly_chart(fig, use_container_width=True)

        fig = histogram_chart(df["Global_reactive_power"], "Reactive Power Distribution", "#7B2FBE")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = histogram_chart(df["Voltage"], "Voltage Distribution", "#FF6B35")
        st.plotly_chart(fig, use_container_width=True)

        fig = histogram_chart(df["Global_intensity"] if "Global_intensity" in df.columns else df["Sub_metering_3"],
                              "Intensity / Sub-metering 3 Distribution", "#10B981")
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    fig_sub = submetering_area_chart(df)
    st.plotly_chart(fig_sub, use_container_width=True)

    # Sub-metering bar totals
    sub_means = {
        "Sub_metering_1 (Kitchen)": df["Sub_metering_1"].mean(),
        "Sub_metering_2 (Heating/Laundry)": df["Sub_metering_2"].mean(),
        "Sub_metering_3 (Cooling/Water)": df["Sub_metering_3"].mean(),
    }
    fig_bar = go.Figure(go.Bar(
        x=list(sub_means.keys()),
        y=list(sub_means.values()),
        marker_color=["#00D4FF", "#7B2FBE", "#FF6B35"],
        marker_line_color="rgba(255,255,255,0.1)",
        marker_line_width=1,
    ))
    fig_bar.update_layout(**CHART_LAYOUT,
                          title=dict(text="Average Sub-metering Consumption (Wh)",
                                     font=dict(family="Syne, sans-serif", size=14)))
    st.plotly_chart(fig_bar, use_container_width=True)

    # Stats table
    st.markdown("""
    <div style="font-family:'Syne',sans-serif;font-weight:700;font-size:0.9rem;
        color:#CDD6E0;margin:1rem 0 0.5rem;">Sub-metering Statistics</div>
    """, unsafe_allow_html=True)
    sub_stats = df[["Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]].describe().round(4)
    st.dataframe(sub_stats, use_container_width=True)

with tab4:
    # Correlation heatmap
    num_cols = ["Global_active_power", "Global_reactive_power", "Voltage",
                "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]
    corr = df[num_cols].corr().round(3)

    fig_heat = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale=[[0, "#0D1117"], [0.5, "#7B2FBE"], [1, "#00D4FF"]],
        text=corr.values.round(2),
        texttemplate="%{text}",
        textfont=dict(size=10, color="#CDD6E0"),
        showscale=True,
    ))
    fig_heat.update_layout(
        **CHART_LAYOUT,
        title=dict(text="Feature Correlation Matrix", font=dict(family="Syne, sans-serif", size=14)),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # Descriptive stats
    st.markdown("""
    <div style="font-family:'Syne',sans-serif;font-weight:700;font-size:0.9rem;
        color:#CDD6E0;margin:1rem 0 0.5rem;">Full Descriptive Statistics</div>
    """, unsafe_allow_html=True)
    st.dataframe(df[num_cols].describe().round(4), use_container_width=True)
