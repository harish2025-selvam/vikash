import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(page_title="Dashboard — Energy AI", layout="wide", initial_sidebar_state="expanded")

from components.ui import inject_css, page_header, sidebar_info
from components.charts import line_chart, histogram_chart, submetering_area_chart
from ml.preprocessing import load_data
from ml.model import train_all_models

inject_css()
sidebar_info()

if "df" not in st.session_state:
    st.session_state.df = load_data()
if "model_data" not in st.session_state:
    st.session_state.model_data = train_all_models(st.session_state.df)

df = st.session_state.df

page_header("Dashboard", "Dataset overview and energy usage statistics")

# Top metrics
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Data Samples", f"{len(df):,}")
col2.metric("Avg Active Power", f"{df['Global_active_power'].mean():.3f} kW")
col3.metric("Peak Active Power", f"{df['Global_active_power'].max():.3f} kW")
col4.metric("Avg Voltage", f"{df['Voltage'].mean():.1f} V")
col5.metric("Avg Reactive Power", f"{df['Global_reactive_power'].mean():.3f} kVAR")

st.markdown("<br>", unsafe_allow_html=True)

# Line chart of usage (sample for speed)
sample = df["Global_active_power"].sample(min(2000, len(df)), random_state=42).sort_index().reset_index(drop=True)
fig_line = line_chart(
    x=list(range(len(sample))),
    y=sample.tolist(),
    title="Global Active Power — Sample of 2,000 Data Points",
    xlabel="Sample Index",
    ylabel="Active Power (kW)",
)
st.plotly_chart(fig_line, use_container_width=True)

# Two charts side by side
col_a, col_b = st.columns(2)
with col_a:
    fig_hist = histogram_chart(df["Global_active_power"], "Active Power Distribution", "#00D4FF")
    st.plotly_chart(fig_hist, use_container_width=True)
with col_b:
    fig_volt = histogram_chart(df["Voltage"], "Voltage Distribution", "#7B2FBE")
    st.plotly_chart(fig_volt, use_container_width=True)

# Sub-metering area
fig_sub = submetering_area_chart(df)
st.plotly_chart(fig_sub, use_container_width=True)

# Dataset preview
with st.expander("Dataset Preview (first 100 rows)"):
    st.dataframe(df.head(100), use_container_width=True)
