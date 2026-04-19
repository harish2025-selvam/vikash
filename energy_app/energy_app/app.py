import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(
    page_title="Energy Savings AI — FedProx",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon=None,
)

from components.ui import inject_css, page_header, sidebar_info
from ml.preprocessing import load_data, preprocess
from ml.model import train_all_models
from config import NUM_CLIENTS, DATA_PATH

inject_css()
sidebar_info()

# ── Global state: load data & train once ──────────────────────────────────────
@st.cache_data(show_spinner=False)
def cached_load():
    return load_data()

@st.cache_resource(show_spinner=False)
def cached_train(data_hash):
    df = cached_load()
    return train_all_models(df)

if "df" not in st.session_state:
    with st.spinner("Loading dataset..."):
        st.session_state.df = cached_load()

if "model_data" not in st.session_state:
    with st.spinner("Training models (first load)..."):
        st.session_state.model_data = train_all_models(st.session_state.df)

# ── Home / Landing ─────────────────────────────────────────────────────────────
page_header(
    "Personalized Energy Saving Recommendation System",
    "Federated Learning · FedProx Optimized · Random Forest"
)

df = st.session_state.df
md = st.session_state.model_data

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Data Samples", f"{len(df):,}")
col2.metric("Federated Clients", f"{NUM_CLIENTS}")
col3.metric("Samples per Client", f"~{len(df)//NUM_CLIENTS:,}")
col4.metric("Best Model RMSE", f"{md['results'][md['best_name']]['RMSE']:.4f}")

st.markdown("<br>", unsafe_allow_html=True)

st.markdown("""
<div class="glass-card">
    <div style="font-family:'Syne',sans-serif;font-weight:700;font-size:1.1rem;color:#00D4FF;margin-bottom:0.6rem;">
        About This System
    </div>
    <div style="font-size:0.88rem;color:#94A3B8;line-height:1.7;">
        This system applies <strong style="color:#CDD6E0;">Federated Learning with FedProx</strong> to train
        energy consumption regression models across 10 simulated client partitions (each ~5,000 samples),
        without centralizing raw data. The best-performing model — <strong style="color:#00D4FF;">Random Forest</strong>
        — is then used to generate personalized, rule-based + model-based energy saving recommendations.
    </div>
</div>
""", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("""
    <div class="glass-card">
        <div style="color:#00D4FF;font-weight:700;margin-bottom:0.5rem;">Regression Task</div>
        <div style="font-size:0.83rem;color:#94A3B8;">Predict Global Active Power from sub-metering,
        voltage, and reactive power features using three regression models evaluated on RMSE and MAE.</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown("""
    <div class="glass-card">
        <div style="color:#7B2FBE;font-weight:700;margin-bottom:0.5rem;">Federated Privacy</div>
        <div style="font-size:0.83rem;color:#94A3B8;">Dataset partitioned into 10 client shards.
        FedProx training with proximal regularization ensures robust convergence vs standard FedAvg.</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown("""
    <div class="glass-card">
        <div style="color:#FF6B35;font-weight:700;margin-bottom:0.5rem;">10 Recommendations</div>
        <div style="font-size:0.83rem;color:#94A3B8;">Rule-based system cross-validated by the
        Random Forest model generates actionable energy savings guidance with 10–20% reduction targets.</div>
    </div>""", unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center;padding:2rem 0 0;font-family:'DM Mono',monospace;
    font-size:0.7rem;color:#334155;">
    Use the sidebar to navigate between pages
</div>
""", unsafe_allow_html=True)
