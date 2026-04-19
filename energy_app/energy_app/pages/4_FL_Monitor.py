import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(page_title="FL Monitor — Energy AI", layout="wide", initial_sidebar_state="expanded")

from components.ui import inject_css, page_header, sidebar_info, fl_stat
from components.charts import fl_comparison_chart, fl_loss_chart
from ml.preprocessing import load_data, preprocess
from ml.model import train_all_models
from ml.federated import compare_strategies
from config import NUM_CLIENTS

inject_css()
sidebar_info()

if "df" not in st.session_state:
    st.session_state.df = load_data()
if "model_data" not in st.session_state:
    st.session_state.model_data = train_all_models(st.session_state.df)

df = st.session_state.df

page_header("Federated Learning Monitor", "FedAvg vs FedProx training simulation")

# Top FL stats
st.markdown(f"""
<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;margin-bottom:2rem;">
    {fl_stat("~50,000", "Total Data Samples")}
    {fl_stat("10", "Federated Clients")}
    {fl_stat("~5,000", "Samples per Client")}
</div>
""", unsafe_allow_html=True)

# Controls
col_ctrl, col_info = st.columns([1, 2])
with col_ctrl:
    num_rounds = st.slider("Number of Training Rounds", min_value=5, max_value=50, value=20, step=5)
    run_btn = st.button("Simulate Federated Training")

with col_info:
    st.markdown("""
    <div class="glass-card" style="margin-top:0;">
        <div style="font-family:'Syne',sans-serif;font-weight:700;color:#00D4FF;margin-bottom:0.4rem;">
            About FedProx
        </div>
        <div style="font-size:0.82rem;color:#94A3B8;line-height:1.6;">
            FedProx is a <strong style="color:#CDD6E0;">federated training strategy</strong> (not a model)
            that adds a proximal regularization term to each client's local objective.
            This reduces client drift caused by heterogeneous data distributions across partitions,
            resulting in faster convergence and lower final loss compared to standard FedAvg.
        </div>
    </div>
    """, unsafe_allow_html=True)

if run_btn or "fl_results" not in st.session_state:
    with st.spinner(f"Simulating {num_rounds} federated rounds..."):
        X, y, _, _ = preprocess(df)
        fedavg_res, fedprox_res = compare_strategies(X, y, num_rounds=num_rounds)
        st.session_state.fl_results = (fedavg_res, fedprox_res)

if "fl_results" in st.session_state:
    fedavg_res, fedprox_res = st.session_state.fl_results

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("FedAvg Final R²", f"{fedavg_res['final_accuracy']:.4f}")
    col2.metric("FedProx Final R²", f"{fedprox_res['final_accuracy']:.4f}",
                delta=f"+{fedprox_res['final_accuracy']-fedavg_res['final_accuracy']:.4f}")
    col3.metric("FedAvg Final Loss", f"{fedavg_res['final_loss']:.4f}")
    col4.metric("FedProx Final Loss", f"{fedprox_res['final_loss']:.4f}",
                delta=f"{fedprox_res['final_loss']-fedavg_res['final_loss']:.4f}")

    st.markdown("<br>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        fig_acc = fl_comparison_chart(fedavg_res, fedprox_res)
        st.plotly_chart(fig_acc, use_container_width=True)
    with col_b:
        fig_loss = fl_loss_chart(fedavg_res, fedprox_res)
        st.plotly_chart(fig_loss, use_container_width=True)

    # Strategy comparison table
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-family:'Syne',sans-serif;font-weight:700;font-size:1rem;
        color:#CDD6E0;margin-bottom:1rem;">Strategy Comparison Summary</div>
    """, unsafe_allow_html=True)

    col_x, col_y = st.columns(2)
    with col_x:
        st.markdown(f"""
        <div class="glass-card">
            <div style="font-family:'Syne',sans-serif;font-weight:700;color:#64748B;margin-bottom:0.5rem;">
                FedAvg (Baseline)
            </div>
            <div style="font-size:0.83rem;color:#94A3B8;line-height:1.6;">
                Standard federated averaging. Simple weighted aggregation of client weights.
                Susceptible to client drift on heterogeneous data.<br><br>
                <span style="color:#CDD6E0;">Final R²: {fedavg_res['final_accuracy']:.4f}</span><br>
                <span style="color:#CDD6E0;">Final Loss: {fedavg_res['final_loss']:.4f}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col_y:
        st.markdown(f"""
        <div class="glass-card" style="border-color:rgba(0,212,255,0.25);">
            <div style="font-family:'Syne',sans-serif;font-weight:700;color:#00D4FF;margin-bottom:0.5rem;">
                FedProx (Selected Strategy)
            </div>
            <div style="font-size:0.83rem;color:#94A3B8;line-height:1.6;">
                Adds proximal term (mu={0.01}) to each client's loss to penalize deviation from the
                global model. Ensures robust convergence on partitioned energy data.<br><br>
                <span style="color:#10B981;">Final R²: {fedprox_res['final_accuracy']:.4f}</span><br>
                <span style="color:#10B981;">Final Loss: {fedprox_res['final_loss']:.4f}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
