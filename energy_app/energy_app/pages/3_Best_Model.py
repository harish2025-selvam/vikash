import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(page_title="Best Model — Energy AI", layout="wide", initial_sidebar_state="expanded")

from components.ui import inject_css, page_header, sidebar_info
from components.charts import feature_importance_chart, scatter_chart
from ml.preprocessing import load_data
from ml.model import train_all_models

inject_css()
sidebar_info()

if "df" not in st.session_state:
    st.session_state.df = load_data()
if "model_data" not in st.session_state:
    st.session_state.model_data = train_all_models(st.session_state.df)

md = st.session_state.model_data

page_header("Best Model", "Random Forest · FedProx Optimized")

# Banner
st.markdown(f"""
<div class="best-model-banner">
    <div class="best-model-name">Random Forest</div>
    <div class="best-model-sub">Training Strategy: Federated Learning (FedProx) · Lowest RMSE</div>
    <div style="margin-top:1rem;display:flex;justify-content:center;gap:2rem;flex-wrap:wrap;">
        <div>
            <span style="font-family:'DM Mono',monospace;font-size:0.65rem;color:#64748B;text-transform:uppercase;">RMSE</span>
            <div style="font-family:'Syne',sans-serif;font-weight:800;font-size:1.5rem;color:#10B981;">
                {md['results'][md['best_name']]['RMSE']:.4f}
            </div>
        </div>
        <div>
            <span style="font-family:'DM Mono',monospace;font-size:0.65rem;color:#64748B;text-transform:uppercase;">MAE</span>
            <div style="font-family:'Syne',sans-serif;font-weight:800;font-size:1.5rem;color:#10B981;">
                {md['results'][md['best_name']]['MAE']:.4f}
            </div>
        </div>
        <div>
            <span style="font-family:'DM Mono',monospace;font-size:0.65rem;color:#64748B;text-transform:uppercase;">Strategy</span>
            <div style="font-family:'Syne',sans-serif;font-weight:800;font-size:1.1rem;color:#A78BFA;">
                FedProx
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Why Random Forest
st.markdown("""
<div class="glass-card">
    <div style="font-family:'Syne',sans-serif;font-weight:700;color:#00D4FF;margin-bottom:0.5rem;">Why Random Forest?</div>
    <div style="font-size:0.84rem;color:#94A3B8;line-height:1.7;">
        Random Forest consistently outperforms Linear Regression and Decision Tree on this dataset
        due to its ensemble of decision trees that reduces overfitting while capturing complex,
        non-linear relationships between sub-metering readings, voltage, and active power consumption.
        Combined with <strong style="color:#CDD6E0;">FedProx federated training</strong>,
        the proximal term regularization ensures stable weight aggregation across 10 client partitions.
    </div>
</div>
""", unsafe_allow_html=True)

col_a, col_b = st.columns(2)

with col_a:
    if md["importance"]:
        fig_imp = feature_importance_chart(md["importance"])
        st.plotly_chart(fig_imp, use_container_width=True)
    else:
        st.info("Feature importance not available.")

with col_b:
    import numpy as np
    fig_scatter = scatter_chart(md["y_test"], md["y_pred"], "Predicted vs Actual — Random Forest")
    st.plotly_chart(fig_scatter, use_container_width=True)
