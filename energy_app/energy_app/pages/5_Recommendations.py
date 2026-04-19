import streamlit as st
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(page_title="Recommendations — Energy AI", layout="wide", initial_sidebar_state="expanded")

from components.ui import inject_css, page_header, sidebar_info
from ml.preprocessing import load_data, validate_columns
from ml.model import train_all_models
from ml.recommendations import generate_recommendations, compute_stats

inject_css()
sidebar_info()

if "df" not in st.session_state:
    st.session_state.df = load_data()
if "model_data" not in st.session_state:
    st.session_state.model_data = train_all_models(st.session_state.df)

page_header("Energy Saving Recommendations", "Rule-based + Model-based · Random Forest (FedProx Optimized)")

# ── Dataset Section ────────────────────────────────────────────────────────────
with st.expander("Dataset Source", expanded=False):
    uploaded = st.file_uploader("Upload custom CSV (optional — overrides default)", type=["csv"])
    if uploaded:
        try:
            df_up = pd.read_csv(uploaded)
            valid, missing = validate_columns(df_up)
            if valid:
                st.session_state.df = df_up
                st.session_state.model_data = train_all_models(df_up)
                st.success(f"Custom dataset loaded: {len(df_up):,} samples")
            else:
                st.error(f"Invalid dataset. Missing columns: {missing}. Falling back to default.")
        except Exception as e:
            st.error(f"Could not read file: {e}. Using default dataset.")

    df_show = st.session_state.df
    st.markdown(f"""
    <div style="font-family:'DM Mono',monospace;font-size:0.72rem;color:#64748B;margin-bottom:0.5rem;">
        Active dataset: <span style="color:#00D4FF;">{len(df_show):,} samples</span>
    </div>
    """, unsafe_allow_html=True)
    st.dataframe(df_show.head(50), use_container_width=True)

# ── Controls ───────────────────────────────────────────────────────────────────
df = st.session_state.df
md = st.session_state.model_data

col_ctrl1, col_ctrl2 = st.columns(2)
with col_ctrl1:
    threshold = st.slider(
        "Detection Threshold",
        min_value=0.1, max_value=2.0, value=0.7, step=0.05,
        help="Lower = more recommendations triggered. Higher = stricter threshold."
    )
with col_ctrl2:
    sensitivity = st.selectbox(
        "Sensitivity",
        ["Low", "Medium", "High"],
        index=1,
        help="Controls how aggressively patterns are flagged."
    )

generate_btn = st.button("Generate Recommendations")

if generate_btn or "rec_results" not in st.session_state:
    with st.spinner("Analyzing energy patterns..."):
        stats = compute_stats(
            df,
            model=md["best_model"],
            scaler=md["scaler"],
            feature_cols=md["feature_cols"],
        )
        recs = generate_recommendations(stats, threshold=threshold, sensitivity=sensitivity)
        st.session_state.rec_results = recs
        st.session_state.rec_stats = stats

# ── Display Recommendations ────────────────────────────────────────────────────
if "rec_results" in st.session_state:
    recs = st.session_state.rec_results
    stats = st.session_state.rec_stats

    st.markdown("<br>", unsafe_allow_html=True)

    if not recs:
        st.markdown("""
        <div class="glass-card" style="text-align:center;">
            <div style="color:#10B981;font-family:'Syne',sans-serif;font-weight:700;font-size:1.1rem;">
                No critical patterns detected
            </div>
            <div style="color:#64748B;font-size:0.83rem;margin-top:0.4rem;">
                Your energy usage is within acceptable bounds at the current threshold.
                Try lowering the threshold or switching to High sensitivity to reveal subtle patterns.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        col_sum1, col_sum2, col_sum3 = st.columns(3)
        col_sum1.metric("Recommendations Found", len(recs))
        col_sum2.metric("Avg Active Power", f"{stats['avg_active_power']:.3f} kW")
        col_sum3.metric("Spike Ratio", f"{stats['spike_ratio']:.1%}")

        st.markdown(f"""
        <div style="font-family:'Syne',sans-serif;font-weight:700;font-size:1rem;
            color:#CDD6E0;margin:1.5rem 0 1rem;">
            {len(recs)} Recommendation{'s' if len(recs) != 1 else ''} Generated
        </div>
        """, unsafe_allow_html=True)

        for i, rec in enumerate(recs, 1):
            st.markdown(f"""
            <div class="rec-card">
                <div class="rec-title">{i}. {rec['title']}</div>
                <div class="rec-reason">{rec['reason']}</div>
                <div style="font-family:'DM Mono',monospace;font-size:0.75rem;color:#7C8FA6;
                    background:rgba(255,255,255,0.03);border-radius:6px;
                    padding:0.4rem 0.7rem;margin:0.5rem 0;border-left:2px solid rgba(0,212,255,0.3);">
                    Detected Pattern: {rec['pattern']}
                </div>
                <div>
                    <span class="rec-badge">Expected Saving: {rec['saving']}</span>
                    <span class="rec-model-tag">Generated Using: {rec['model']}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
