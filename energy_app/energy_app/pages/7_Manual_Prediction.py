import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(page_title="Manual Prediction — Energy AI", layout="wide", initial_sidebar_state="expanded")

from components.ui import inject_css, page_header, sidebar_info
from ml.preprocessing import load_data
from ml.model import train_all_models, predict_single
from ml.recommendations import generate_recommendations, compute_stats_from_single

inject_css()
sidebar_info()

if "df" not in st.session_state:
    st.session_state.df = load_data()
if "model_data" not in st.session_state:
    st.session_state.model_data = train_all_models(st.session_state.df)

md = st.session_state.model_data
df = st.session_state.df

page_header("Manual Prediction", "Input sensor values · Get prediction + recommendations")

st.markdown("""
<div class="glass-card" style="margin-bottom:1.5rem;">
    <div style="font-family:'Syne',sans-serif;font-weight:700;color:#00D4FF;margin-bottom:0.4rem;">
        How It Works
    </div>
    <div style="font-size:0.83rem;color:#94A3B8;line-height:1.6;">
        Enter your sensor readings below. The system preprocesses your input,
        passes it through <strong style="color:#CDD6E0;">Random Forest (FedProx Optimized)</strong>,
        predicts Global Active Power, and applies the same 10-rule recommendation engine
        used system-wide to generate personalized energy saving guidance.
    </div>
</div>
""", unsafe_allow_html=True)

# ── Input Form ─────────────────────────────────────────────────────────────────
st.markdown("""
<div style="font-family:'Syne',sans-serif;font-weight:700;font-size:1rem;
    color:#CDD6E0;margin-bottom:1rem;">Sensor Input Values</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    global_reactive = st.number_input(
        "Global Reactive Power (kVAR)",
        min_value=0.0, max_value=2.0, value=0.1, step=0.001, format="%.3f"
    )
    sub1 = st.number_input(
        "Sub Metering 1 — Kitchen (Wh)",
        min_value=0, max_value=100, value=0, step=1
    )

with col2:
    voltage = st.number_input(
        "Voltage (V)",
        min_value=220.0, max_value=255.0, value=241.0, step=0.1, format="%.1f"
    )
    sub2 = st.number_input(
        "Sub Metering 2 — Heating/Laundry (Wh)",
        min_value=0, max_value=100, value=0, step=1
    )

with col3:
    st.markdown("<br>", unsafe_allow_html=True)
    sub3 = st.number_input(
        "Sub Metering 3 — Cooling/Water (Wh)",
        min_value=0, max_value=100, value=1, step=1
    )

st.markdown("<br>", unsafe_allow_html=True)

# Threshold & sensitivity
col_t1, col_t2 = st.columns(2)
with col_t1:
    threshold = st.slider("Recommendation Threshold", 0.1, 2.0, 0.5, 0.05)
with col_t2:
    sensitivity = st.selectbox("Sensitivity", ["Low", "Medium", "High"], index=1)

predict_btn = st.button("Predict and Generate Recommendations")

# ── Prediction ─────────────────────────────────────────────────────────────────
if predict_btn:
    input_dict = {
        "Global_reactive_power": global_reactive,
        "Voltage": voltage,
        "Sub_metering_1": float(sub1),
        "Sub_metering_2": float(sub2),
        "Sub_metering_3": float(sub3),
    }

    with st.spinner("Running prediction pipeline..."):
        # Step 1: Predict
        prediction = predict_single(md["best_model"], md["scaler"], input_dict)

        # Step 2: Build stats for recommendation engine
        full_input = {**input_dict, "Global_active_power": prediction}
        stats = compute_stats_from_single(full_input)
        stats["pred_deviation"] = abs(prediction - df["Global_active_power"].mean())

        # Step 3: Recommendations
        recs = generate_recommendations(stats, threshold=threshold, sensitivity=sensitivity)

    # ── Show Prediction Result ─────────────────────────────────────────────────
    st.markdown(f"""
    <div class="pred-result">
        <div style="font-family:'DM Mono',monospace;font-size:0.72rem;color:#64748B;
            text-transform:uppercase;letter-spacing:1px;margin-bottom:0.5rem;">
            Predicted Global Active Power
        </div>
        <div class="pred-value">{prediction:.4f}</div>
        <div class="pred-unit">kilowatts (kW)</div>
        <div style="margin-top:1rem;font-family:'DM Mono',monospace;font-size:0.72rem;
            color:#A78BFA;">Model: Random Forest (FedProx Optimized)</div>
    </div>
    """, unsafe_allow_html=True)

    # Preprocessing summary
    with st.expander("Preprocessing Details", expanded=False):
        from config import FEATURE_COLUMNS
        import numpy as np
        raw_vals = [input_dict[c] for c in FEATURE_COLUMNS]
        scaled_vals = md["scaler"].transform([raw_vals])[0]
        col_r, col_s = st.columns(2)
        with col_r:
            st.markdown("**Raw Input**")
            for feat, val in zip(FEATURE_COLUMNS, raw_vals):
                st.markdown(f"""
                <div style="font-family:'DM Mono',monospace;font-size:0.78rem;
                    color:#94A3B8;padding:2px 0;">{feat}: <span style="color:#CDD6E0;">{val:.4f}</span></div>
                """, unsafe_allow_html=True)
        with col_s:
            st.markdown("**Scaled Input (StandardScaler)**")
            for feat, val in zip(FEATURE_COLUMNS, scaled_vals):
                st.markdown(f"""
                <div style="font-family:'DM Mono',monospace;font-size:0.78rem;
                    color:#94A3B8;padding:2px 0;">{feat}: <span style="color:#00D4FF;">{val:.4f}</span></div>
                """, unsafe_allow_html=True)

    # ── Recommendations ────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="font-family:'Syne',sans-serif;font-weight:700;font-size:1rem;
        color:#CDD6E0;margin:2rem 0 1rem;">
        {len(recs)} Personalized Recommendation{'s' if len(recs) != 1 else ''} for Your Input
    </div>
    """, unsafe_allow_html=True)

    if not recs:
        st.markdown("""
        <div class="glass-card" style="text-align:center;">
            <div style="color:#10B981;font-family:'Syne',sans-serif;font-weight:700;">
                No critical patterns detected
            </div>
            <div style="color:#64748B;font-size:0.83rem;margin-top:0.4rem;">
                Your input values are within acceptable bounds. Try lowering the threshold.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
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

else:
    # Placeholder state
    st.markdown("""
    <div style="text-align:center;padding:3rem;border:1px dashed rgba(0,212,255,0.1);
        border-radius:16px;color:#334155;">
        <div style="font-family:'Syne',sans-serif;font-size:1rem;color:#475569;margin-bottom:0.4rem;">
            Enter sensor values above and click Predict
        </div>
        <div style="font-family:'DM Mono',monospace;font-size:0.72rem;color:#334155;">
            Pipeline: Input → StandardScaler → Random Forest (FedProx) → Prediction → Recommendations
        </div>
    </div>
    """, unsafe_allow_html=True)
