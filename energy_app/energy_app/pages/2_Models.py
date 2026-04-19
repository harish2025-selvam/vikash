import streamlit as st
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(page_title="Models — Energy AI", layout="wide", initial_sidebar_state="expanded")

from components.ui import inject_css, page_header, sidebar_info
from components.charts import model_comparison_chart
from ml.preprocessing import load_data
from ml.model import train_all_models

inject_css()
sidebar_info()

if "df" not in st.session_state:
    st.session_state.df = load_data()
if "model_data" not in st.session_state:
    st.session_state.model_data = train_all_models(st.session_state.df)

md = st.session_state.model_data
results = md["results"]
best = md["best_name"]

page_header("Model Comparison", "Regression models evaluated on RMSE and MAE")

st.markdown("<br>", unsafe_allow_html=True)

# Metrics row - one column per model
cols = st.columns(3)
model_names = list(results.keys())

for i, name in enumerate(model_names):
    with cols[i]:
        is_best = (name == best)
        label = name + (" — BEST" if is_best else "")
        st.markdown(f"**{label}**")
        m1, m2 = st.columns(2)
        m1.metric("RMSE", f"{results[name]['RMSE']:.4f}")
        m2.metric("MAE",  f"{results[name]['MAE']:.4f}")
        if is_best:
            st.success("Lowest RMSE — Selected")
        st.markdown("---")

# Summary table
st.markdown("**Performance Summary**")
rows = []
for name in model_names:
    rows.append({
        "Model": name,
        "RMSE": results[name]["RMSE"],
        "MAE":  results[name]["MAE"],
        "Selected": "YES" if name == best else "",
    })
df_table = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
st.dataframe(df_table, use_container_width=True, hide_index=True)

st.markdown("<br>", unsafe_allow_html=True)

# Bar chart
fig = model_comparison_chart(results)
st.plotly_chart(fig, use_container_width=True)

st.info(
    "All models trained on 80% of the dataset, evaluated on 20% holdout. "
    "Lower RMSE and MAE = better regression accuracy. "
    "Random Forest wins with the lowest RMSE and is deployed with FedProx training strategy."
)
