# Personalized Energy Saving Recommendation System
### Using Federated Learning (FedProx)

---

## Overview

A production-level multi-page Streamlit application that:
- Trains **Linear Regression**, **Decision Tree**, and **Random Forest** regressors
- Simulates **Federated Learning** (FedAvg vs FedProx) across **10 clients** on ~50,000 data samples
- Selects the best model (Random Forest) based on lowest RMSE
- Generates **personalized energy saving recommendations** using a 10-rule system

---

## Project Structure

```
energy_app/
├── app.py                    # Main entry point
├── config.py                 # Central configuration
├── requirements.txt
├── pages/
│   ├── 1_Dashboard.py        # Dataset overview & metrics
│   ├── 2_Models.py           # Model comparison (RMSE, MAE)
│   ├── 3_Best_Model.py       # Random Forest details
│   ├── 4_FL_Monitor.py       # FedAvg vs FedProx training
│   ├── 5_Recommendations.py  # Auto + upload dataset recs
│   ├── 6_Analytics.py        # Trends, distributions, correlations
│   └── 7_Manual_Prediction.py# Manual input → prediction + recs
├── ml/
│   ├── model.py              # Train all models, select best
│   ├── federated.py          # FedAvg & FedProx simulation
│   ├── preprocessing.py      # Load, validate, scale, partition
│   └── recommendations.py    # 10-rule recommendation engine
├── components/
│   ├── ui.py                 # Dark theme CSS, shared helpers
│   └── charts.py             # Plotly chart functions
└── data/
    └── energy_data.csv       # Default dataset (~50,000 samples)
```

---

## Setup & Run

```bash
cd energy_app
pip install -r requirements.txt
streamlit run app.py
```

---

## Federated Learning Details

| Parameter | Value |
|-----------|-------|
| Total Data Samples | ~50,000 |
| Clients | 10 |
| Samples per Client | ~5,000 |
| FedAvg | Baseline strategy |
| FedProx µ | 0.01 |
| Rounds | 5–50 (configurable) |

---

## Models

| Model | Metric |
|-------|--------|
| Linear Regression | RMSE, MAE |
| Decision Tree | RMSE, MAE |
| **Random Forest** | **Best RMSE** |

Best model label: **Random Forest (FedProx Optimized)**

---

## Recommendation Rules

1. High Global_active_power → Reduce peak usage  
2. High Voltage fluctuation → Use stabilizer  
3. High Global_reactive_power → Improve power factor  
4. High Sub_metering_3 → Optimize cooling systems  
5. High Sub_metering_1 → Optimize kitchen usage  
6. High Sub_metering_2 → Reduce heating/laundry load  
7. Night usage high → Turn off idle devices  
8. High minimum usage → Reduce standby power  
9. Frequent spikes → Use scheduling  
10. Prediction deviation high → Follow model-based optimization  
