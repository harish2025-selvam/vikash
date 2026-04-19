import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BEST_MODEL_NAME, FEATURE_COLUMNS
from ml.preprocessing import preprocess, split_data, partition_for_federated


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true, y_pred):
    return float(mean_absolute_error(y_true, y_pred))


def get_model_instances():
    return {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=42),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, max_depth=12, random_state=42, n_jobs=-1
        ),
    }


def train_all_models(df):
    """Train all models and return results dict."""
    X, y, scaler, feat_cols = preprocess(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    results = {}
    trained_models = {}

    for name, model in get_model_instances().items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        results[name] = {
            "RMSE": round(rmse(y_test, preds), 4),
            "MAE": round(mae(y_test, preds), 4),
        }
        trained_models[name] = model

    # Select best by lowest RMSE
    best_name = min(results, key=lambda k: results[k]["RMSE"])
    best_model = trained_models[best_name]

    # Feature importance
    importance = None
    if hasattr(best_model, "feature_importances_"):
        importance = dict(zip(feat_cols, best_model.feature_importances_))

    return {
        "results": results,
        "best_name": best_name,
        "best_model": best_model,
        "scaler": scaler,
        "feature_cols": feat_cols,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": best_model.predict(X_test),
        "importance": importance,
    }


def predict_single(model, scaler, input_dict):
    """Predict for a single sample."""
    from config import FEATURE_COLUMNS
    row = np.array([[input_dict[c] for c in FEATURE_COLUMNS]], dtype=float)
    row_scaled = scaler.transform(row)
    return float(model.predict(row_scaled)[0])
