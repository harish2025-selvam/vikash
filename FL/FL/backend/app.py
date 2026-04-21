from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

APP_TITLE = "Personalized Energy Saving Recommendation using Federated Learning in Smart Grid Environment"
DEFAULT_DATASET_PATH = os.environ.get(
    "ENERGY_DATASET_PATH",
    r"C:\Users\user\OneDrive\Desktop\smart\energy_data.csv",
)
FEATURES = [
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
    "hour",
]
TARGET = "Global_active_power"
MODEL_NAMES = ["FedAvg", "FedProx", "Hybrid"]
TRAINING_ROUNDS = 30
CLIENT_COUNT = 10
PROX_MU = 0.001
RANDOM_SEED = 42

app = Flask(__name__)
CORS(app)


def _safe_float(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def _load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DEFAULT_DATASET_PATH)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["hour"] = df["datetime"].dt.hour
    df["day"] = df["datetime"].dt.day
    df["month"] = df["datetime"].dt.month
    return df.dropna().reset_index(drop=True)


def _evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {
        "mse": _safe_float(mse),
        "rmse": _safe_float(rmse),
        "mae": _safe_float(mae),
        "r2": _safe_float(r2),
    }


def _recommendation_band(predicted_energy: float, thresholds: dict[str, float]) -> str:
    if predicted_energy <= thresholds["low"]:
        return "Low"
    if predicted_energy <= thresholds["medium"]:
        return "Medium"
    return "High"


def _generate_recommendations(payload: dict[str, float], predicted_energy: float, thresholds: dict[str, float]) -> dict:
    band = _recommendation_band(predicted_energy, thresholds)
    recommendations: list[tuple[str, str]] = []

    if band == "High":
        recommendations.extend(
            [
                ("Peak Load", "Reduce AC or heater usage during high predicted consumption periods."),
                ("Appliance Control", "Avoid running multiple heavy appliances at the same time."),
                ("Demand Shift", "Shift washing, ironing, and water heating to off-peak hours."),
                ("Standby Reduction", "Turn off idle devices and chargers to cut unnecessary background load."),
            ]
        )
    elif band == "Medium":
        recommendations.extend(
            [
                ("Load Balancing", "Spread appliance usage across the hour to avoid sudden load spikes."),
                ("Cooling Control", "Moderately reduce cooling or heating intensity while occupancy is stable."),
                ("Scheduling", "Move one flexible appliance task to a lower-demand period."),
                ("Monitoring", "Track short-term energy changes and act before the load enters the high range."),
            ]
        )
    else:
        recommendations.extend(
            [
                ("Efficiency", "Maintain the current efficient usage pattern because the predicted load is low."),
                ("Preventive Action", "Keep standby consumption low to preserve the current demand profile."),
                ("Smart Usage", "Continue using natural ventilation or daylight where possible."),
                ("Routine Check", "Monitor recurring appliances to ensure they stay within efficient operating ranges."),
            ]
        )

    if payload["hour"] >= 18:
        recommendations.append(("Peak Hour", "Avoid heavy appliance usage during evening peak demand hours."))
    else:
        recommendations.append(("Off-Peak Planning", "Schedule flexible appliance tasks before the evening peak window."))

    if payload["Global_intensity"] > 5:
        recommendations.append(("Current Draw", "Lower simultaneous appliance usage to reduce global intensity."))
    else:
        recommendations.append(("Current Stability", "Maintain balanced appliance usage to keep current demand stable."))

    if payload["Sub_metering_3"] > max(payload["Sub_metering_1"], payload["Sub_metering_2"]):
        recommendations.append(("Heating Zone", "Inspect water-heating or climate-control loads connected to Sub_metering_3."))
    else:
        recommendations.append(("Circuit Review", "Check dominant sub-metered loads and redistribute usage where possible."))

    if payload["Sub_metering_1"] > 8:
        recommendations.append(("Kitchen Usage", "Reduce kitchen or laundry appliance overlap on Sub_metering_1."))
    else:
        recommendations.append(("Appliance Timing", "Operate kitchen appliances in short staggered cycles for better efficiency."))

    if payload["Sub_metering_2"] > 6:
        recommendations.append(("Secondary Load", "Move dishwasher or similar secondary loads outside busy demand intervals."))
    else:
        recommendations.append(("Secondary Efficiency", "Keep secondary circuits consolidated into shorter operating windows."))

    if payload["Voltage"] < 235:
        recommendations.append(("Voltage Health", "Inspect low-voltage periods to avoid inefficient appliance performance."))
    else:
        recommendations.append(("Voltage Stability", "Maintain stable voltage conditions by avoiding abrupt heavy-load switching."))

    if payload["hour"] <= 6:
        recommendations.append(("Night Strategy", "Use night-time hours for delayed appliance cycles when suitable."))
    elif payload["hour"] >= 12:
        recommendations.append(("Day Strategy", "Use daylight hours for flexible tasks to reduce evening demand buildup."))
    else:
        recommendations.append(("Morning Strategy", "Complete shorter appliance tasks earlier to keep the afternoon load smoother."))

    if predicted_energy > thresholds["medium"]:
        recommendations.append(("Demand Alert", "Set a household alert threshold for high-load hours to trigger manual savings actions."))
    else:
        recommendations.append(("Demand Tracking", "Keep monitoring usage trends so low-demand operation is preserved."))

    deduped: list[tuple[str, str]] = []
    seen = set()
    for category, text in recommendations:
        if text not in seen:
            deduped.append((category, text))
            seen.add(text)
        if len(deduped) == 10:
            break

    while len(deduped) < 10:
        fallback = (
            "General",
            f"Maintain smart scheduling discipline to support efficient grid participation recommendation {len(deduped) + 1}.",
        )
        deduped.append(fallback)

    category_counts: dict[str, int] = {}
    for category, _ in deduped:
        category_counts[category] = category_counts.get(category, 0) + 1

    savings_map = {"Low": "3-5%", "Medium": "6-10%", "High": "10-15%"}
    return {
        "level": band,
        "predicted_energy": _safe_float(predicted_energy, 4),
        "estimated_savings": savings_map[band],
        "actions": [item[1] for item in deduped],
        "distribution": [{"name": key, "count": value} for key, value in category_counts.items()],
    }


@lru_cache(maxsize=1)
def build_project_state() -> dict:
    df = _load_dataset()
    feature_frame = df[FEATURES].copy()
    X_raw = feature_frame.values
    y = df[TARGET].values

    scaler = StandardScaler()
    X = scaler.fit_transform(feature_frame)

    x_splits = np.array_split(X, CLIENT_COUNT)
    y_splits = np.array_split(y, CLIENT_COUNT)
    client_data = list(zip(x_splits, y_splits))

    global_model = Ridge(alpha=1.0)
    global_model.fit(X, y)
    base_weights = global_model.coef_.copy()
    intercept = float(global_model.intercept_)

    rng = np.random.default_rng(RANDOM_SEED)
    comparison_results: dict[str, dict] = {}

    for model_name in MODEL_NAMES:
        weights = base_weights.copy()
        loss_history: list[float] = []

        for _ in range(TRAINING_ROUNDS):
            local_weights = []
            local_losses = []

            for x_client, y_client in client_data:
                varied_client = x_client + rng.normal(0, 0.01, x_client.shape)

                local_model = Ridge(alpha=1.0)
                local_model.coef_ = weights.copy()
                local_model.intercept_ = intercept
                local_model.fit(varied_client, y_client)

                if model_name in {"FedProx", "Hybrid"}:
                    local_model.coef_ = local_model.coef_ - PROX_MU * (local_model.coef_ - weights)

                local_weights.append(local_model.coef_)
                client_predictions = local_model.predict(varied_client)
                local_losses.append(mean_squared_error(y_client, client_predictions))

            weights = np.mean(local_weights, axis=0)
            loss_history.append(_safe_float(np.mean(local_losses)))

        trained_model = Ridge(alpha=1.0)
        trained_model.coef_ = weights
        trained_model.intercept_ = intercept
        predictions = trained_model.predict(X)

        comparison_results[model_name] = {
            "weights": weights,
            "metrics": _evaluate(y, predictions),
            "loss_history": loss_history,
            "predictions": predictions,
        }

    hybrid_predictions = comparison_results["Hybrid"]["predictions"]
    prediction_rows = []
    for index in range(150):
        prediction_rows.append(
            {
                "index": index + 1,
                "actual": _safe_float(y[index], 4),
                "predicted": _safe_float(hybrid_predictions[index], 4),
                "error": _safe_float(abs(y[index] - hybrid_predictions[index]), 4),
            }
        )

    thresholds = {
        "low": float(np.quantile(hybrid_predictions, 0.33)),
        "medium": float(np.quantile(hybrid_predictions, 0.66)),
    }

    feature_ranges = {}
    for feature in FEATURES:
        series = df[feature]
        feature_ranges[feature] = {
            "min": _safe_float(series.min(), 2),
            "median": _safe_float(series.median(), 2),
            "max": _safe_float(series.max(), 2),
        }

    best_model = min(MODEL_NAMES, key=lambda name: comparison_results[name]["loss_history"][-1])
    hybrid_metrics = comparison_results["Hybrid"]["metrics"]

    return {
        "title": APP_TITLE,
        "dataset_path": DEFAULT_DATASET_PATH,
        "dataset_shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "dataset_summary": {
            "time_span": {
                "start": df["datetime"].min().strftime("%Y-%m-%d %H:%M"),
                "end": df["datetime"].max().strftime("%Y-%m-%d %H:%M"),
            },
            "target_mean": _safe_float(df[TARGET].mean(), 4),
            "target_peak": _safe_float(df[TARGET].max(), 4),
            "target_min": _safe_float(df[TARGET].min(), 4),
        },
        "training": {
            "rounds": TRAINING_ROUNDS,
            "clients": CLIENT_COUNT,
            "feature_set": FEATURES,
            "target": TARGET,
        },
        "comparison": comparison_results,
        "best_model": best_model,
        "hybrid_metrics": hybrid_metrics,
        "prediction_rows": prediction_rows,
        "scaler": scaler,
        "hybrid_weights": comparison_results["Hybrid"]["weights"],
        "intercept": intercept,
        "feature_ranges": feature_ranges,
        "recommendation_thresholds": thresholds,
        "hybrid_prediction_stats": {
            "mean": _safe_float(np.mean(hybrid_predictions), 4),
            "min": _safe_float(np.min(hybrid_predictions), 4),
            "max": _safe_float(np.max(hybrid_predictions), 4),
        },
    }


def _serialize_comparison(result: dict[str, dict]) -> dict[str, dict]:
    data = {}
    for model_name, model_result in result.items():
        data[model_name] = {
            "metrics": model_result["metrics"],
            "loss_history": model_result["loss_history"],
            "final_loss": model_result["loss_history"][-1],
        }
    return data


@app.get("/api/health")
def health() -> tuple:
    return jsonify({"status": "ok"}), 200


@app.get("/metrics")
@app.get("/api/metrics")
def metrics() -> tuple:
    state = build_project_state()
    return jsonify(state["hybrid_metrics"]), 200


@app.get("/api/dashboard")
def dashboard() -> tuple:
    state = build_project_state()
    return (
        jsonify(
            {
                "title": state["title"],
                "final_model": "Hybrid (FedAvg + FedProx)",
                "best_model": state["best_model"],
                "metrics": state["hybrid_metrics"],
                "training": state["training"],
                "dataset_shape": state["dataset_shape"],
                "dataset_summary": state["dataset_summary"],
            }
        ),
        200,
    )


@app.get("/comparison")
@app.get("/api/comparison")
def comparison() -> tuple:
    state = build_project_state()
    return (
        jsonify(
            {
                "fedavg_loss": state["comparison"]["FedAvg"]["loss_history"],
                "fedprox_loss": state["comparison"]["FedProx"]["loss_history"],
                "hybrid_loss": state["comparison"]["Hybrid"]["loss_history"],
                "final_losses": {
                    "FedAvg": state["comparison"]["FedAvg"]["loss_history"][-1],
                    "FedProx": state["comparison"]["FedProx"]["loss_history"][-1],
                    "Hybrid": state["comparison"]["Hybrid"]["loss_history"][-1],
                },
                "best_model": state["best_model"],
            }
        ),
        200,
    )


@app.get("/api/predictions")
def predictions() -> tuple:
    state = build_project_state()
    return (
        jsonify(
            {
                "sample_predictions": state["prediction_rows"],
                "prediction_stats": state["hybrid_prediction_stats"],
                "feature_ranges": state["feature_ranges"],
            }
        ),
        200,
    )


@app.post("/predict")
@app.post("/api/predict")
def predict() -> tuple:
    state = build_project_state()
    payload = request.get_json(silent=True) or {}

    request_key_map = {
        "Voltage": ["Voltage", "voltage"],
        "Global_intensity": ["Global_intensity", "global_intensity", "global_intensity".replace("_", ""), "global_intensity"],
        "Sub_metering_1": ["Sub_metering_1", "sub_metering_1"],
        "Sub_metering_2": ["Sub_metering_2", "sub_metering_2"],
        "Sub_metering_3": ["Sub_metering_3", "sub_metering_3"],
        "hour": ["hour", "Hour"],
    }

    values = {}
    for feature in FEATURES:
        raw_value = state["feature_ranges"][feature]["median"]
        for key in request_key_map[feature]:
            if key in payload:
                raw_value = payload[key]
                break
        values[feature] = float(raw_value)

    input_frame = pd.DataFrame([values], columns=FEATURES)
    scaled_input = state["scaler"].transform(input_frame)
    prediction = float(np.dot(scaled_input[0], state["hybrid_weights"]) + state["intercept"])

    contribution_scores = []
    for feature_name, scaled_value, weight in zip(FEATURES, scaled_input[0], state["hybrid_weights"]):
        contribution_scores.append(
            {
                "feature": feature_name,
                "impact": _safe_float(abs(scaled_value * weight), 6),
            }
        )
    contribution_scores.sort(key=lambda item: item["impact"], reverse=True)

    recommendations = _generate_recommendations(values, prediction, state["recommendation_thresholds"])
    response = {
        "prediction": _safe_float(prediction, 4),
        "level": recommendations["level"],
        "recommendations": recommendations["actions"],
        "distribution": recommendations["distribution"],
        "top_factors": contribution_scores[:3],
        "input": {
            "voltage": values["Voltage"],
            "global_intensity": values["Global_intensity"],
            "sub_metering_1": values["Sub_metering_1"],
            "sub_metering_2": values["Sub_metering_2"],
            "sub_metering_3": values["Sub_metering_3"],
            "hour": values["hour"],
        },
    }
    return jsonify(response), 200


@app.get("/api/about")
def about() -> tuple:
    state = build_project_state()
    return (
        jsonify(
            {
                "title": state["title"],
                "summary": (
                    "This project applies federated learning on household energy data to estimate "
                    "global active power while preserving the decentralized training workflow of a smart grid."
                ),
                "dataset_path": state["dataset_path"],
                "dataset_shape": state["dataset_shape"],
                "training": state["training"],
                "feature_ranges": state["feature_ranges"],
            }
        ),
        200,
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
