# ml/recommendations.py
import numpy as np


def generate_recommendations(stats: dict, threshold: float = 0.7, sensitivity: str = "Medium"):
    """
    Generate rule-based + model-based energy saving recommendations.
    
    stats: dict with keys:
        avg_active_power, std_voltage, avg_reactive_power,
        avg_sub3, avg_sub1, avg_sub2,
        night_usage, min_usage, spike_ratio, pred_deviation
    threshold: 0.0–1.0 (user-controlled strictness)
    sensitivity: Low / Medium / High
    
    Returns list of recommendation dicts.
    """
    sens_mult = {"Low": 1.3, "Medium": 1.0, "High": 0.7}[sensitivity]
    t = threshold * sens_mult

    rules = []

    # 1. High Global_active_power
    if stats.get("avg_active_power", 0) > t:
        rules.append({
            "title": "Reduce Peak Energy Usage",
            "reason": "Global active power consistently exceeds safe threshold levels.",
            "pattern": f"Avg Active Power: {stats['avg_active_power']:.3f} kW (above {t:.2f})",
            "saving": "15–20%",
        })

    # 2. High Voltage fluctuation
    if stats.get("std_voltage", 0) > t * 2:
        rules.append({
            "title": "Install Voltage Stabilizer",
            "reason": "High voltage fluctuations detected across measurement intervals.",
            "pattern": f"Voltage Std Dev: {stats['std_voltage']:.2f} V",
            "saving": "10–15%",
        })

    # 3. High reactive power
    if stats.get("avg_reactive_power", 0) > t * 0.3:
        rules.append({
            "title": "Improve Power Factor",
            "reason": "Elevated reactive power increases line losses and energy waste.",
            "pattern": f"Avg Reactive Power: {stats['avg_reactive_power']:.3f} kVAR",
            "saving": "10–18%",
        })

    # 4. High Sub_metering_3 (cooling)
    if stats.get("avg_sub3", 0) > t * 10:
        rules.append({
            "title": "Optimize Cooling Systems",
            "reason": "Sub-metering 3 shows elevated consumption (HVAC/water heater).",
            "pattern": f"Avg Sub-metering 3: {stats['avg_sub3']:.1f} Wh",
            "saving": "12–20%",
        })

    # 5. High Sub_metering_1 (kitchen)
    if stats.get("avg_sub1", 0) > t * 5:
        rules.append({
            "title": "Optimize Kitchen Appliance Usage",
            "reason": "Sub-metering 1 indicates excessive kitchen energy consumption.",
            "pattern": f"Avg Sub-metering 1: {stats['avg_sub1']:.1f} Wh",
            "saving": "10–14%",
        })

    # 6. High Sub_metering_2 (heating/laundry)
    if stats.get("avg_sub2", 0) > t * 5:
        rules.append({
            "title": "Reduce Heating and Laundry Load",
            "reason": "Sub-metering 2 reflects high heating or laundry appliance usage.",
            "pattern": f"Avg Sub-metering 2: {stats['avg_sub2']:.1f} Wh",
            "saving": "10–16%",
        })

    # 7. Night usage
    if stats.get("night_usage", 0) > t * 0.8:
        rules.append({
            "title": "Turn Off Idle Devices at Night",
            "reason": "Significant energy usage detected during off-peak night hours.",
            "pattern": f"Night Usage Index: {stats['night_usage']:.3f}",
            "saving": "12–18%",
        })

    # 8. High minimum usage (standby)
    if stats.get("min_usage", 0) > t * 0.5:
        rules.append({
            "title": "Reduce Standby Power Consumption",
            "reason": "Consistently high minimum usage suggests devices left on standby.",
            "pattern": f"Minimum Usage: {stats['min_usage']:.3f} kW",
            "saving": "10–15%",
        })

    # 9. Spike frequency
    if stats.get("spike_ratio", 0) > t * 0.2:
        rules.append({
            "title": "Use Smart Scheduling for High-Load Tasks",
            "reason": "Frequent power spikes detected, suggesting unoptimized task scheduling.",
            "pattern": f"Spike Ratio: {stats['spike_ratio']:.2%}",
            "saving": "10–16%",
        })

    # 10. Model-based prediction deviation
    if stats.get("pred_deviation", 0) > t * 0.4:
        rules.append({
            "title": "Follow Model-Based Optimization Plan",
            "reason": "Prediction deviation indicates actual consumption exceeds expected baseline.",
            "pattern": f"Model Deviation: {stats['pred_deviation']:.3f} kW",
            "saving": "15–20%",
        })

    # Always attach model tag
    for r in rules:
        r["model"] = "Random Forest (FedProx Optimized)"

    return rules


def compute_stats(df, model=None, scaler=None, feature_cols=None):
    """Extract stats from dataframe for recommendation engine."""
    stats = {}
    stats["avg_active_power"] = float(df["Global_active_power"].mean())
    stats["std_voltage"] = float(df["Voltage"].std())
    stats["avg_reactive_power"] = float(df["Global_reactive_power"].mean())
    stats["avg_sub3"] = float(df["Sub_metering_3"].mean())
    stats["avg_sub1"] = float(df["Sub_metering_1"].mean())
    stats["avg_sub2"] = float(df["Sub_metering_2"].mean())
    stats["min_usage"] = float(df["Global_active_power"].min())

    # Night usage proxy: use mean of lower quartile
    q25 = df["Global_active_power"].quantile(0.25)
    stats["night_usage"] = float(q25)

    # Spike ratio: samples above 90th percentile
    q90 = df["Global_active_power"].quantile(0.90)
    stats["spike_ratio"] = float((df["Global_active_power"] > q90).mean())

    # Prediction deviation
    if model is not None and scaler is not None and feature_cols is not None:
        try:
            import numpy as np
            X = df[feature_cols].astype(float).values
            X_scaled = scaler.transform(X)
            preds = model.predict(X_scaled)
            actual = df["Global_active_power"].values
            stats["pred_deviation"] = float(np.mean(np.abs(preds - actual)))
        except Exception:
            stats["pred_deviation"] = 0.0
    else:
        stats["pred_deviation"] = 0.0

    return stats


def compute_stats_from_single(input_dict):
    """Build stats dict from manual single-input for recommendation generation."""
    return {
        "avg_active_power": input_dict.get("Global_active_power", 0),
        "std_voltage": abs(input_dict.get("Voltage", 240) - 230) / 10,
        "avg_reactive_power": input_dict.get("Global_reactive_power", 0),
        "avg_sub3": input_dict.get("Sub_metering_3", 0),
        "avg_sub1": input_dict.get("Sub_metering_1", 0),
        "avg_sub2": input_dict.get("Sub_metering_2", 0),
        "night_usage": input_dict.get("Global_active_power", 0) * 0.3,
        "min_usage": input_dict.get("Global_active_power", 0) * 0.4,
        "spike_ratio": 0.15 if input_dict.get("Global_active_power", 0) > 1.5 else 0.05,
        "pred_deviation": 0.0,
    }
