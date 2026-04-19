import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Mono, monospace", color="#CDD6E0", size=11),
    margin=dict(l=40, r=20, t=40, b=40),
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.05)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.05)"),
)


def line_chart(x, y, title, xlabel="", ylabel="", color="#00D4FF"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="lines",
        line=dict(color=color, width=2),
        fill="tozeroy",
        fillcolor=f"rgba(0,212,255,0.06)",
    ))
    fig.update_layout(**CHART_LAYOUT, title=dict(text=title, font=dict(family="Syne, sans-serif", size=14)))
    fig.update_xaxes(title_text=xlabel)
    fig.update_yaxes(title_text=ylabel)
    return fig


def bar_chart(names, values, title, color="#00D4FF"):
    colors = [color] * len(names)
    fig = go.Figure(go.Bar(x=names, y=values, marker_color=colors, marker_line_color="rgba(0,212,255,0.3)", marker_line_width=1))
    fig.update_layout(**CHART_LAYOUT, title=dict(text=title, font=dict(family="Syne, sans-serif", size=14)))
    return fig


def scatter_chart(y_true, y_pred, title="Actual vs Predicted"):
    n = min(500, len(y_true))
    idx = np.random.choice(len(y_true), n, replace=False)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_true[idx], y=y_pred[idx],
        mode="markers",
        marker=dict(color="#00D4FF", size=4, opacity=0.6),
        name="Predictions",
    ))
    mn, mx = float(min(y_true.min(), y_pred.min())), float(max(y_true.max(), y_pred.max()))
    fig.add_trace(go.Scatter(
        x=[mn, mx], y=[mn, mx],
        mode="lines",
        line=dict(color="#FF6B35", dash="dash", width=1.5),
        name="Perfect Fit",
    ))
    fig.update_layout(**CHART_LAYOUT, title=dict(text=title, font=dict(family="Syne, sans-serif", size=14)))
    fig.update_xaxes(title_text="Actual")
    fig.update_yaxes(title_text="Predicted")
    return fig


def model_comparison_chart(results):
    models = list(results.keys())
    rmses = [results[m]["RMSE"] for m in models]
    maes = [results[m]["MAE"] for m in models]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="RMSE", x=models, y=rmses, marker_color="#00D4FF"))
    fig.add_trace(go.Bar(name="MAE", x=models, y=maes, marker_color="#7B2FBE"))
    fig.update_layout(**CHART_LAYOUT, barmode="group",
                      title=dict(text="Model Performance Comparison", font=dict(family="Syne, sans-serif", size=14)))
    return fig


def feature_importance_chart(importance_dict):
    sorted_items = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    names = [i[0] for i in sorted_items]
    vals = [i[1] for i in sorted_items]
    colors = ["#00D4FF", "#7B2FBE", "#FF6B35", "#10B981", "#F59E0B"][:len(names)]
    fig = go.Figure(go.Bar(
        x=vals, y=names, orientation="h",
        marker_color=colors,
        marker_line_color="rgba(255,255,255,0.1)",
        marker_line_width=1,
    ))
    layout = dict(CHART_LAYOUT)
    layout["yaxis"] = dict(autorange="reversed", gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.05)")
    layout["title"] = dict(text="Feature Importance (Random Forest)", font=dict(family="Syne, sans-serif", size=14))
    fig.update_layout(**layout)
    return fig


def fl_comparison_chart(fedavg_result, fedprox_result):
    rounds = list(range(1, len(fedavg_result["accuracy_history"]) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rounds, y=fedavg_result["accuracy_history"],
        mode="lines+markers", name="FedAvg",
        line=dict(color="#64748B", width=2, dash="dash"),
        marker=dict(size=4),
    ))
    fig.add_trace(go.Scatter(
        x=rounds, y=fedprox_result["accuracy_history"],
        mode="lines+markers", name="FedProx",
        line=dict(color="#00D4FF", width=2),
        marker=dict(size=5),
    ))
    fig.update_layout(**CHART_LAYOUT,
                      title=dict(text="FedAvg vs FedProx — Accuracy per Round", font=dict(family="Syne, sans-serif", size=14)),
                      legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#CDD6E0")))
    fig.update_xaxes(title_text="Round")
    fig.update_yaxes(title_text="R² Score")
    return fig


def fl_loss_chart(fedavg_result, fedprox_result):
    rounds = list(range(1, len(fedavg_result["loss_history"]) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rounds, y=fedavg_result["loss_history"],
        mode="lines", name="FedAvg",
        line=dict(color="#64748B", width=2, dash="dash"),
    ))
    fig.add_trace(go.Scatter(
        x=rounds, y=fedprox_result["loss_history"],
        mode="lines", name="FedProx",
        line=dict(color="#FF6B35", width=2),
    ))
    fig.update_layout(**CHART_LAYOUT,
                      title=dict(text="FedAvg vs FedProx — Loss per Round", font=dict(family="Syne, sans-serif", size=14)),
                      legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#CDD6E0")))
    fig.update_xaxes(title_text="Round")
    fig.update_yaxes(title_text="MSE Loss")
    return fig


def histogram_chart(series, title, color="#00D4FF"):
    fig = go.Figure(go.Histogram(x=series, marker_color=color, opacity=0.8, nbinsx=40))
    fig.update_layout(**CHART_LAYOUT, title=dict(text=title, font=dict(family="Syne, sans-serif", size=14)))
    return fig


def submetering_area_chart(df):
    sample = df.sample(min(500, len(df)), random_state=42).sort_index()
    fig = go.Figure()
    for col, color in [("Sub_metering_1", "#00D4FF"), ("Sub_metering_2", "#7B2FBE"), ("Sub_metering_3", "#FF6B35")]:
        fig.add_trace(go.Scatter(
            x=list(range(len(sample))), y=sample[col],
            mode="lines", name=col,
            line=dict(color=color, width=1.5),
            fill="tozeroy",
            fillcolor=color.replace("#", "rgba(") + ",0.06)" if "#" in color else color,
        ))
    fig.update_layout(**CHART_LAYOUT,
                      title=dict(text="Sub-metering Breakdown (Sample)", font=dict(family="Syne, sans-serif", size=14)),
                      legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#CDD6E0")))
    return fig
