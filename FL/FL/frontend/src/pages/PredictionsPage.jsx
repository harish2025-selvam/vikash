import { useEffect, useMemo, useState } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { fetchPredictions } from "../api";
import MetricCard from "../components/MetricCard";
import PredictionForm from "../components/PredictionForm";
import { ErrorState, LoadingState } from "../components/PageState";

function PredictionsPage() {
  const [data, setData] = useState(null);
  const [status, setStatus] = useState("loading");
  const [predictionResult, setPredictionResult] = useState(null);

  useEffect(() => {
    let active = true;
    fetchPredictions()
      .then((response) => {
        if (!active) {
          return;
        }
        setData(response);
        setStatus("ready");
      })
      .catch(() => {
        if (active) {
          setStatus("error");
        }
      });

    return () => {
      active = false;
    };
  }, []);

  const lineChartData = useMemo(() => {
    if (!data) {
      return [];
    }
    return data.sample_predictions.slice(0, 30).map((row) => ({
      sample: row.index,
      actual: row.actual,
      predicted: row.predicted,
    }));
  }, [data]);

  if (status === "loading") {
    return <LoadingState label="Loading prediction workspace..." />;
  }

  if (status === "error" || !data) {
    return <ErrorState message="Unable to load predictions from the Flask API." />;
  }

  const levelClassName = predictionResult
    ? `status-pill status-${predictionResult.level.toLowerCase()}`
    : "status-pill";

  return (
    <section className="page">
      <div className="page-header">
        <p className="eyebrow">Predictions</p>
        <h2>Hybrid Model Inference</h2>
        <p className="page-copy">
          User input flows from React to Flask and returns a predicted energy value with a usage level.
        </p>
      </div>

      <div className="workspace-grid">
        <PredictionForm featureRanges={data.feature_ranges} onPrediction={setPredictionResult} />

        <div className="content-panel">
          <div className="section-heading">
            <h3>Prediction Output</h3>
            <p>Highlighted result from the `/predict` endpoint.</p>
          </div>

          {predictionResult ? (
            <div className="prediction-result">
              <div className="highlight-metric">
                <span>Predicted Energy Value</span>
                <strong>{predictionResult.prediction} kW</strong>
              </div>
              <div className={levelClassName}>{predictionResult.level}</div>
              <div className="factor-list">
                <h4>Top influencing features</h4>
                {predictionResult.top_factors.map((factor) => (
                  <div key={factor.feature} className="factor-item">
                    <span>{factor.feature}</span>
                    <strong>{factor.impact}</strong>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <div className="empty-panel">
              <p>Enter values and run a prediction to see the output card and status level.</p>
            </div>
          )}
        </div>
      </div>

      <div className="metric-grid">
        <MetricCard label="Mean Prediction" value={data.prediction_stats.mean} hint="Hybrid sample mean" />
        <MetricCard label="Min Prediction" value={data.prediction_stats.min} hint="Lowest sampled estimate" />
        <MetricCard label="Max Prediction" value={data.prediction_stats.max} hint="Highest sampled estimate" />
      </div>

      <article className="content-panel chart-panel">
        <div className="section-heading">
          <h3>Actual vs Predicted</h3>
          <p>Line chart built from backend sample data for the trained Hybrid model.</p>
        </div>
        <div className="chart-wrap">
          <ResponsiveContainer width="100%" height={320}>
            <LineChart data={lineChartData}>
              <CartesianGrid stroke="#2A2A2A" strokeDasharray="3 3" />
              <XAxis dataKey="sample" stroke="#B3B3B3" />
              <YAxis stroke="#B3B3B3" />
              <Tooltip
                contentStyle={{ background: "#1E1E1E", border: "1px solid #2F2F2F" }}
                labelStyle={{ color: "#FFFFFF" }}
              />
              <Legend />
              <Line type="monotone" dataKey="actual" stroke="#10B981" strokeWidth={2} dot={false} />
              <Line type="monotone" dataKey="predicted" stroke="#3B82F6" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </article>
    </section>
  );
}

export default PredictionsPage;
