import { useEffect, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { fetchPredictions } from "../api";
import PredictionForm from "../components/PredictionForm";
import { ErrorState, LoadingState } from "../components/PageState";

function RecommendationsPage() {
  const [featureRanges, setFeatureRanges] = useState(null);
  const [status, setStatus] = useState("loading");
  const [predictionResult, setPredictionResult] = useState(null);

  useEffect(() => {
    let active = true;
    fetchPredictions()
      .then((response) => {
        if (!active) {
          return;
        }
        setFeatureRanges(response.feature_ranges);
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

  if (status === "loading") {
    return <LoadingState label="Loading recommendation engine..." />;
  }

  if (status === "error" || !featureRanges) {
    return <ErrorState message="Unable to load recommendation inputs from the Flask API." />;
  }

  return (
    <section className="page">
      <div className="page-header">
        <p className="eyebrow">Recommendations</p>
        <h2>Personalized Energy Saving Advice</h2>
        <p className="page-copy">
          The backend dynamically generates exactly 10 recommendations based on prediction value and hour.
        </p>
      </div>

      <div className="workspace-grid">
        <PredictionForm featureRanges={featureRanges} onPrediction={setPredictionResult} />

        <div className="content-panel">
          <div className="section-heading">
            <h3>Top 10 Recommendations</h3>
            <p>Project-specific suggestions returned from the Flask prediction API.</p>
          </div>

          {predictionResult ? (
            <div className="recommendation-panel">
              <div className="recommendation-badge">
                <span>Consumption Level</span>
                <strong>{predictionResult.level}</strong>
              </div>
              <div className="recommendation-badge">
                <span>Total Recommendations</span>
                <strong>{predictionResult.recommendations.length}</strong>
              </div>
              <ul className="recommendation-list">
                {predictionResult.recommendations.map((action, index) => (
                  <li key={action}>
                    <strong>{index + 1}.</strong> {action}
                  </li>
                ))}
              </ul>
            </div>
          ) : (
            <div className="empty-panel">
              <p>Run a prediction to generate the full recommendation list.</p>
            </div>
          )}
        </div>
      </div>

      {predictionResult ? (
        <article className="content-panel chart-panel">
          <div className="section-heading">
            <h3>Recommendation Distribution</h3>
            <p>Optional bar chart for the generated recommendation categories.</p>
          </div>
          <div className="chart-wrap">
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={predictionResult.distribution}>
                <CartesianGrid stroke="#2A2A2A" strokeDasharray="3 3" />
                <XAxis dataKey="name" stroke="#B3B3B3" angle={-12} textAnchor="end" height={72} />
                <YAxis stroke="#B3B3B3" allowDecimals={false} />
                <Tooltip
                  contentStyle={{ background: "#1E1E1E", border: "1px solid #2F2F2F" }}
                  labelStyle={{ color: "#FFFFFF" }}
                />
                <Bar dataKey="count" fill="#10B981" radius={[8, 8, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </article>
      ) : null}
    </section>
  );
}

export default RecommendationsPage;
