import { useEffect, useState } from "react";
import MetricCard from "../components/MetricCard";
import { ErrorState, LoadingState } from "../components/PageState";
import { fetchDashboard, fetchMetrics } from "../api";

function DashboardPage() {
  const [data, setData] = useState(null);
  const [status, setStatus] = useState("loading");

  useEffect(() => {
    let active = true;
    Promise.all([fetchDashboard(), fetchMetrics()])
      .then(([dashboardResponse, metricsResponse]) => {
        if (!active) {
          return;
        }
        setData({
          ...dashboardResponse,
          metrics: metricsResponse,
        });
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
    return <LoadingState label="Loading dashboard metrics..." />;
  }

  if (status === "error" || !data) {
    return <ErrorState message="Unable to load dashboard data from the Flask API." />;
  }

  const metrics = [
    { label: "MSE", value: data.metrics.mse },
    { label: "RMSE", value: data.metrics.rmse },
    { label: "MAE", value: data.metrics.mae },
    { label: "R² Score", value: data.metrics.r2 },
  ];

  return (
    <section className="page">
      <div className="page-header">
        <p className="eyebrow">Dashboard</p>
        <h2>{data.title}</h2>
        <p className="page-copy">
          Final Model: <span className="accent-text">{data.final_model}</span>
        </p>
      </div>

      <div className="hero-panel">
        <div>
          <span className="chip">Best RMSE: {data.best_model}</span>
          <h3>Hybrid federated regression for smart-grid energy forecasting</h3>
          <p>
            The backend reproduces the notebook workflow with 10 clients and 30 training rounds
            using the household energy dataset.
          </p>
        </div>
        <div className="hero-meta">
          <div>
            <span>Rows</span>
            <strong>{data.dataset_shape.rows.toLocaleString()}</strong>
          </div>
          <div>
            <span>Columns</span>
            <strong>{data.dataset_shape.columns}</strong>
          </div>
          <div>
            <span>Clients</span>
            <strong>{data.training.clients}</strong>
          </div>
          <div>
            <span>Rounds</span>
            <strong>{data.training.rounds}</strong>
          </div>
        </div>
      </div>

      <div className="metric-grid">
        {metrics.map((metric) => (
          <MetricCard
            key={metric.label}
            label={metric.label}
            value={metric.value}
            hint="Hybrid model evaluation on the project dataset"
          />
        ))}
      </div>

      <div className="info-grid">
        <article className="content-panel">
          <div className="section-heading">
            <h3>Dataset Window</h3>
            <p>Captured from the supplied energy consumption CSV.</p>
          </div>
          <dl className="detail-list">
            <div>
              <dt>Start</dt>
              <dd>{data.dataset_summary.time_span.start}</dd>
            </div>
            <div>
              <dt>End</dt>
              <dd>{data.dataset_summary.time_span.end}</dd>
            </div>
            <div>
              <dt>Target Mean</dt>
              <dd>{data.dataset_summary.target_mean}</dd>
            </div>
            <div>
              <dt>Peak Power</dt>
              <dd>{data.dataset_summary.target_peak}</dd>
            </div>
          </dl>
        </article>

        <article className="content-panel">
          <div className="section-heading">
            <h3>Feature Set</h3>
            <p>Model inputs taken directly from the notebook pipeline.</p>
          </div>
          <ul className="feature-list">
            {data.training.feature_set.map((feature) => (
              <li key={feature}>{feature}</li>
            ))}
          </ul>
        </article>
      </div>
    </section>
  );
}

export default DashboardPage;
