import { useEffect, useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { fetchComparison } from "../api";
import { ErrorState, LoadingState } from "../components/PageState";

const MODEL_COLORS = {
  FedAvg: "#3B82F6",
  FedProx: "#10B981",
  Hybrid: "#60A5FA",
};

function ComparisonPage() {
  const [data, setData] = useState(null);
  const [status, setStatus] = useState("loading");

  useEffect(() => {
    let active = true;
    fetchComparison()
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

  const lossSeries = useMemo(() => {
    if (!data) {
      return [];
    }

    return Array.from({ length: data.hybrid_loss.length }, (_, index) => ({
      round: index + 1,
      FedAvg: data.fedavg_loss[index],
      FedProx: data.fedprox_loss[index],
      Hybrid: data.hybrid_loss[index],
    }));
  }, [data]);

  const finalLossBars = useMemo(() => {
    if (!data) {
      return [];
    }

    return Object.entries(data.final_losses).map(([model, loss]) => ({
      model,
      loss,
    }));
  }, [data]);

  if (status === "loading") {
    return <LoadingState label="Loading algorithm comparison..." />;
  }

  if (status === "error" || !data) {
    return <ErrorState message="Unable to load comparison data from the Flask API." />;
  }

  return (
    <section className="page">
      <div className="page-header">
        <p className="eyebrow">Comparison</p>
        <h2>Federated Learning Comparison</h2>
        <p className="page-copy">
          Loss curves and final loss values for FedAvg, FedProx, and the Hybrid model.
        </p>
      </div>

      <div className="content-panel highlight-panel">
        <span className="chip">Best Model</span>
        <h3>{data.best_model}</h3>
        <p>The lowest final loss is used to highlight the strongest federated training strategy.</p>
      </div>

      <div className="chart-grid">
        <article className="content-panel chart-panel">
          <div className="section-heading">
            <h3>Loss vs Rounds</h3>
            <p>Line chart of convergence across 30 federated rounds.</p>
          </div>
          <div className="chart-wrap">
            <ResponsiveContainer width="100%" height={320}>
              <LineChart data={lossSeries}>
                <CartesianGrid stroke="#2A2A2A" strokeDasharray="3 3" />
                <XAxis dataKey="round" stroke="#B3B3B3" />
                <YAxis stroke="#B3B3B3" />
                <Tooltip
                  contentStyle={{ background: "#1E1E1E", border: "1px solid #2F2F2F" }}
                  labelStyle={{ color: "#FFFFFF" }}
                />
                <Line type="monotone" dataKey="FedAvg" stroke={MODEL_COLORS.FedAvg} strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="FedProx" stroke={MODEL_COLORS.FedProx} strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="Hybrid" stroke={MODEL_COLORS.Hybrid} strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </article>

        <article className="content-panel chart-panel">
          <div className="section-heading">
            <h3>Final Loss Comparison</h3>
            <p>Bar chart of the final loss for each model.</p>
          </div>
          <div className="chart-wrap">
            <ResponsiveContainer width="100%" height={320}>
              <BarChart data={finalLossBars}>
                <CartesianGrid stroke="#2A2A2A" strokeDasharray="3 3" />
                <XAxis dataKey="model" stroke="#B3B3B3" />
                <YAxis stroke="#B3B3B3" />
                <Tooltip
                  contentStyle={{ background: "#1E1E1E", border: "1px solid #2F2F2F" }}
                  labelStyle={{ color: "#FFFFFF" }}
                />
                <Bar dataKey="loss" fill="#3B82F6" radius={[8, 8, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </article>
      </div>
    </section>
  );
}

export default ComparisonPage;
