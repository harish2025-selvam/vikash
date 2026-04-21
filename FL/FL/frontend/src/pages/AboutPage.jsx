import { useEffect, useState } from "react";
import { fetchAbout } from "../api";
import { ErrorState, LoadingState } from "../components/PageState";

function AboutPage() {
  const [data, setData] = useState(null);
  const [status, setStatus] = useState("loading");

  useEffect(() => {
    let active = true;
    fetchAbout()
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

  if (status === "loading") {
    return <LoadingState label="Loading project summary..." />;
  }

  if (status === "error" || !data) {
    return <ErrorState message="Unable to load About data from the Flask API." />;
  }

  return (
    <section className="page">
      <div className="page-header">
        <p className="eyebrow">About</p>
        <h2>Project Context</h2>
        <p className="page-copy">{data.title}</p>
      </div>

      <div className="info-grid">
        <article className="content-panel">
          <div className="section-heading">
            <h3>Project Description</h3>
            <p>{data.summary}</p>
          </div>
          <ul className="feature-list">
            <li>Federated learning workflow for smart-grid energy forecasting</li>
            <li>Hybrid training built on a Ridge Regression model</li>
            <li>Flask backend for API-based prediction and metrics</li>
            <li>React frontend for visualization and interaction</li>
          </ul>
        </article>

        <article className="content-panel">
          <div className="section-heading">
            <h3>Technologies Used</h3>
            <p>Focused tools used in this project.</p>
          </div>
          <ul className="feature-list">
            <li>Federated Learning</li>
            <li>Ridge Regression</li>
            <li>Flask</li>
            <li>React</li>
          </ul>
        </article>
      </div>
    </section>
  );
}

export default AboutPage;
