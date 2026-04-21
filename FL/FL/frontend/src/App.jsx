import { Suspense, lazy } from "react";
import { NavLink, Route, Routes } from "react-router-dom";
import { LoadingState } from "./components/PageState";

const DashboardPage = lazy(() => import("./pages/DashboardPage"));
const PredictionsPage = lazy(() => import("./pages/PredictionsPage"));
const ComparisonPage = lazy(() => import("./pages/ComparisonPage"));
const RecommendationsPage = lazy(() => import("./pages/RecommendationsPage"));
const AboutPage = lazy(() => import("./pages/AboutPage"));

const navItems = [
  { label: "Dashboard", path: "/" },
  { label: "Predictions", path: "/predictions" },
  { label: "Comparison", path: "/comparison" },
  { label: "Recommendations", path: "/recommendations" },
  { label: "About", path: "/about" },
];

function App() {
  return (
    <div className="app-shell">
      <header className="topbar">
        <div>
          <p className="topbar-kicker">Smart Grid Project</p>
          <h1>Personalized Energy Saving Recommendation using Federated Learning</h1>
        </div>
      </header>

      <aside className="sidebar">
        <div className="sidebar-brand">
          <span className="brand-dot" />
          FL Energy UI
        </div>
        <nav className="sidebar-nav">
          {navItems.map((item) => (
            <NavLink
              key={item.path}
              to={item.path}
              end={item.path === "/"}
              className={({ isActive }) =>
                `nav-link${isActive ? " nav-link-active" : ""}`
              }
            >
              {item.label}
            </NavLink>
          ))}
        </nav>
      </aside>

      <main className="content">
        <Suspense fallback={<LoadingState label="Loading page..." />}>
          <Routes>
            <Route path="/" element={<DashboardPage />} />
            <Route path="/predictions" element={<PredictionsPage />} />
            <Route path="/comparison" element={<ComparisonPage />} />
            <Route path="/recommendations" element={<RecommendationsPage />} />
            <Route path="/about" element={<AboutPage />} />
          </Routes>
        </Suspense>
      </main>
    </div>
  );
}

export default App;
