const API_BASE_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:5000/api";

async function apiRequest(path, options = {}) {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
    ...options,
  });

  if (!response.ok) {
    throw new Error(`Request failed: ${response.status}`);
  }

  return response.json();
}

export const fetchDashboard = () => apiRequest("/dashboard");
export const fetchMetrics = () => apiRequest("/metrics");
export const fetchComparison = () => apiRequest("/comparison");
export const fetchPredictions = () => apiRequest("/predictions");
export const fetchAbout = () => apiRequest("/about");

export const requestPrediction = (payload) =>
  apiRequest("/predict", {
    method: "POST",
    body: JSON.stringify(payload),
  });
