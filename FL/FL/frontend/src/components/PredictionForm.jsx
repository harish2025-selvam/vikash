import { useEffect, useState, useTransition } from "react";
import { requestPrediction } from "../api";

const FIELD_LABELS = {
  Voltage: "Voltage",
  Global_intensity: "Global Intensity",
  Sub_metering_1: "Sub Metering 1",
  Sub_metering_2: "Sub Metering 2",
  Sub_metering_3: "Sub Metering 3",
  hour: "Hour",
};

function buildInitialState(featureRanges) {
  return Object.fromEntries(
    Object.entries(featureRanges).map(([key, value]) => [key, value.median]),
  );
}

function PredictionForm({ featureRanges, onPrediction }) {
  const [formData, setFormData] = useState(() => buildInitialState(featureRanges));
  const [isPending, startTransition] = useTransition();

  useEffect(() => {
    setFormData(buildInitialState(featureRanges));
  }, [featureRanges]);

  function handleChange(event) {
    const { name, value } = event.target;
    setFormData((current) => ({
      ...current,
      [name]: value,
    }));
  }

  async function handleSubmit(event) {
    event.preventDefault();
    startTransition(async () => {
      const result = await requestPrediction({
        voltage: Number(formData.Voltage),
        global_intensity: Number(formData.Global_intensity),
        sub_metering_1: Number(formData.Sub_metering_1),
        sub_metering_2: Number(formData.Sub_metering_2),
        sub_metering_3: Number(formData.Sub_metering_3),
        hour: Number(formData.hour),
      });
      onPrediction(result);
    });
  }

  return (
    <form className="input-panel" onSubmit={handleSubmit}>
      <div className="section-heading">
        <h3>Consumption Input</h3>
        <p>Submit the smart-grid feature values to get a Hybrid model forecast from the Flask API.</p>
      </div>

      <div className="form-grid">
        {Object.entries(featureRanges).map(([field, range]) => (
          <label key={field} className="form-field">
            <span>{FIELD_LABELS[field]}</span>
            <input
              type="number"
              name={field}
              step={field === "hour" ? "1" : "0.1"}
              min={range.min}
              max={range.max}
              value={formData[field]}
              onChange={handleChange}
            />
            <small>
              Range: {range.min} - {range.max}
            </small>
          </label>
        ))}
      </div>

      <button className="primary-button" type="submit" disabled={isPending}>
        {isPending ? "Running prediction..." : "Run Hybrid Prediction"}
      </button>
    </form>
  );
}

export default PredictionForm;
