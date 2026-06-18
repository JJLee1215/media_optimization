import { useState } from "react";
import axios from "axios";

const API = "http://localhost:8000";

const MODELS = [
  { value: "gaussian_process", label: "Gaussian Process" },
  { value: "random_forest",    label: "Random Forest" },
  { value: "xgboost",          label: "XGBoost" },
  { value: "mlp",              label: "MLP" },
  { value: "static_time_gnn",  label: "StaticTimeGNN" },
];

const COMPONENTS = [
  { key: "Glucose_0",    label: "Glucose",    unit: "g/L",    default: 4.5   },
  { key: "Glutamine_0",  label: "Glutamine",  unit: "g/L",    default: 2.0   },
  { key: "Asparagine_0", label: "Asparagine", unit: "mg/L",   default: 100.0 },
  { key: "Lactate_0",    label: "Lactate",    unit: "g/L",    default: 0.1   },
  { key: "Ammonia_0",    label: "Ammonia",    unit: "mmol/L", default: 0.05  },
  { key: "Cu_0",         label: "Cu²⁺",       unit: "mg/L",   default: 0.03  },
  { key: "Zn_0",         label: "Zn²⁺",       unit: "mg/L",   default: 0.05  },
  { key: "Mn_0",         label: "Mn²⁺",       unit: "mg/L",   default: 0.02  },
  { key: "Fe_0",         label: "Fe³⁺",       unit: "mg/L",   default: 0.10  },
];

export default function PredictPage() {
  const initValues = Object.fromEntries(COMPONENTS.map(c => [c.key, c.default]));

  const [model,   setModel]   = useState("gaussian_process");
  const [values,  setValues]  = useState(initValues);
  const [loading, setLoading] = useState(false);
  const [result,  setResult]  = useState(null);
  const [error,   setError]   = useState("");

  const handleChange = (key, val) => {
    setValues(prev => ({ ...prev, [key]: parseFloat(val) || 0 }));
  };

  const handlePredict = async () => {
    setLoading(true);
    setError("");
    setResult(null);
    try {
      const { data } = await axios.post(`${API}/predict`, {
        model,
        inputs: values,
      });
      setResult(data);
    } catch (e) {
      setError(
        e.response?.data?.detail ||
        "Prediction failed. Make sure the model is trained."
      );
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setValues(initValues);
    setResult(null);
    setError("");
  };

  return (
    <div>
      <h1 className="page-title">Prediction</h1>
      <p className="page-desc">
        Enter initial media composition → predict titer (g/L) and viability.
      </p>

      {/* Model select */}
      <div className="card">
        <h2>Select Model</h2>
        <div className="form-group" style={{ maxWidth: 300 }}>
          <label>Model</label>
          <select value={model} onChange={e => setModel(e.target.value)}>
            {MODELS.map(m => (
              <option key={m.value} value={m.value}>{m.label}</option>
            ))}
          </select>
        </div>
      </div>

      {/* Media composition inputs */}
      <div className="card">
        <h2>Initial Media Composition</h2>
        <div className="input-grid">
          {COMPONENTS.map(c => (
            <div className="form-group" key={c.key}>
              <label>
                {c.label}
                <span style={{ color: "#aaa", fontWeight: 400 }}> ({c.unit})</span>
              </label>
              <input
                type="number"
                step="0.01"
                value={values[c.key]}
                onChange={e => handleChange(c.key, e.target.value)}
              />
            </div>
          ))}
        </div>
        <div style={{ display: "flex", gap: "0.75rem", marginTop: "1rem" }}>
          <button
            className="btn btn-primary"
            onClick={handlePredict}
            disabled={loading}
          >
            {loading ? "Predicting..." : "Predict"}
          </button>
          <button
            className="btn"
            style={{ background: "#eee", color: "#555" }}
            onClick={handleReset}
          >
            Reset
          </button>
        </div>
      </div>

      {error && <div className="status-bar status-error">❌ {error}</div>}

      {/* Result */}
      {result && (
        <div className="card">
          <h2>Prediction Result</h2>
          <div className="result-box">
            <div className="model-tag">{result.model.replace(/_/g, " ").toUpperCase()}</div>
            <div className="result-row">
              <div className="result-item">
                <div className="result-label">Titer</div>
                <div className="result-value">
                  {result.titer_pred}
                  <span className="result-unit"> g/L</span>
                </div>
                {result.sigma !== null && result.sigma !== undefined && (
                  <div className="result-sigma">± {result.sigma} g/L (σ)</div>
                )}
              </div>
              {result.viab_pred !== null && result.viab_pred !== undefined && (
                <div className="result-item">
                  <div className="result-label">Viability</div>
                  <div className="result-value">
                    {(result.viab_pred * 100).toFixed(1)}
                    <span className="result-unit"> %</span>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}