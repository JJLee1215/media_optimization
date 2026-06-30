import { useState, useEffect } from "react";
import axios from "axios";

const API = "http://localhost:8000";

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

  const [availableModels, setAvailableModels] = useState([]);
  const [modelsLoading,   setModelsLoading]   = useState(true);
  const [model,   setModel]   = useState("");
  const [values,  setValues]  = useState(initValues);
  const [loading, setLoading] = useState(false);
  const [result,  setResult]  = useState(null);
  const [error,   setError]   = useState("");

  // ── 학습된 모델만 가져오기 ──
  useEffect(() => {
    fetchAvailableModels();
  }, []);

  const fetchAvailableModels = async () => {
    setModelsLoading(true);
    try {
      const { data } = await axios.get(`${API}/train/models`);
      // static + static_time 그룹에서 has_model=true인 것만 추출
      const trained = [
        ...data.static,
        ...data.static_time,
      ].filter(m => m.has_model);

      setAvailableModels(trained);
      if (trained.length > 0) {
        setModel(trained[0].id);
      }
    } catch (e) {
      setError("Failed to load model list.");
    } finally {
      setModelsLoading(false);
    }
  };

  const handleChange = (key, val) => {
    setValues(prev => ({ ...prev, [key]: parseFloat(val) || 0 }));
  };

  const handlePredict = async () => {
    if (!model) return;
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

        {modelsLoading ? (
          <p style={{ color: "#888", fontSize: 13 }}>Loading available models...</p>
        ) : availableModels.length === 0 ? (
          <div className="status-bar status-error">
            ❌ No trained models found. Go to Model Train and train a model first.
          </div>
        ) : (
          <div className="form-group" style={{ maxWidth: 300 }}>
            <label>Model</label>
            <select value={model} onChange={e => setModel(e.target.value)}>
              {availableModels.map(m => (
                <option key={m.id} value={m.id}>
                  {m.name} — {m.desc}
                  {m.result?.r2 !== undefined ? `  (R² ${m.result.r2.toFixed(3)})` : ""}
                </option>
              ))}
            </select>
          </div>
        )}
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
            disabled={loading || !model || availableModels.length === 0}
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