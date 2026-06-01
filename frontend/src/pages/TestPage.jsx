import { useState } from "react";
import axios from "axios";

const API = "http://localhost:8000";

const MODELS = [
  { value: "gp",            label: "Gaussian Process (GP)" },
  { value: "xgboost",       label: "XGBoost" },
  { value: "random_forest", label: "Random Forest" },
  { value: "mlp",           label: "MLP (Neural Network)" },
];

const COMPONENTS = [
  { key: "Aeration rate",                   label: "Aeration rate",         unit: "L/h",  default: 30.0  },
  { key: "Agitator RPM",                    label: "Agitator RPM",          unit: "RPM",  default: 100.0 },
  { key: "Sugar feed rate",                 label: "Sugar feed rate",       unit: "L/h",  default: 8.0   },
  { key: "Acid flow rate",                  label: "Acid flow rate",        unit: "L/h",  default: 0.0   },
  { key: "Base flow rate",                  label: "Base flow rate",        unit: "L/h",  default: 30.0  },
  { key: "Heating/cooling water flow rate", label: "Heating/cooling water", unit: "L/h",  default: 10.0  },
  { key: "Heating water flow rate",         label: "Heating water",         unit: "L/h",  default: 0.0   },
  { key: "Water for injection/dilution",    label: "Water injection",       unit: "L/h",  default: 0.0   },
  { key: "PAA flow",                        label: "PAA flow",              unit: "L/h",  default: 5.0   },
  { key: "Oil flow",                        label: "Oil flow",              unit: "L/h",  default: 22.0  },
];

export default function TestPage() {
  const initValues = Object.fromEntries(
    COMPONENTS.map(c => [c.key, c.default])
  );

  const [model, setModel]     = useState("gp");
  const [values, setValues]   = useState(initValues);
  const [loading, setLoading] = useState(false);
  const [result, setResult]   = useState(null);
  const [error, setError]     = useState("");

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
        components: values,
      });
      if (data.error) { setError(data.error); }
      else { setResult(data); }
    } catch (e) {
      setError("API 연결 실패. 모델이 학습되었는지 확인해주세요.");
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
      <div className="card">
        <h2>배지 조성 입력</h2>

        <div className="form-row" style={{ marginBottom: "1.25rem" }}>
          <div className="form-group" style={{ maxWidth: 260 }}>
            <label>Model</label>
            <select value={model} onChange={e => setModel(e.target.value)}>
              {MODELS.map(m => <option key={m.value} value={m.value}>{m.label}</option>)}
            </select>
          </div>
        </div>

        <div className="input-grid">
          {COMPONENTS.map(c => (
            <div className="form-group" key={c.key}>
              <label>
                {c.label} <span style={{ color: "#aaa", fontWeight: 400 }}>({c.unit})</span>
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

        <div style={{ display: "flex", gap: "0.75rem" }}>
          <button className="btn btn-primary" onClick={handlePredict} disabled={loading}>
            {loading ? "예측 중..." : "Predict"}
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

      {error && (
        <div className="status-bar status-error">❌ {error}</div>
      )}

      {result && (
        <div className="card">
          <h2>예측 결과</h2>
          <div className="result-box">
            <div className="model-tag">{result.model.toUpperCase()}</div>
            <div className="titer">
              {result.titer_pred}
              <span style={{ fontSize: 18, color: "#888" }}> g/L</span>
            </div>
            {result.sigma !== null && (
              <div className="sigma">불확실성 σ = ±{result.sigma} g/L</div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}