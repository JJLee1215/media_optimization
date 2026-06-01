import { useState, useEffect, useRef } from "react";
import axios from "axios";
import {
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid,
  Tooltip, ReferenceLine, ResponsiveContainer
} from "recharts";

const API = "http://localhost:8000";

const MODELS = [
  { value: "gp",            label: "Gaussian Process (GP)" },
  { value: "xgboost",       label: "XGBoost" },
  { value: "random_forest", label: "Random Forest" },
  { value: "mlp",           label: "MLP (Neural Network)" },
];

export default function TrainPage() {
  const [datasets, setDatasets] = useState([]);
  const [dataset, setDataset]   = useState("batch_table.csv");
  const [model, setModel]       = useState("gp");
  const [status, setStatus]     = useState(null);
  const [message, setMessage]   = useState("");
  const [result, setResult]     = useState(null);
  const pollRef = useRef(null);

  useEffect(() => {
    axios.get(`${API}/datasets`)
      .then(r => setDatasets(r.data.datasets))
      .catch(() => {});
    return () => clearInterval(pollRef.current);
  }, []);

  const handleTrain = async () => {
    setStatus("running");
    setMessage("학습 시작 중...");
    setResult(null);

    await axios.post(`${API}/train`, { dataset, model, test_size: 0.2 });

    pollRef.current = setInterval(async () => {
      const { data } = await axios.get(`${API}/train/status`);
      setMessage(data.message);
      if (data.status === "done") {
        clearInterval(pollRef.current);
        setStatus("done");
        setResult(data.result);
      } else if (data.status === "error") {
        clearInterval(pollRef.current);
        setStatus("error");
      }
    }, 1000);
  };

  const scatterData = result
    ? result.y_test.map((actual, i) => ({ actual, predicted: result.y_pred[i] }))
    : [];

  const minVal = scatterData.length
    ? Math.floor(Math.min(...scatterData.map(d => Math.min(d.actual, d.predicted))) - 1)
    : 0;
  const maxVal = scatterData.length
    ? Math.ceil(Math.max(...scatterData.map(d => Math.max(d.actual, d.predicted))) + 1)
    : 50;

  return (
    <div>
      <div className="card">
        <h2>Training</h2>
        <div className="form-row">
          <div className="form-group">
            <label>Dataset</label>
            <select value={dataset} onChange={e => setDataset(e.target.value)}>
              {datasets.length === 0
                ? <option value="batch_table.csv">batch_table.csv</option>
                : datasets.map(d => <option key={d} value={d}>{d}</option>)
              }
            </select>
          </div>
          <div className="form-group">
            <label>Model</label>
            <select value={model} onChange={e => setModel(e.target.value)}>
              {MODELS.map(m => <option key={m.value} value={m.value}>{m.label}</option>)}
            </select>
          </div>
        </div>
        <button
          className="btn btn-primary"
          onClick={handleTrain}
          disabled={status === "running"}
        >
          {status === "running" ? "학습 중..." : "Train"}
        </button>
      </div>

      {status && (
        <div className={`status-bar status-${status}`}>
          {status === "running" && "⏳ "}
          {status === "done"    && "✅ "}
          {status === "error"   && "❌ "}
          {message}
        </div>
      )}

      {result && (
        <>
          <div className="card">
            <h2>결과</h2>
            <div className="metrics">
              <div className="metric-box">
                <div className="label">RMSE</div>
                <div className="value">{result.rmse}</div>
              </div>
              <div className="metric-box">
                <div className="label">R²</div>
                <div className="value">{result.r2}</div>
              </div>
              <div className="metric-box">
                <div className="label">CV R² mean</div>
                <div className="value">{result.cv_r2_mean}</div>
              </div>
              <div className="metric-box">
                <div className="label">CV R² std</div>
                <div className="value">{result.cv_r2_std}</div>
              </div>
              <div className="metric-box">
                <div className="label">Train / Test</div>
                <div className="value" style={{fontSize:16}}>{result.n_train} / {result.n_test}</div>
              </div>
            </div>
          </div>

          <div className="card">
            <h2>Predicted vs Actual</h2>
            <ResponsiveContainer width="100%" height={320}>
              <ScatterChart margin={{ top: 10, right: 20, bottom: 20, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0"/>
                <XAxis dataKey="actual" name="Actual" type="number"
                  domain={[minVal, maxVal]}
                  label={{ value: "Actual titer", position: "insideBottom", offset: -10 }}/>
                <YAxis dataKey="predicted" name="Predicted" type="number"
                  domain={[minVal, maxVal]}
                  label={{ value: "Predicted", angle: -90, position: "insideLeft" }}/>
                <Tooltip formatter={(v) => v.toFixed(3)}/>
                <ReferenceLine
                  segment={[{x: minVal, y: minVal}, {x: maxVal, y: maxVal}]}
                  stroke="#e55" strokeDasharray="4 4" strokeWidth={1.5}/>
                <Scatter data={scatterData} fill="#4f8ef7" opacity={0.7}/>
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </>
      )}
    </div>
  );
}