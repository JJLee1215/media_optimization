import { useState, useEffect, useRef } from "react";
import axios from "axios";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Cell
} from "recharts";

const API = "http://localhost:8000";

const MODEL_GROUPS = [
  { value: "all",          label: "All Models" },
  { value: "static",       label: "Static  (GP, RF, XGBoost, MLP)" },
  { value: "time",         label: "Time Series  (RNN, LSTM, Transformer)" },
  { value: "static_time",  label: "StaticTimeGNN" },
];

const COLORS = {
  gaussian_process : "#1D9E75",
  random_forest    : "#0F6E56",
  xgboost          : "#534AB7",
  mlp              : "#7F77DD",
  rnn              : "#BA7517",
  lstm             : "#EF9F27",
  transformer      : "#FAC775",
  static_time_gnn  : "#E24B4A",
};

export default function TrainPage() {
  const [group,   setGroup]   = useState("static");
  const [status,  setStatus]  = useState("idle");
  const [message, setMessage] = useState("");
  const [results, setResults] = useState(null);
  const [chartUrl, setChartUrl] = useState(null);
  const pollRef = useRef(null);

  useEffect(() => () => clearInterval(pollRef.current), []);

  const handleTrain = async () => {
    setStatus("running");
    setMessage("Training started...");
    setResults(null);
    setChartUrl(null);

    await axios.post(`${API}/train`, { model: group });

    pollRef.current = setInterval(async () => {
      const { data } = await axios.get(`${API}/train/status`);
      setMessage(data.message);

      if (data.status === "done") {
        clearInterval(pollRef.current);
        setStatus("done");
        setResults(data.result);
        // fetch compare chart
        const cmp = await axios.get(`${API}/compare?mode=train`);
        setChartUrl(cmp.data.chart_url);
      } else if (data.status === "error") {
        clearInterval(pollRef.current);
        setStatus("error");
      }
    }, 1500);
  };

  // Build bar chart data from results
  const barData = results
    ? Object.entries(results)
        .filter(([, r]) => r && (r.rmse || r.titer_rmse))
        .map(([name, r]) => ({
          name : name.replace(/_/g, " "),
          rmse : r.rmse ?? r.titer_rmse,
          color: COLORS[name] ?? "#888",
        }))
        .sort((a, b) => a.rmse - b.rmse)
    : [];

  return (
    <div>
      <h1 className="page-title">Model Train</h1>

      {/* Controls */}
      <div className="card">
        <h2>Select Model Group</h2>
        <div className="form-row">
          <div className="form-group">
            <label>Model Group</label>
            <select value={group} onChange={e => setGroup(e.target.value)}>
              {MODEL_GROUPS.map(g => (
                <option key={g.value} value={g.value}>{g.label}</option>
              ))}
            </select>
          </div>
          <button
            className="btn btn-primary"
            onClick={handleTrain}
            disabled={status === "running"}
            style={{ alignSelf: "flex-end" }}
          >
            {status === "running" ? "Training..." : "Train"}
          </button>
        </div>
      </div>

      {/* Status */}
      {status !== "idle" && (
        <div className={`status-bar status-${status}`}>
          {status === "running" && "⏳ "}
          {status === "done"    && "✅ "}
          {status === "error"   && "❌ "}
          {message}
        </div>
      )}

      {/* RMSE bar chart (inline) */}
      {barData.length > 0 && (
        <div className="card">
          <h2>RMSE Comparison</h2>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={barData} margin={{ top: 10, right: 20, bottom: 40, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis dataKey="name" tick={{ fontSize: 11 }} angle={-20} textAnchor="end" />
              <YAxis label={{ value: "RMSE", angle: -90, position: "insideLeft" }} />
              <Tooltip formatter={v => v.toFixed(4)} />
              <Bar dataKey="rmse" radius={[4, 4, 0, 0]}>
                {barData.map((d, i) => (
                  <Cell key={i} fill={d.color} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Result table */}
      {results && (
        <div className="card">
          <h2>Results</h2>
          <table className="result-table">
            <thead>
              <tr>
                <th>Model</th>
                <th>RMSE</th>
                <th>R²</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(results)
                .filter(([, r]) => r && !r.error)
                .sort((a, b) => (a[1].rmse ?? a[1].titer_rmse) - (b[1].rmse ?? b[1].titer_rmse))
                .map(([name, r]) => (
                  <tr key={name}>
                    <td>{name.replace(/_/g, " ")}</td>
                    <td>{(r.rmse ?? r.titer_rmse ?? "—").toFixed?.(4) ?? "—"}</td>
                    <td>{r.r2?.toFixed(4) ?? "—"}</td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}