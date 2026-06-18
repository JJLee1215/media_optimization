import { useState } from "react";
import axios from "axios";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer
} from "recharts";

const API = "http://localhost:8000";

const COLORS = [
  "#1D9E75","#534AB7","#E24B4A","#BA7517","#0F6E56",
  "#7F77DD","#EF9F27","#FAC775","#9FE1CB","#4B9FE0",
  "#E87D4C","#6CB8E8","#A85CA8","#5C8A3C","#D4A017"
];

export default function TimePanel() {
  const [file,           setFile]           = useState(null);
  const [filename,       setFilename]       = useState("");
  const [uploading,      setUploading]      = useState(false);
  const [columns,        setColumns]        = useState([]);
  const [selectedCols,   setSelectedCols]   = useState([]);
  const [batchIds,       setBatchIds]       = useState([]);
  const [selectedBatch,  setSelectedBatch]  = useState("all");
  const [rawData,        setRawData]        = useState({});
  const [timeCol,        setTimeCol]        = useState("");
  const [analyzing,      setAnalyzing]      = useState(false);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [error,          setError]          = useState("");

  // ── Upload ──────────────────────────────────────
  const handleUpload = async () => {
    if (!file) return;
    setUploading(true);
    setError("");
    try {
      const form = new FormData();
      form.append("file", file);
      const { data: up } = await axios.post(`${API}/data/upload`, form);
      const fname = up.filename;
      setFilename(fname);

      const { data } = await axios.get(`${API}/data/columns`, {
        params: { filename: fname, type: "timeseries" }
      });

      setColumns(data.columns);
      setSelectedCols(data.columns.slice(0, 5));
      setBatchIds(data.batch_ids);
      setRawData(data.data);
      setTimeCol(data.time_col);
    } catch (e) {
      setError("Upload failed.");
    } finally {
      setUploading(false);
    }
  };

  const toggleCol   = (col) => setSelectedCols(prev =>
    prev.includes(col) ? prev.filter(c => c !== col) : [...prev, col]
  );
  const selectAll   = () => setSelectedCols([...columns]);
  const deselectAll = () => setSelectedCols([]);

  // ── Build chart data ──────────────────────────
  const buildChartData = () => {
    const batches = selectedBatch === "all"
      ? batchIds.slice(0, 5)
      : [selectedBatch];

    if (!batches.length || !rawData[String(batches[0])]) return { chartData: [], lines: [] };

    if (selectedBatch === "all") {
      const firstBatch = String(batchIds[0]);
      const times = rawData[firstBatch]?.time || [];
      const chartData = times.map((t, i) => {
        const row = { time: t };
        batches.forEach(bid => {
          const bd = rawData[String(bid)];
          if (!bd) return;
          columns.forEach(col => { row[`${col}_b${bid}`] = bd[col]?.[i] ?? null; });
        });
        return row;
      });
      const lines = [];
      batches.forEach((bid) => {
        columns.forEach((col, ci) => {
          lines.push({
            key     : `${col}_b${bid}`,
            col,
            color   : COLORS[ci % COLORS.length],
            selected: selectedCols.includes(col),
          });
        });
      });
      return { chartData, lines };
    } else {
      const bd = rawData[String(selectedBatch)];
      if (!bd) return { chartData: [], lines: [] };
      const times = bd.time || [];
      const chartData = times.map((t, i) => {
        const row = { time: t };
        columns.forEach(col => { row[col] = bd[col]?.[i] ?? null; });
        return row;
      });
      const lines = columns.map((col, i) => ({
        key     : col,
        col,
        color   : COLORS[i % COLORS.length],
        selected: selectedCols.includes(col),
      }));
      return { chartData, lines };
    }
  };

  const { chartData, lines } = buildChartData();

  // Y axis domain based on selected columns only
  const getYDomain = () => {
    if (!chartData.length || !selectedCols.length) return ["auto", "auto"];
    const selectedKeys = lines.filter(l => l.selected).map(l => l.key);
    const vals = chartData.flatMap(row =>
      selectedKeys.map(k => row[k]).filter(v => v !== null && !isNaN(v))
    );
    if (!vals.length) return ["auto", "auto"];
    const min = Math.min(...vals);
    const max = Math.max(...vals);
    const pad = (max - min) * 0.1 || 0.1;
    return [parseFloat((min - pad).toFixed(4)), parseFloat((max + pad).toFixed(4))];
  };

  // ── Run Analysis ──────────────────────────────
  const handleAnalyze = async () => {
    if (!filename) return;
    setAnalyzing(true);
    try {
      const { data } = await axios.post(`${API}/data/analyze`, { filename });
      setAnalysisResult(data);
    } catch (e) {
      setError("Analysis failed.");
    } finally {
      setAnalyzing(false);
    }
  };

  return (
    <div className="panel">
      <div className="panel-header">📈 Time Series Data</div>

      {/* 1. Upload */}
      <div className="card">
        <div className="form-row">
          <input type="file" accept=".csv" onChange={e => setFile(e.target.files[0])} />
          <button className="btn btn-primary" onClick={handleUpload} disabled={!file || uploading}>
            {uploading ? "Loading..." : "Upload"}
          </button>
        </div>
        {filename && <div className="file-badge">📄 {filename}</div>}
      </div>

      {error && <div className="status-bar status-error">❌ {error}</div>}

      {columns.length > 0 && (<>

        {/* 2. Batch */}
        <div className="card">
          <h3 style={{ fontSize: 13, fontWeight: 600, marginBottom: "0.5rem" }}>Batch</h3>
          <div className="form-group">
            <select value={selectedBatch} onChange={e => setSelectedBatch(e.target.value)}>
              <option value="all">All (first 5)</option>
              {batchIds.map(bid => (
                <option key={bid} value={bid}>Batch {bid}</option>
              ))}
            </select>
          </div>
        </div>

        {/* 3. Variables */}
        <div className="card">
          <div className="card-header-row">
            <h3>Variables</h3>
            <div style={{ display: "flex", gap: "0.4rem" }}>
              <button className="btn-sm" onClick={selectAll}>All</button>
              <button className="btn-sm" onClick={deselectAll}>None</button>
            </div>
          </div>
          <div className="checkbox-grid">
            {columns.map((col, i) => (
              <label key={col} className="checkbox-item">
                <input type="checkbox" checked={selectedCols.includes(col)} onChange={() => toggleCol(col)} />
                <span className="checkbox-dot" style={{ background: COLORS[i % COLORS.length] }} />
                <span className="checkbox-label">{col}</span>
              </label>
            ))}
          </div>
        </div>

        {/* 4. Chart */}
        {chartData.length > 0 && (
          <div className="card">
            <h3 style={{ fontSize: 13, fontWeight: 600, marginBottom: "0.5rem" }}>Time Series</h3>
            <ResponsiveContainer width="100%" height={280}>
              <LineChart data={chartData} margin={{ top: 5, right: 10, bottom: 20, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis dataKey="time" tick={{ fontSize: 10 }}
                  label={{ value: timeCol, position: "insideBottom", offset: -10, fontSize: 10 }} />
                <YAxis tick={{ fontSize: 10 }} domain={getYDomain()} />
                <Tooltip />
                {lines.map(l => (
                  <Line
                    key={l.key}
                    type="monotone"
                    dataKey={l.key}
                    stroke={l.color}
                    strokeWidth={l.selected ? 2.5 : 1.0}
                    opacity={l.selected ? 1.0 : 0.15}
                    dot={false}
                    isAnimationActive={false}
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* 5. Run Analysis */}
        <div className="card">
          <button className="btn btn-primary" onClick={handleAnalyze} disabled={analyzing}>
            {analyzing ? "Analyzing..." : "Run Analysis"}
          </button>
        </div>

        {/* 6. Analysis Results */}
        {analysisResult && (
          <div className="card">
            <h3 style={{ fontSize: 13, fontWeight: 600, marginBottom: "0.75rem" }}>Analysis Results</h3>
            <div className="metrics">
              <div className="metric-box">
                <div className="label">Rows</div>
                <div className="value">{analysisResult.dimensions?.n_rows?.toLocaleString()}</div>
              </div>
              <div className="metric-box">
                <div className="label">Columns</div>
                <div className="value">{analysisResult.dimensions?.n_cols}</div>
              </div>
            </div>
            {analysisResult.plot_urls?.map(url => (
              <img key={url} src={`${API}${url}`} alt="analysis"
                style={{ width: "100%", marginTop: "1rem", borderRadius: 8 }} />
            ))}
          </div>
        )}

      </>)}
    </div>
  );
}