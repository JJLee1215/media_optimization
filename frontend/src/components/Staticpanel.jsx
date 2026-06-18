import { useState } from "react";
import axios from "axios";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Cell
} from "recharts";

const API = "http://localhost:8000";

const COLORS = [
  "#1D9E75","#534AB7","#E24B4A","#BA7517","#0F6E56",
  "#7F77DD","#EF9F27","#FAC775","#9FE1CB","#4B9FE0",
  "#E87D4C","#6CB8E8","#A85CA8","#5C8A3C","#D4A017"
];

export default function StaticPanel() {
  const [file,           setFile]           = useState(null);
  const [filename,       setFilename]       = useState("");
  const [uploading,      setUploading]      = useState(false);
  const [columns,        setColumns]        = useState([]);
  const [selectedCols,   setSelectedCols]   = useState([]);
  const [batchIds,       setBatchIds]       = useState([]);
  const [selectedBatch,  setSelectedBatch]  = useState("all");
  const [rawData,        setRawData]        = useState({});   // {col: [values]} or {batch: {col: [values]}}
  const [batchCol,       setBatchCol]       = useState("");
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
        params: { filename: fname, type: "static" }
      });

      // Separate target cols from feature cols
      const targetCols = ["titer_final", "viab_final"];
      const featureCols = data.columns.filter(c => !targetCols.includes(c));

      setColumns(featureCols);
      setSelectedCols(featureCols);
      setRawData(data.data);

      // Detect batch info
      const bc = data.batch_col || "batch_id";
      setBatchCol(bc);
      if (data.data[bc]) {
        const ids = [...new Set(data.data[bc])].sort((a, b) => a - b);
        setBatchIds(ids);
      }
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

  // ── Build bar chart data ─────────────────────────
  const barData = columns
    .filter(col => rawData[col])
    .map((col, i) => {
      let vals;
      if (selectedBatch === "all") {
        vals = rawData[col].filter(v => v !== null && !isNaN(v));
      } else {
        // Filter by batch
        const batchIndex = (rawData[batchCol] || [])
          .map((b, idx) => (String(b) === String(selectedBatch) ? idx : -1))
          .filter(idx => idx !== -1);
        vals = batchIndex.map(idx => rawData[col][idx]).filter(v => v !== null && !isNaN(v));
      }
      const mean = vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : 0;
      return {
        name    : col,
        value   : parseFloat(mean.toFixed(4)),
        selected: selectedCols.includes(col),
        color   : COLORS[i % COLORS.length],
      };
    });

  // Y axis domain based on selected columns only
  const selectedVals = barData.filter(d => d.selected).map(d => d.value);
  const yMin = selectedVals.length ? Math.min(...selectedVals) * 0.9 : 0;
  const yMax = selectedVals.length ? Math.max(...selectedVals) * 1.1 : 1;

  // ── Run Analysis ─────────────────────────────────
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
      <div className="panel-header">📊 Static Data</div>

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
              <option value="all">All</option>
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
        <div className="card">
          <h3 style={{ fontSize: 13, fontWeight: 600, marginBottom: "0.5rem" }}>Mean Values</h3>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={barData} margin={{ top: 5, right: 10, bottom: 70, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis dataKey="name" tick={{ fontSize: 10 }} angle={-35} textAnchor="end" interval={0} />
              <YAxis tick={{ fontSize: 10 }} domain={[yMin, yMax]} />
              <Tooltip />
              <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                {barData.map((d, i) => (
                  <Cell key={i} fill={d.color} opacity={d.selected ? 1.0 : 0.15} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

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
                <div className="value">{analysisResult.dimensions?.n_rows}</div>
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