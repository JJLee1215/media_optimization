import { useState, useRef } from "react";
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
  const [filename,       setFilename]       = useState("");
  const [uploading,      setUploading]      = useState(false);
  const [columns,        setColumns]        = useState([]);
  const [selectedCols,   setSelectedCols]   = useState([]);
  const [batchIds,       setBatchIds]       = useState([]);
  const [selectedBatch,  setSelectedBatch]  = useState("all");
  const [rawData,        setRawData]        = useState({});
  const [batchCol,       setBatchCol]       = useState("");
  const [analyzing,      setAnalyzing]      = useState(false);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [error,          setError]          = useState("");
  const fileInputRef = useRef(null);

  const handleFileChange = async (e) => {
    const f = e.target.files[0];
    if (!f) return;
    setUploading(true);
    setError("");
    setAnalysisResult(null);
    setColumns([]);
    try {
      const form = new FormData();
      form.append("file", f);
      const { data: up } = await axios.post(`${API}/data/upload`, form);
      const fname = up.filename;
      setFilename(fname);

      const { data } = await axios.get(`${API}/data/columns`, {
        params: { filename: fname, type: "static" }
      });

      setColumns(data.columns);
      setSelectedCols(data.columns);
      setRawData(data.data);

      const bc = data.batch_col || "Batch_ID";
      setBatchCol(bc);
      setBatchIds(data.batch_ids || []);
    } catch (e) {
      setError("Upload failed.");
    } finally {
      setUploading(false);
    }
  };

  const handleClearFile = () => {
    setFilename("");
    setColumns([]);
    setSelectedCols([]);
    setBatchIds([]);
    setSelectedBatch("all");
    setRawData({});
    setAnalysisResult(null);
    setError("");
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

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

  const toggleCol   = (col) => setSelectedCols(prev =>
    prev.includes(col) ? prev.filter(c => c !== col) : [...prev, col]
  );
  const selectAll   = () => setSelectedCols([...columns]);
  const deselectAll = () => setSelectedCols([]);

  const barData = columns
    .filter(col => rawData[col])
    .map((col, i) => {
      let vals;
      if (selectedBatch === "all") {
        vals = rawData[col].filter(v => v !== null && !isNaN(v));
      } else {
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

  const selectedVals = barData.filter(d => d.selected).map(d => d.value);
  const yMin = selectedVals.length ? Math.min(...selectedVals) * 0.9 : 0;
  const yMax = selectedVals.length ? Math.max(...selectedVals) * 1.1 : 1;

  return (
    <div className="panel">

      {/* ── 상단 가로 바 ── */}
      <div className="upload-bar">
        <div className="upload-bar-item upload-btn-wrap">
          <span className="upload-bar-label">Upload</span>
          {!filename ? (
            <>
              <input
                ref={fileInputRef}
                type="file"
                accept=".csv"
                id="static-file-input"
                style={{ display: "none" }}
                onChange={handleFileChange}
                disabled={uploading}
              />
              <label htmlFor="static-file-input" className="file-upload-btn">
                {uploading ? "Uploading..." : "Choose file"}
              </label>
            </>
          ) : (
            <div className="file-uploaded-row">
              <span className="upload-bar-value" style={{ fontSize: 12 }}>📄 {filename}</span>
              <button className="file-clear-btn" onClick={handleClearFile}>✕</button>
            </div>
          )}
        </div>
        <div className="upload-bar-item">
          <span className="upload-bar-label">Batches</span>
          <span className="upload-bar-value">{batchIds.length || "—"}</span>
        </div>
        <div className="upload-bar-item">
          <span className="upload-bar-label">Columns</span>
          <span className="upload-bar-value">{columns.length || "—"}</span>
        </div>
        <div className="upload-bar-item">
          <span className="upload-bar-label">Analysis</span>
          <button
            className="btn btn-primary"
            onClick={handleAnalyze}
            disabled={analyzing || !filename}
            style={{ fontSize: 12, padding: "0.4rem 0.8rem" }}
          >
            {analyzing ? "⏳ Running..." : "▶ Run Analysis"}
          </button>
        </div>
      </div>

      {error && <div className="status-bar status-error">❌ {error}</div>}

      {columns.length > 0 && (<>

        {/* ── Batch + Variables ── */}
        <div className="card">
          <div style={{ display: "flex", gap: "1rem", alignItems: "flex-start", flexWrap: "wrap" }}>

            <div style={{ minWidth: 160 }}>
              <div className="card-header-row" style={{ marginBottom: "0.4rem" }}>
                <h3>Batch</h3>
              </div>
              <select
                value={selectedBatch}
                onChange={e => setSelectedBatch(e.target.value)}
                style={{ padding: "0.4rem 0.6rem", border: "1px solid #ddd", borderRadius: 8, fontSize: 13 }}
              >
                <option value="all">All</option>
                {batchIds.map(bid => (
                  <option key={bid} value={String(bid)}>Batch {bid}</option>
                ))}
              </select>
            </div>

            <div style={{ flex: 1, minWidth: 200 }}>
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

          </div>
        </div>

        {/* ── Mean Values 차트 ── */}
        <div className="card">
          <h3 style={{ fontSize: 13, fontWeight: 600, marginBottom: "0.5rem" }}>Mean Values</h3>
          <ResponsiveContainer width="100%" height={260}>
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

        {/* ── Analysis Results ── */}
        {analysisResult && (
          <div className="card">
            <h3 style={{ fontSize: 13, fontWeight: 600, marginBottom: "0.75rem" }}>Analysis Results</h3>
            <div className="metrics" style={{ marginBottom: "0.75rem" }}>
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
                style={{ width: "100%", marginTop: "0.75rem", borderRadius: 8 }} />
            ))}
          </div>
        )}

      </>)}
    </div>
  );
}