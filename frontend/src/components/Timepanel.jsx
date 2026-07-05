import { useState, useEffect } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer,
} from "recharts";
import axios from "axios";

const API = "http://localhost:8000";

const COLORS = [
  "#1D9E75","#534AB7","#E24B4A","#EF9F27","#185FA5",
  "#9FE1CB","#AFA9EC","#F0997B","#888780",
  "#3B6D11","#A32D2D","#633806","#0C447C",
];

export default function TimePanel() {
  const [file,          setFile]          = useState(null);
  const [uploading,     setUploading]     = useState(false);
  const [colInfo,       setColInfo]       = useState(null);
  const [columns,       setColumns]       = useState([]);
  const [batchIds,      setBatchIds]      = useState([]);
  const [selectedCols,  setSelectedCols]  = useState([]);
  const [selectedBatch, setSelectedBatch] = useState("all");
  const [rawData,       setRawData]       = useState({});
  const [loading,       setLoading]       = useState(false);
  const [error,         setError]         = useState("");

  // ── 파일 업로드 ──────────────────────────────
  const handleUpload = async (e) => {
    const f = e.target.files[0];
    if (!f) return;
    setUploading(true);
    setError("");
    try {
      const form = new FormData();
      form.append("file", f);
      const { data } = await axios.post(`${API}/data/upload`, form);
      setFile(data.filename);
      await loadColumns(data.filename);
    } catch (err) {
      setError("Upload failed.");
    } finally {
      setUploading(false);
    }
  };

  const clearFile = () => {
    setFile(null);
    setColInfo(null);
    setColumns([]);
    setBatchIds([]);
    setSelectedCols([]);
    setRawData({});
    setError("");
  };

  // ── 컬럼 + 배치 목록 로드 ─────────────────────
  const loadColumns = async (filename) => {
    try {
      const { data } = await axios.get(
        `${API}/data/columns?filename=${filename}&type=timeseries`
      );
      setColInfo(data);
      setColumns(data.columns ?? []);
      setBatchIds(data.batches ?? []);
      setSelectedCols((data.columns ?? []).slice(0, 5));
    } catch (err) {
      setError("Failed to load columns.");
    }
  };

  // ── 데이터 로드 ───────────────────────────────
  const loadData = async () => {
    if (!file) return;
    setLoading(true);
    setError("");
    try {
      const { data } = await axios.post(`${API}/data/analyze`, {
        filename : file,
        type     : "timeseries",
        columns  : selectedCols,
        batch_id : selectedBatch,
      });
      setRawData(data);
    } catch (err) {
      setError("Failed to load data.");
    } finally {
      setLoading(false);
    }
  };

  // 파일 변경 또는 배치/컬럼 변경 시 자동 로드
  useEffect(() => {
    if (file && columns.length > 0) loadData();
  }, [file, selectedBatch, selectedCols]);

  // ── 차트 데이터 생성 ──────────────────────────
  const buildChartData = () => {
    if (!batchIds || batchIds.length === 0) return { chartData: [], lines: [] };

    const batches = selectedBatch === "all"
      ? batchIds.slice(0, 5)
      : [selectedBatch];

    if (!batches.length || !rawData[String(batches[0])]) return { chartData: [], lines: [] };

    if (selectedBatch === "all") {
      const firstBatch = String(batchIds[0]);
      const times      = rawData[firstBatch]?.time || [];
      const chartData  = times.map((t, i) => {
        const row = { time: t };
        batches.forEach(bid => {
          const bd = rawData[String(bid)];
          if (!bd) return;
          selectedCols.forEach(col => { row[`${col}_b${bid}`] = bd[col]?.[i] ?? null; });
        });
        return row;
      });
      const lines = [];
      batches.forEach((bid, bi) => {
        selectedCols.forEach((col, ci) => {
          lines.push({
            key  : `${col}_b${bid}`,
            color: COLORS[(bi * selectedCols.length + ci) % COLORS.length],
            name : `${col} (B${bid})`,
          });
        });
      });
      return { chartData, lines };
    } else {
      const bd       = rawData[String(selectedBatch)];
      const times    = bd?.time || [];
      const chartData = times.map((t, i) => {
        const row = { time: t };
        selectedCols.forEach(col => { row[col] = bd[col]?.[i] ?? null; });
        return row;
      });
      const lines = selectedCols.map((col, i) => ({
        key  : col,
        color: COLORS[i % COLORS.length],
        name : col,
      }));
      return { chartData, lines };
    }
  };

  const toggleCol = (col) => {
    setSelectedCols(prev =>
      prev.includes(col) ? prev.filter(c => c !== col) : [...prev, col]
    );
  };

  const { chartData, lines } = buildChartData();

  return (
    <div>
      {/* ── 상단 바 ── */}
      <div className="stat-bar">
        <div className="stat-bar-item" style={{ flex: 2 }}>
          <span className="stat-bar-label">Upload</span>
          {!file ? (
            <>
              <input type="file" accept=".csv" id="ts-upload"
                style={{ display: "none" }} onChange={handleUpload}
                disabled={uploading} />
              <label htmlFor="ts-upload" className="file-upload-btn">
                {uploading ? "Uploading..." : "Choose file"}
              </label>
            </>
          ) : (
            <div className="file-uploaded-row">
              <span style={{ fontSize: 11, color: "#185FA5", fontWeight: 600 }}>
                📄 {file}
              </span>
              <button className="file-clear-btn" onClick={clearFile}>✕</button>
            </div>
          )}
        </div>
        <div className="stat-bar-item">
          <span className="stat-bar-label">Batches</span>
          <span className="stat-bar-value">{batchIds?.length || "—"}</span>
        </div>
        <div className="stat-bar-item">
          <span className="stat-bar-label">Columns</span>
          <span className="stat-bar-value">{columns?.length || "—"}</span>
        </div>
        <div className="stat-bar-item"
          style={{ display: "flex", alignItems: "center", justifyContent: "center" }}>
          <button className="train-run-btn"
            onClick={loadData} disabled={!file || loading}>
            {loading ? "⏳ Loading..." : "▶ Run Analysis"}
          </button>
        </div>
      </div>

      {error && (
        <div style={{ marginTop: "0.5rem", padding: "0.5rem 1rem",
                      background: "#FCEBEB", border: "0.5px solid #F09595",
                      borderRadius: 8, fontSize: 12, color: "#A32D2D" }}>
          ❌ {error}
        </div>
      )}

      {/* ── 필터 ── */}
      {columns.length > 0 && (
        <div className="card" style={{ marginTop: "0.75rem" }}>
          <div style={{ display: "flex", alignItems: "flex-start",
                        gap: "1.5rem", flexWrap: "wrap" }}>

            {/* 배치 선택 */}
            <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
              <span style={{ fontSize: 10, fontWeight: 600,
                             color: "var(--text-muted)", textTransform: "uppercase" }}>
                Batch
              </span>
              <select value={selectedBatch}
                onChange={e => setSelectedBatch(e.target.value)}
                style={{ fontSize: 12, padding: "3px 8px", borderRadius: 6,
                         border: "0.5px solid var(--border-strong)",
                         background: "var(--surface-1)",
                         color: "var(--text-primary)", cursor: "pointer" }}>
                <option value="all">All (first 5)</option>
                {batchIds.map(b => (
                  <option key={b} value={b}>Batch {b}</option>
                ))}
              </select>
            </div>

            {/* 컬럼 선택 */}
            <div style={{ flex: 1 }}>
              <div style={{ display: "flex", alignItems: "center",
                            justifyContent: "space-between", marginBottom: 6 }}>
                <span style={{ fontSize: 10, fontWeight: 600,
                               color: "var(--text-muted)", textTransform: "uppercase" }}>
                  Variables
                </span>
                <div style={{ display: "flex", gap: 4 }}>
                  <button className="log-clear"
                    onClick={() => setSelectedCols(columns)}>All</button>
                  <button className="log-clear"
                    onClick={() => setSelectedCols([])}>None</button>
                </div>
              </div>
              <div style={{ display: "flex", flexWrap: "wrap", gap: "0.5rem" }}>
                {columns.map((col, i) => (
                  <label key={col}
                    style={{ display: "flex", alignItems: "center", gap: 4,
                             fontSize: 12, cursor: "pointer",
                             opacity: selectedCols.includes(col) ? 1 : 0.4,
                             transition: "opacity 0.15s" }}>
                    <input type="checkbox"
                      checked={selectedCols.includes(col)}
                      onChange={() => toggleCol(col)} />
                    <span style={{ color: COLORS[i % COLORS.length],
                                   fontWeight: 500 }}>●</span>
                    {col}
                  </label>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── 차트 ── */}
      {chartData.length > 0 && (
        <div className="card" style={{ marginTop: "0.75rem" }}>
          <div style={{ fontSize: 11, fontWeight: 600, color: "var(--text-muted)",
                        textTransform: "uppercase", marginBottom: "0.75rem" }}>
            Time series
            {selectedBatch !== "all" ? ` — Batch ${selectedBatch}` : " — All batches (first 5)"}
          </div>
          <ResponsiveContainer width="100%" height={400}>
            <LineChart data={chartData}
              margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
              <XAxis dataKey="time" label={{ value: "Day", position: "insideBottom", offset: -2 }}
                tick={{ fontSize: 11 }} />
              <YAxis label={{ value: "Concentration (g/L)", angle: -90,
                              position: "insideLeft", offset: 10, fontSize: 11 }}
                tick={{ fontSize: 11 }} />
              <Tooltip />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              {lines.map(l => (
                <Line key={l.key} type="monotone" dataKey={l.key}
                  stroke={l.color} name={l.name}
                  dot={false} strokeWidth={1.5} />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* 데이터 없을 때 */}
      {file && !loading && chartData.length === 0 && (
        <div className="card" style={{ marginTop: "0.75rem", height: 120,
                                       display: "flex", alignItems: "center",
                                       justifyContent: "center",
                                       color: "var(--text-muted)", fontSize: 12 }}>
          No data to display. Select variables and run analysis.
        </div>
      )}
    </div>
  );
}