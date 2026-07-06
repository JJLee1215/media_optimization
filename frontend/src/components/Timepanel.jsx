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

const ANALYSIS_TABS = [
  { id: "overview",     label: "Overview",          icon: "ti-chart-line"     },
  { id: "batch",        label: "Batch overlay",     icon: "ti-layers-subtract" },
  { id: "titer",        label: "Titer trajectory",  icon: "ti-chart-scatter"  },
  { id: "correlation",  label: "Correlation",       icon: "ti-chart-dots"     },
  { id: "fault",        label: "Fault detection",   icon: "ti-alert-triangle" },
];

export default function TimePanel() {
  const [file,          setFile]          = useState(null);
  const [uploading,     setUploading]     = useState(false);
  const [columns,       setColumns]       = useState([]);
  const [batchIds,      setBatchIds]      = useState([]);
  const [selectedCols,  setSelectedCols]  = useState([]);
  const [selectedBatch, setSelectedBatch] = useState("all");
  const [rawData,       setRawData]       = useState({});
  const [loading,       setLoading]       = useState(false);
  const [error,         setError]         = useState("");
  const [activeTab,     setActiveTab]     = useState("overview");

  // PNG 이미지 상태
  const [batchImg,       setBatchImg]       = useState(null);
  const [titerImg,       setTiterImg]       = useState(null);
  const [corrImg,        setCorrImg]        = useState(null);
  const [faultImg,       setFaultImg]       = useState(null);

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
    setColumns([]);
    setBatchIds([]);
    setSelectedCols([]);
    setRawData({});
    setError("");
    setBatchImg(null); setTiterImg(null);
    setCorrImg(null);  setFaultImg(null);
  };

  // ── 컬럼 + 배치 목록 로드 ─────────────────────
  const loadColumns = async (filename) => {
    try {
      const { data } = await axios.get(
        `${API}/data/columns?filename=${filename}&type=timeseries`
      );
      setColumns(data.columns ?? []);
      setBatchIds(data.batches ?? []);
      setSelectedCols((data.columns ?? []).slice(0, 5));
    } catch (err) {
      setError("Failed to load columns.");
    }
  };

  // ── 시계열 JSON 데이터 로드 (Overview용) ──────
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

  // ── PNG 분석 이미지 로드 ──────────────────────
  const loadImages = async () => {
    if (!file) return;
    const ts   = Date.now();
    const base = `${API}/data/analyze/timeseries`;
    const fp   = `filepath=${file}&t=${ts}`;
    setBatchImg(`${base}?type=batch_overlay&${fp}`);
    setTiterImg(`${base}?type=titer_trajectory&${fp}`);
    setCorrImg(`${base}?type=ts_correlation&${fp}`);
    setFaultImg(`${base}?type=fault_detection&${fp}`);
  };

  useEffect(() => {
    if (file && columns.length > 0) loadData();
  }, [file, selectedBatch, selectedCols]);

  // ── 차트 데이터 생성 (Overview) ───────────────
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
      const bd        = rawData[String(selectedBatch)];
      const times     = bd?.time || [];
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

  // ── 이미지 카드 ───────────────────────────────
  const ImgCard = ({ title, src }) => (
    <div className="card" style={{ marginTop: "0.75rem" }}>
      <div style={{ fontSize: 11, fontWeight: 600, color: "var(--text-muted)",
                    textTransform: "uppercase", marginBottom: "0.75rem" }}>
        {title}
      </div>
      {src
        ? <img src={src} alt={title} style={{ width: "100%", borderRadius: 8 }}
            onError={() => {}} />
        : <div style={{ height: 120, display: "flex", alignItems: "center",
                        justifyContent: "center", color: "var(--text-muted)", fontSize: 12 }}>
            Run analysis to generate
          </div>
      }
    </div>
  );

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
            onClick={() => { loadData(); loadImages(); }}
            disabled={!file || loading}>
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

      {/* ── 분석 탭 ── */}
      <div className="analysis-tab-bar">
        {ANALYSIS_TABS.map(t => (
          <button key={t.id}
            className={"analysis-tab" + (activeTab === t.id ? " active" : "")}
            onClick={() => setActiveTab(t.id)}>
            <i className={`ti ${t.icon}`} aria-hidden="true" />
            {t.label}
          </button>
        ))}
      </div>

      {/* ── Batch + Variables 필터 (Overview만) ── */}
      {activeTab === "overview" && columns.length > 0 && (
        <div className="card" style={{ marginTop: "0.75rem" }}>
          <div style={{ display: "flex", alignItems: "flex-start",
                        gap: "1.5rem", flexWrap: "wrap" }}>
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

      {/* ════════ 탭별 콘텐츠 ════════ */}

      {/* Overview */}
      {activeTab === "overview" && (
        <div>
          {chartData.length > 0 ? (
            <div className="card" style={{ marginTop: "0.75rem" }}>
              <div style={{ fontSize: 11, fontWeight: 600, color: "var(--text-muted)",
                            textTransform: "uppercase", marginBottom: "0.75rem" }}>
                Time Series —
                {selectedBatch !== "all" ? ` Batch ${selectedBatch}` : " All batches (first 5)"}
              </div>
              <ResponsiveContainer width="100%" height={400}>
                <LineChart data={chartData}
                  margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                  <XAxis dataKey="time"
                    label={{ value: "Day", position: "insideBottom", offset: -2 }}
                    tick={{ fontSize: 11 }} />
                  <YAxis
                    label={{ value: "Concentration (g/L)", angle: -90,
                             position: "insideLeft", offset: 10, fontSize: 11 }}
                    tick={{ fontSize: 11 }} />
                  <Tooltip />
                  {lines.map(l => (
                    <Line key={l.key} type="monotone" dataKey={l.key}
                      stroke={l.color} name={l.name}
                      dot={false} strokeWidth={1.5} />
                  ))}
                </LineChart>
              </ResponsiveContainer>
              {/* 외부 Legend */}
              <div style={{ display: "flex", flexWrap: "wrap", gap: "0.4rem", marginTop: "0.75rem" }}>
                {lines.map(l => (
                  <span key={l.key} style={{ display: "flex", alignItems: "center", gap: 4,
                                              fontSize: 11, color: "var(--text-secondary)" }}>
                    <span style={{ width: 12, height: 2, background: l.color,
                                  display: "inline-block", borderRadius: 1 }} />
                    {l.name}
                  </span>
                ))}
              </div>
            </div>
          ) : (
            file && (
              <div className="card" style={{ marginTop: "0.75rem", height: 120,
                                             display: "flex", alignItems: "center",
                                             justifyContent: "center",
                                             color: "var(--text-muted)", fontSize: 12 }}>
                Select variables and run analysis.
              </div>
            )
          )}
        </div>
      )}

      {/* Batch overlay */}
      {activeTab === "batch" && (
        <ImgCard title="Batch overlay — 배치별 시계열 비교" src={batchImg} />
      )}

      {/* Titer trajectory */}
      {activeTab === "titer" && (
        <ImgCard title="Titer trajectory — 시간별 titer 변화" src={titerImg} />
      )}

      {/* Correlation */}
      {activeTab === "correlation" && (
        <ImgCard title="Day별 컴포넌트 vs 최종 Titer 상관관계" src={corrImg} />
      )}

      {/* Fault detection */}
      {activeTab === "fault" && (
        <ImgCard title="Fault detection — 이상 배치 탐지" src={faultImg} />
      )}

    </div>
  );
}