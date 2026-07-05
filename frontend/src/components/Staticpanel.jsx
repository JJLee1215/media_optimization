import { useState, useEffect, useRef, useMemo } from "react";
import { Bar } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale, LinearScale, BarElement,
  Title, Tooltip, Legend,
} from "chart.js";
import axios from "axios";
import "../pages/DataPage.css";

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

const API = "http://localhost:8000";

const COLORS = [
  "#1D9E75","#534AB7","#E24B4A","#EF9F27","#185FA5",
  "#9FE1CB","#AFA9EC","#F0997B","#888780",
];

const ANALYSIS_TABS = [
  { id: "overview",     label: "Overview",       icon: "ti-layout-dashboard" },
  { id: "correlation",  label: "Correlation",    icon: "ti-chart-dots"       },
  { id: "distribution", label: "Distribution",   icon: "ti-chart-histogram"  },
  { id: "titer",        label: "Titer analysis", icon: "ti-chart-scatter"    },
  { id: "pca",          label: "PCA",            icon: "ti-circles-relation" },
];

export default function StaticPanel() {
  const [file,         setFile]         = useState(null);
  const [uploading,    setUploading]    = useState(false);
  const [info,         setInfo]         = useState(null);
  const [variables,    setVariables]    = useState([]);
  const [selectedVars, setSelectedVars] = useState([]);
  const [batchId,      setBatchId]      = useState("all");
  const [batches,      setBatches]      = useState([]);
  const [activeTab,    setActiveTab]    = useState("overview");
  const [loading,      setLoading]      = useState(false);
  const [error,        setError]        = useState("");

  // 전체 stats 데이터 (배치별 포함)
  const [allStats,      setAllStats]      = useState(null);  // {all: {...}, 1: {...}, 2: {...}, ...}
  const [heatmapImg,    setHeatmapImg]    = useState(null);
  const [corrStatsImg,  setCorrStatsImg]  = useState(null);
  const [distImg,       setDistImg]       = useState(null);
  const [outlierImg,    setOutlierImg]    = useState(null);
  const [titerImg,      setTiterImg]      = useState(null);
  const [pcaImg,        setPcaImg]        = useState(null);

  const fileInputRef = useRef(null);

  // ── 파일 업로드 후 즉시 그래프 표시 ──────────────
  const handleUpload = async (e) => {
    const f = e.target.files[0];
    if (!f) return;
    setUploading(true);
    try {
      const form = new FormData();
      form.append("file", f);
      const { data } = await axios.post(`${API}/data/upload`, form);
      setFile(data.filename);
      await loadFileInfo(data.filename);  // 업로드 즉시 데이터 로드
    } catch (e) {
      setError("Upload failed.");
    } finally {
      setUploading(false);
    }
  };

  const clearFile = () => {
    setFile(null);
    setInfo(null);
    setVariables([]);
    setSelectedVars([]);
    setBatches([]);
    setAllStats(null);
    resetImages();
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const resetImages = () => {
    setHeatmapImg(null); setCorrStatsImg(null);
    setDistImg(null); setOutlierImg(null);
    setTiterImg(null); setPcaImg(null);
  };

  // ── 파일 정보 로드 + 배치 목록 + 전체 stats ──────
  const loadFileInfo = async (filename) => {
    try {
      // 전체 stats 로드
      const { data } = await axios.get(
        `${API}/data/analyze/static?type=stats&filepath=${filename}`
      );
      const cols = Object.keys(data).filter(
        k => !["Batch_ID","titer_final","viab_final"].includes(k)
      );
      setVariables(cols);
      setSelectedVars(cols);
      setInfo({
        rows   : Math.round(data[cols[0]]?.count ?? 100),
        columns: cols.length,
      });

      // 배치 목록 생성 (count 기준)
      const n = Math.round(data[cols[0]]?.count ?? 0);
      const batchList = Array.from({ length: n }, (_, i) => i + 1);
      setBatches(batchList);

      // allStats에 "all" 저장
      setAllStats({ all: data });

    } catch (e) {
      setError("Failed to load file info.");
    }
  };

  // ── 배치 변경 시 해당 배치 stats 로드 ────────────
  // batch_id를 포함해서 요청 → 백엔드에서 해당 배치만 필터링해서 반환
  useEffect(() => {
    if (!file || !allStats) return;
    if (allStats[batchId]) return;  // 이미 로드된 배치는 스킵 (캐싱)

    const loadBatchStats = async () => {
      try {
        const url = batchId === "all"
          ? `${API}/data/analyze/static?type=stats&filepath=${file}`
          : `${API}/data/analyze/static?type=stats&filepath=${file}&batch_id=${batchId}`;
        const { data } = await axios.get(url);
        setAllStats(prev => ({ ...prev, [batchId]: data }));
      } catch (e) {}
    };
    loadBatchStats();
  }, [batchId, file]);

  // ── 현재 선택된 배치의 stats ─────────────────────
  const currentStats = useMemo(() => {
    if (!allStats) return null;
    return allStats[batchId] ?? allStats["all"] ?? null;
  }, [allStats, batchId]);

  // ── Mean bar chart 데이터 (selectedVars + batchId 반응형) ──
  // 1. 체크된 성분만 표시
  // 2. 체크 해제된 성분은 투명하게 (숨김X, 투명처리)
  // 3. y축은 체크된 성분의 최대값 기준으로 자동 조정
  const barData = useMemo(() => {
    if (!currentStats || variables.length === 0) return null;

    const means = variables.map(v => currentStats[v]?.mean ?? 0);
    const bgColors = variables.map((v, i) => {
      const isSelected = selectedVars.includes(v);
      const hex = COLORS[i % COLORS.length];
      return isSelected ? hex + "CC" : hex + "22";  // CC=80% opacity, 22=13% opacity
    });

    return {
      labels  : variables,
      datasets: [{
        label          : "Mean value",
        data           : means,
        backgroundColor: bgColors,
        borderRadius   : 4,
      }],
    };
  }, [currentStats, variables, selectedVars]);

  // y축 max: 체크된 성분 중 최대값 기준으로 자동 조정
  const yMax = useMemo(() => {
    if (!currentStats || selectedVars.length === 0) return undefined;
    const vals = selectedVars.map(v => currentStats[v]?.mean ?? 0);
    return Math.max(...vals) * 1.15;
  }, [currentStats, selectedVars]);

  const barOptions = useMemo(() => ({
    responsive: true,
    aspectRatio: 3,
    plugins: {
      legend: { display: false },
      title : { display: true, text: `Mean Values${batchId === "all" ? " (all batches)" : ` (Batch ${batchId})`}` },
    },
    scales: {
      y: {
        beginAtZero: true,
        max        : yMax,          // 체크된 성분 기준 y축 자동 조정
        ticks      : { color: "var(--text-secondary)" },
      },
      x: { ticks: { color: "var(--text-secondary)" } },
    },
  }), [batchId, yMax]);

  // ── Run Analysis (PNG 탭들) ───────────────────────
  const runAnalysis = async () => {
    if (!file) return;
    setLoading(true);
    setError("");
    resetImages();
    try {
      const colParam = selectedVars.join(",");
      const base     = `${API}/data/analyze/static`;
      const fp       = `filepath=${file}`;
      const cols     = `selected_cols=${colParam}`;
      const batch    = `batch_id=${batchId}`;
      const ts       = Date.now();

      const fetchImg = (type) =>
        `${base}?type=${type}&${fp}&${cols}&${batch}&t=${ts}`;

      setHeatmapImg(fetchImg("heatmap"));
      setCorrStatsImg(fetchImg("correlation_stats"));
      setDistImg(fetchImg("distribution"));
      setOutlierImg(fetchImg("outlier"));
      setTiterImg(fetchImg("titer"));
      setPcaImg(fetchImg("pca"));

    } catch (e) {
      setError("Analysis failed.");
    } finally {
      setLoading(false);
    }
  };

  const toggleVar = (v) => {
    setSelectedVars(prev =>
      prev.includes(v) ? prev.filter(x => x !== v) : [...prev, v]
    );
  };

  // ── Stats 테이블 ──────────────────────────────────
  const renderStatsTable = () => {
    if (!currentStats) return null;
    const cols    = selectedVars.filter(v => currentStats[v]);
    const metrics = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"];
    return (
      <div style={{ overflowX: "auto", marginTop: "1rem" }}>
        <table className="result-table" style={{ fontSize: 11 }}>
          <thead>
            <tr>
              <th>Metric</th>
              {cols.map(c => <th key={c}>{c.replace("_0","")}</th>)}
            </tr>
          </thead>
          <tbody>
            {metrics.map(m => (
              <tr key={m}>
                <td style={{ fontWeight: 500, color: "var(--text-secondary)" }}>{m}</td>
                {cols.map(c => (
                  <td key={c}>
                    {currentStats[c]?.[m] !== undefined
                      ? Number(currentStats[c][m]).toFixed(3)
                      : "—"}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  };

  // ── 이미지 카드 ───────────────────────────────────
  const ImgCard = ({ title, src }) => (
    <div className="card" style={{ marginTop: "0.75rem" }}>
      <div style={{ fontSize: 11, fontWeight: 600, color: "var(--text-muted)",
                    textTransform: "uppercase", marginBottom: "0.75rem" }}>
        {title}
      </div>
      {src
        ? <img src={src} alt={title} style={{ width: "100%", borderRadius: 8 }} />
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
              <input ref={fileInputRef} type="file" accept=".csv"
                id="static-upload" style={{ display: "none" }}
                onChange={handleUpload} disabled={uploading} />
              <label htmlFor="static-upload" className="file-upload-btn">
                {uploading ? "Uploading..." : "Choose file"}
              </label>
            </>
          ) : (
            <div className="file-uploaded-row">
              <span style={{ fontSize: 11, color: "#1D9E75", fontWeight: 600 }}>
                📄 {file}
              </span>
              <button className="file-clear-btn" onClick={clearFile}>✕</button>
            </div>
          )}
        </div>
        <div className="stat-bar-item">
          <span className="stat-bar-label">Batches</span>
          <span className="stat-bar-value">{info?.rows ?? "—"}</span>
        </div>
        <div className="stat-bar-item">
          <span className="stat-bar-label">Columns</span>
          <span className="stat-bar-value">{info?.columns ?? "—"}</span>
        </div>
        <div className="stat-bar-item"
          style={{ display: "flex", alignItems: "center", justifyContent: "center" }}>
          <button className="train-run-btn"
            onClick={runAnalysis}
            disabled={!file || loading}>
            {loading ? "⏳ Running..." : "▶ Run Analysis"}
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

      {/* ── Batch + Variables 필터 ── */}
      {variables.length > 0 && (
        <div className="card" style={{ marginTop: "0.75rem" }}>
          <div style={{ display: "flex", alignItems: "flex-start",
                        gap: "1.5rem", flexWrap: "wrap" }}>

            {/* 배치 선택 */}
            <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
              <span style={{ fontSize: 10, fontWeight: 600,
                             color: "var(--text-muted)", textTransform: "uppercase" }}>
                Batch
              </span>
              <select
                value={batchId}
                onChange={e => setBatchId(e.target.value)}
                style={{ fontSize: 12, padding: "3px 8px", borderRadius: 6,
                         border: "0.5px solid var(--border-strong)",
                         background: "var(--surface-1)", color: "var(--text-primary)",
                         cursor: "pointer" }}>
                <option value="all">All</option>
                {batches.map(b => (
                  <option key={b} value={b}>Batch {b}</option>
                ))}
              </select>
            </div>

            {/* Variables 체크박스 */}
            <div style={{ flex: 1 }}>
              <div style={{ display: "flex", alignItems: "center",
                            justifyContent: "space-between", marginBottom: 6 }}>
                <span style={{ fontSize: 10, fontWeight: 600,
                               color: "var(--text-muted)", textTransform: "uppercase" }}>
                  Variables
                </span>
                <div style={{ display: "flex", gap: 4 }}>
                  <button className="log-clear"
                    onClick={() => setSelectedVars(variables)}>All</button>
                  <button className="log-clear"
                    onClick={() => setSelectedVars([])}>None</button>
                </div>
              </div>
              <div style={{ display: "flex", flexWrap: "wrap", gap: "0.5rem" }}>
                {variables.map((v, i) => (
                  <label key={v}
                    style={{ display: "flex", alignItems: "center", gap: 4,
                             fontSize: 12, cursor: "pointer",
                             opacity: selectedVars.includes(v) ? 1 : 0.4,
                             transition: "opacity 0.15s" }}>
                    <input type="checkbox"
                      checked={selectedVars.includes(v)}
                      onChange={() => toggleVar(v)} />
                    <span style={{ color: COLORS[i % COLORS.length],
                                   fontWeight: 500 }}>●</span>
                    {v}
                  </label>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ════════ 탭별 콘텐츠 ════════ */}

      {/* Overview — 파일 업로드 즉시 표시 */}
      {activeTab === "overview" && (
        <div>
          {barData ? (
            <div className="card" style={{ marginTop: "0.75rem" }}>
              <Bar data={barData} options={barOptions} />
            </div>
          ) : (
            <div className="card" style={{ marginTop: "0.75rem",
                                           height: 120, display: "flex",
                                           alignItems: "center",
                                           justifyContent: "center",
                                           color: "var(--text-muted)", fontSize: 12 }}>
              Upload a CSV file to view the chart
            </div>
          )}
          {currentStats && (
            <div className="card" style={{ marginTop: "0.75rem" }}>
              <div style={{ fontSize: 11, fontWeight: 600, color: "var(--text-muted)",
                            textTransform: "uppercase", marginBottom: "0.5rem" }}>
                Basic statistics
              </div>
              {renderStatsTable()}
            </div>
          )}
        </div>
      )}

      {/* Correlation */}
      {activeTab === "correlation" && (
        <div>
          <ImgCard title="Component correlation heatmap" src={heatmapImg} />
          <ImgCard title="Pearson / Spearman / p-value vs Titer" src={corrStatsImg} />
        </div>
      )}

      {/* Distribution */}
      {activeTab === "distribution" && (
        <div>
          <ImgCard title="Feature distributions" src={distImg} />
          <ImgCard title="Outlier detection (IQR boxplot)" src={outlierImg} />
        </div>
      )}

      {/* Titer analysis */}
      {activeTab === "titer" && (
        <ImgCard title="Component vs Titer" src={titerImg} />
      )}

      {/* PCA */}
      {activeTab === "pca" && (
        <ImgCard title="PCA biplot + explained variance" src={pcaImg} />
      )}

    </div>
  );
}