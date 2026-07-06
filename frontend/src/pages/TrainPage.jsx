import { useState, useEffect, useRef } from "react";
import axios from "axios";
import "./TrainPage.css";

const API = "http://localhost:8000";

const MODEL_ICONS = {
  gaussian_process : "📈",
  xgboost          : "⚡",
  random_forest    : "🌲",
  mlp              : "🧠",
  rnn              : "🔁",
  lstm             : "⏱️",
  transformer      : "⚙️",
  tcn              : "🌊",
  static_time_gnn  : "🕸️",
};

const HG = ["⏳", "⌛"];

export default function TrainPage() {
  const [models,          setModels]          = useState({ static: [], timeseries: [], static_time: [] });
  const [selected,        setSelected]        = useState([]);
  const [status,          setStatus]          = useState("idle");
  const [allResults,      setAllResults]      = useState({});
  const [allImages,       setAllImages]       = useState({});
  const [selectedTab,     setSelectedTab]     = useState(null);
  const [staticFile,      setStaticFile]      = useState("");
  const [tsFile,          setTsFile]          = useState("");
  const [staticUploading, setStaticUploading] = useState(false);
  const [tsUploading,     setTsUploading]     = useState(false);
  const [usePipeline,     setUsePipeline]     = useState(false);

  // Feature selection
  const [availableFeats,  setAvailableFeats]  = useState([]);
  const [selectedFeats,   setSelectedFeats]   = useState([]);

  // 로그 스트리밍
  const [logs,      setLogs]      = useState([]);
  const [hgIcon,    setHgIcon]    = useState("⏳");
  const [hgText,    setHgText]    = useState("");
  const [elapsed,   setElapsed]   = useState(0);
  const [streaming, setStreaming] = useState(false);

  const staticInputRef = useRef(null);
  const tsInputRef     = useRef(null);
  const sseRef         = useRef(null);
  const logBoxRef      = useRef(null);
  const hgRef          = useRef(null);
  const elapsedRef     = useRef(null);
  const hgIdx          = useRef(0);

  useEffect(() => {
    fetchModels();
    fetchAllResults().then(() => {
      axios.get(`${API}/train/results/all`).then(({ data }) => {
        const ids = Object.keys(data);
        if (ids.length > 0) fetchAllImages(ids);
      }).catch(() => {});
    });
    return () => {
      clearInterval(hgRef.current);
      clearInterval(elapsedRef.current);
      if (sseRef.current) sseRef.current.close();
    };
  }, []);

  useEffect(() => {
    if (logBoxRef.current) logBoxRef.current.scrollTop = logBoxRef.current.scrollHeight;
  }, [logs]);

  const fetchModels = async () => {
    try { const { data } = await axios.get(`${API}/train/models`); setModels(data); } catch (e) {}
  };
  const fetchAllResults = async () => {
    try { const { data } = await axios.get(`${API}/train/results/all`); setAllResults(data); } catch (e) {}
  };
  const fetchAllImages = async (modelIds) => {
    const images = {};
    for (const mid of modelIds) {
      try {
        const { data } = await axios.get(`${API}/train/results/${mid}`);
        if (data.images && Object.keys(data.images).length > 0) images[mid] = data.images;
      } catch (e) {}
    }
    setAllImages(images);
    const first = modelIds.find(m => images[m]);
    if (first) setSelectedTab(first);
  };

  // 파일 업로드 후 컬럼 로드
  const handleStaticUpload = async (e) => {
    const f = e.target.files[0]; if (!f) return;
    setStaticUploading(true);
    try {
      const form = new FormData(); form.append("file", f);
      const { data } = await axios.post(`${API}/data/upload`, form);
      setStaticFile(data.filename);
      // 컬럼 로드
      const { data: colData } = await axios.get(
        `${API}/data/analyze/static?type=stats&filepath=${data.filename}`
      );
      const cols = Object.keys(colData).filter(
        k => !["Batch_ID","titer_final","viab_final"].includes(k)
      );
      setAvailableFeats(cols);
      setSelectedFeats(cols);
    } catch (e) {} finally { setStaticUploading(false); }
  };

  const handleTsUpload = async (e) => {
    const f = e.target.files[0]; if (!f) return;
    setTsUploading(true);
    try {
      const form = new FormData(); form.append("file", f);
      const { data } = await axios.post(`${API}/data/upload`, form);
      setTsFile(data.filename);
    } catch (e) {} finally { setTsUploading(false); }
  };

  const clearStaticFile = () => {
    setStaticFile(""); setAvailableFeats([]); setSelectedFeats([]);
    if (staticInputRef.current) staticInputRef.current.value = "";
  };
  const clearTsFile = () => {
    setTsFile("");
    if (tsInputRef.current) tsInputRef.current.value = "";
  };

  const toggleModel = (id) => {
    setSelected(prev => prev.includes(id) ? prev.filter(m => m !== id) : [...prev, id]);
  };

  const toggleFeat = (f) => {
    setSelectedFeats(prev => prev.includes(f) ? prev.filter(x => x !== f) : [...prev, f]);
  };

  const classifyLog = (text) => {
    if (text.includes("✅") || text.includes("complete") || text.includes("Saved")) return "green";
    if (text.includes("❌") || text.includes("Error"))   return "red";
    if (text.startsWith("▶"))                            return "yellow";
    if (text.startsWith("═"))                            return "divider";
    if (text.startsWith("[preprocess]"))                 return "muted";
    return "white";
  };

  const startHourglass = () => {
    hgIdx.current = 0; setHgIcon(HG[0]);
    hgRef.current = setInterval(() => {
      hgIdx.current = (hgIdx.current + 1) % 2;
      setHgIcon(HG[hgIdx.current]);
    }, 600);
  };

  const startElapsed = () => {
    let sec = 0; setElapsed(0);
    elapsedRef.current = setInterval(() => { sec++; setElapsed(sec); }, 1000);
  };

  const stopIndicators = () => {
    clearInterval(hgRef.current);
    clearInterval(elapsedRef.current);
  };

  const handleTrain = async () => {
    if (status === "running" || selected.length === 0) return;
    setLogs([]); setStatus("running"); setStreaming(true);
    setHgText("Training..."); startHourglass(); startElapsed();

    if (sseRef.current) sseRef.current.close();
    const sse = new EventSource(`${API}/train/stream`);
    sseRef.current = sse;

    sse.onmessage = (e) => {
      const data = JSON.parse(e.data);
      if (data.ping) return;
      if (data.log) {
        const text = data.log;
        const match = text.match(/▶ Training: ([A-Z_]+)/);
        if (match) setHgText(`Training ${match[1].toLowerCase()}...`);
        setLogs(prev => [...prev, { text, cls: classifyLog(text) }]);
        if (text.includes("✅")) {
          stopIndicators(); setStreaming(false); setStatus("done"); sse.close();
          fetchModels(); fetchAllResults().then(() => {
            axios.get(`${API}/train/results/all`).then(({ data }) => fetchAllImages(Object.keys(data)));
          });
        }
        if (text.includes("❌")) { stopIndicators(); setStreaming(false); setStatus("error"); sse.close(); }
      }
    };
    sse.onerror = () => { stopIndicators(); setStreaming(false); setStatus("error"); sse.close(); };

    for (const modelId of selected) {
      await axios.post(`${API}/train`, {
        model        : modelId,
        use_pipeline : usePipeline,
        static_file  : staticFile || null,
        ts_file      : tsFile     || null,
        selected_cols: selectedFeats.length > 0 ? selectedFeats : null,
      });
    }
  };

  const hasStGnn  = selected.includes("static_time_gnn");
  const canTrain  = selected.length > 0 && !(hasStGnn && (!staticFile || !tsFile));
  const elapsedStr = elapsed < 60 ? `${elapsed}s` : `${Math.floor(elapsed/60)}m ${elapsed%60}s`;

  const allModels  = [...models.static, ...models.timeseries, ...models.static_time];
  const resultRows = allModels.filter(m => allResults[m.id]).map(m => ({ ...m, result: allResults[m.id] }));
  const bestR2Model = resultRows.filter(m => m.result.r2 !== undefined)
    .reduce((best, m) => (m.result.r2 ?? -99) > (best?.result?.r2 ?? -99) ? m : best, null);

  const ModelCard = ({ m, type }) => (
    <div className={"model-card" + (selected.includes(m.id) ? ` selected ${type}` : "")}
      onClick={() => status !== "running" && toggleModel(m.id)}>
      {selected.includes(m.id) && <span className={`model-check ${type}`}>✓</span>}
      <span className="model-icon">{MODEL_ICONS[m.id]}</span>
      <span className="model-name">{m.name}</span>
      <span className="model-desc">{m.desc}</span>
      {m.has_model
        ? <span className="model-badge saved">{m.model_file}</span>
        : <span className="model-badge none">no saved model</span>}
      {allResults[m.id] && (
        <span className="model-r2">
          {allResults[m.id].r2 !== undefined
            ? `R² ${allResults[m.id].r2.toFixed(3)}`
            : `RMSE ${allResults[m.id].titer_rmse ?? allResults[m.id].rmse ?? "—"}`}
        </span>
      )}
    </div>
  );

  const FileUploadItem = ({ label, file, uploading, inputRef, onUpload, onClear, color, id }) => (
    <div className="train-bar-item" style={{ flex: 1.5 }}>
      <span className="train-bar-label">{label}</span>
      {!file ? (
        <>
          <input ref={inputRef} type="file" accept=".csv" id={id}
            style={{ display: "none" }} onChange={onUpload} disabled={uploading} />
          <label htmlFor={id} className="file-upload-btn">
            {uploading ? "Uploading..." : "Choose file"}
          </label>
        </>
      ) : (
        <div className="file-uploaded-row">
          <span style={{ fontSize: 11, color, fontWeight: 600 }}>📄 {file}</span>
          <button className="file-clear-btn" onClick={onClear}>✕</button>
        </div>
      )}
    </div>
  );

  // Input dimension 계산
  const inputDim = usePipeline ? 230 : selectedFeats.length;

  return (
    <div className="train-page">

      {/* 상단 바 */}
      <div className="train-bar-top">
        <FileUploadItem label="Static file" file={staticFile} uploading={staticUploading}
          inputRef={staticInputRef} onUpload={handleStaticUpload} onClear={clearStaticFile}
          color="#1D9E75" id="static-upload" />
        <FileUploadItem label="Timeseries file" file={tsFile} uploading={tsUploading}
          inputRef={tsInputRef} onUpload={handleTsUpload} onClear={clearTsFile}
          color="#185FA5" id="ts-upload" />
        <div className="train-bar-item">
          <span className="train-bar-label">Selected</span>
          <span className="train-bar-value">{selected.length} models</span>
        </div>
        <div className="train-bar-item">
          <span className="train-bar-label">Trained</span>
          <span className="train-bar-value">{resultRows.length} models</span>
        </div>
      </div>

      {/* Training log */}
      <div className="log-wrap">
        <div className="log-header">
          <span className="log-title">Training log</span>
          <div className="log-meta">
            {streaming && (
              <>
                <span className="hourglass-icon">{hgIcon}</span>
                <span className="live-dot" />
                <span className="hg-text">{hgText}</span>
                <span className="log-elapsed">{elapsedStr}</span>
              </>
            )}
            {!streaming && elapsed > 0 && <span className="log-elapsed">완료 {elapsedStr}</span>}
            <button className="log-clear" onClick={() => setLogs([])}>Clear</button>
          </div>
        </div>
        <div className="log-box" ref={logBoxRef}>
          {logs.length === 0
            ? <span className="log-line muted">학습을 시작하면 로그가 여기에 실시간으로 표시됩니다.</span>
            : logs.map((l, i) => <span key={i} className={`log-line ${l.cls}`}>{l.text}<br /></span>)
          }
        </div>
      </div>

      {/* 메인 그리드: 왼쪽 설정 + 오른쪽 요약 */}
      <div className="train-main-grid">

        {/* 왼쪽 */}
        <div className="train-left">

          {/* 1. Feature Selection */}
          <div className="train-section">
            <div className="train-section-head feat">
              <span className="train-section-title">Feature Selection</span>
              {availableFeats.length > 0 && (
                <span className="feat-count-badge">
                  {selectedFeats.length} / {availableFeats.length}
                </span>
              )}
              <div style={{ marginLeft: "auto", display: "flex", gap: 4 }}>
                <button className="log-clear" onClick={() => setSelectedFeats(availableFeats)}>All</button>
                <button className="log-clear" onClick={() => setSelectedFeats([])}>None</button>
              </div>
            </div>
            {availableFeats.length === 0 ? (
              <div style={{ fontSize: 11, color: "var(--text-muted)", padding: "0.25rem 0" }}>
                Upload a static CSV file to select features
              </div>
            ) : (
              <div className="feat-tag-area">
                {availableFeats.map(f => (
                  <div key={f}
                    className={"feat-tag" + (selectedFeats.includes(f) ? " on" : "")}
                    onClick={() => toggleFeat(f)}>
                    {selectedFeats.includes(f) && "✓ "}{f}
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* 2. Heterogeneity */}
          <div className="train-section">
            <div className="train-section-head het">
              <span className="train-section-title">Heterogeneity</span>
            </div>
            <div className="model-grid het-grid">
              <div className={"model-card gray" + (!usePipeline ? " selected gray" : "")}
                onClick={() => status !== "running" && setUsePipeline(false)}>
                {!usePipeline && <span className="model-check gray">✓</span>}
                <span className="model-icon">🚫</span>
                <span className="model-name">No Pipeline</span>
                <span className="model-desc">Raw concentration features</span>
              </div>
              <div className={"model-card gray" + (usePipeline ? " selected gray" : "")}
                onClick={() => status !== "running" && setUsePipeline(true)}>
                {usePipeline && <span className="model-check gray">✓</span>}
                <span className="model-icon">🧬</span>
                <span className="model-name">SMILES · RDKit · GEM</span>
                <span className="model-desc">Molecular embedding pipeline</span>
              </div>
            </div>
            {usePipeline && (
              <div className="st-note" style={{ marginTop: "0.5rem" }}>
                ⚠ 데이터가 적을 경우 성능 저하 가능 (차원 230)
              </div>
            )}
          </div>

          {/* 3. Static models */}
          <div className="train-section">
            <div className="train-section-head static">
              <span className="train-section-title">Static models</span>
              <span className="section-badge static" style={{ marginLeft: 4 }}>batch_table_syn.csv</span>
            </div>
            <div className="model-grid">
              {models.static.map(m => <ModelCard key={m.id} m={m} type="static" />)}
            </div>
          </div>

          {/* 4. Time Series models */}
          <div className="train-section">
            <div className="train-section-head ts">
              <span className="train-section-title">Time Series models</span>
              <span className="section-badge ts" style={{ marginLeft: 4 }}>timeseries_syn.csv</span>
            </div>
            <div className="model-grid">
              {models.timeseries.map(m => <ModelCard key={m.id} m={m} type="ts" />)}
            </div>
          </div>

          {/* 5. Static & Time Series */}
          <div className="train-section">
            <div className="train-section-head st">
              <span className="train-section-title">Static &amp; Time Series models</span>
            </div>
            <div className="st-note">두 파일이 모두 선택되어야 학습 가능합니다</div>
            <div className="model-grid" style={{ marginTop: "0.5rem" }}>
              {models.static_time.map(m => <ModelCard key={m.id} m={m} type="st" />)}
            </div>
          </div>

        </div>

        {/* 오른쪽 요약 */}
        <div className="train-right">
          <div className="summary-wrap">

            {/* Features block */}
            <div className="sum-block feat-block">
              <div className="sum-block-header">
                <span className="sum-block-label">Features</span>
                {availableFeats.length > 0 && (
                  <span className="sum-feat-count">
                    {selectedFeats.length} / {availableFeats.length}
                  </span>
                )}
              </div>
              <div className="sum-feat-tags">
                {selectedFeats.length > 0
                  ? selectedFeats.map(f => (
                      <span key={f} className="sum-feat-tag">{f.replace("_0","")}</span>
                    ))
                  : <span style={{ fontSize: 10, color: "var(--text-muted)" }}>None selected</span>
                }
              </div>
            </div>

            {/* Heterogeneity block */}
            <div className="sum-block het-block">
              <span className="sum-block-label">Heterogeneity</span>
              <span className="sum-het-val">
                {usePipeline ? "🧬 SMILES · RDKit · GEM" : "🚫 No Pipeline"}
              </span>
            </div>

            {/* Input dim block */}
            <div className="sum-block dim-block">
              <span className="sum-block-label">Input dimension</span>
              <span className="sum-dim-val">{inputDim}</span>
              <span className="sum-dim-sub">
                {usePipeline
                  ? `${selectedFeats.length} features → 230 (embedded)`
                  : `${selectedFeats.length} features × raw value`}
              </span>
            </div>

            {/* Models block */}
            <div className="sum-block model-block">
              <span className="sum-block-label">Selected models</span>
              <div className="sum-model-tags">
                {selected.length > 0
                  ? selected.map(id => {
                      const m = allModels.find(m => m.id === id);
                      const isTs = models.timeseries.some(tm => tm.id === id);
                      return (
                        <span key={id} className={`sum-model-tag ${isTs ? "sum-model-ts" : "sum-model-s"}`}>
                          {MODEL_ICONS[id]} {m?.name ?? id}
                        </span>
                      );
                    })
                  : <span style={{ fontSize: 10, color: "var(--text-muted)" }}>None</span>
                }
              </div>
            </div>

            {/* Run block */}
            <div className="sum-block run-block">
              <button className="sum-run-btn"
                onClick={handleTrain}
                disabled={status === "running" || !canTrain}>
                {status === "running" ? "⏳ Training..." : "▶ Train selected"}
              </button>
            </div>

          </div>
        </div>

      </div>

      {/* 결과 테이블 */}
      {resultRows.length > 0 && (
        <div className="card">
          <h3 style={{ fontSize: 13, fontWeight: 600, marginBottom: "0.75rem" }}>Model Comparison</h3>
          <table className="result-table">
            <thead>
              <tr><th>Model</th><th>R²</th><th>RMSE</th><th>CV R² (mean ± std)</th></tr>
            </thead>
            <tbody>
              {resultRows.map(m => (
                <tr key={m.id} className={m.id === bestR2Model?.id ? "best-row" : ""}>
                  <td style={{ display: "flex", alignItems: "center", gap: 6 }}>
                    <span>{MODEL_ICONS[m.id]}</span><span>{m.name}</span>
                    {m.id === bestR2Model?.id && <span className="best-badge">best</span>}
                  </td>
                  <td style={{ fontWeight: m.id === bestR2Model?.id ? 700 : 400 }}>{m.result.r2?.toFixed(4) ?? "—"}</td>
                  <td>{m.result.rmse?.toFixed(4) ?? m.result.titer_rmse?.toFixed(4) ?? "—"}</td>
                  <td>{m.result.cv_r2_mean !== undefined ? `${m.result.cv_r2_mean.toFixed(3)} ± ${m.result.cv_r2_std?.toFixed(3)}` : "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* 그래프 섹션 */}
      {Object.keys(allImages).length > 0 && (
        <div className="card">
          <h3 style={{ fontSize: 13, fontWeight: 600, marginBottom: "0.75rem" }}>Result Graphs</h3>
          <div className="graph-tab-bar">
            {Object.keys(allImages).map(mid => {
              const m = allModels.find(m => m.id === mid);
              return (
                <button key={mid} className={"graph-tab" + (selectedTab === mid ? " active" : "")}
                  onClick={() => setSelectedTab(mid)}>
                  {MODEL_ICONS[mid]} {m?.name ?? mid}
                </button>
              );
            })}
          </div>
          {selectedTab && allImages[selectedTab] && (
            <div className="graph-grid">
              {Object.entries(allImages[selectedTab]).map(([stem, url]) => (
                <div key={stem} className="graph-item">
                  <div className="graph-item-title">{stem.replace(/_/g, " ")}</div>
                  <img src={`${API}${url}`} alt={stem} style={{ width: "100%", borderRadius: 8 }} />
                </div>
              ))}
            </div>
          )}
        </div>
      )}

    </div>
  );
}