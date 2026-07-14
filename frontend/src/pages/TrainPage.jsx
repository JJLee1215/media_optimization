import { useState, useEffect, useRef } from "react";
import axios from "axios";
import "./TrainPage.css";

import FileUploadItem from "../components/train/FileUploadItem";
import FeatureSelectionSection from "../components/train/FeatureSelectionSection";
import HeterogeneitySection from "../components/train/HeterogeneitySection";
import ModelSections from "../components/train/ModelSections";
import SummaryPanel from "../components/train/SummaryPanel";
import ResultsSection from "../components/train/ResultsSection";

const API = "http://localhost:8000";
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

  // Heterogeneity
  const [pipelineType,    setPipelineType]    = useState("none");
  const [pipelineDims,    setPipelineDims]    = useState({});
  const [concatBlocks,    setConcatBlocks]    = useState(["log_conc", "metal_physchem", "gem"]);
  const [poolingMethod,   setPoolingMethod]   = useState("mean");
  const [pcaEnabled,      setPcaEnabled]      = useState(false);
  const [pcaDim,          setPcaDim]          = useState(30);

  // Feature selection
  const [availableFeats,   setAvailableFeats]   = useState([]);
  const [selectedFeats,    setSelectedFeats]    = useState([]);
  const [availableTsFeats, setAvailableTsFeats] = useState([]);
  const [selectedTsFeats,  setSelectedTsFeats]  = useState([]);

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
    axios.get(`${API}/train/pipeline_dims`).then(({ data }) => setPipelineDims(data)).catch(() => {});
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

  const handleStaticUpload = async (e) => {
    const f = e.target.files[0]; if (!f) return;
    setStaticUploading(true);
    try {
      const form = new FormData(); form.append("file", f);
      const { data } = await axios.post(`${API}/data/upload`, form);
      setStaticFile(data.filename);
      const { data: colData } = await axios.get(
        `${API}/data/analyze/static?type=stats&filepath=${data.filename}`
      );
      const cols = Object.keys(colData).filter(k => !["Batch_ID","titer_final","viab_final"].includes(k));
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
      const { data: colData } = await axios.get(
        `${API}/data/columns?filename=${data.filename}&type=timeseries`
      );
      setAvailableTsFeats(colData.columns ?? []);
      setSelectedTsFeats(colData.columns ?? []);
    } catch (e) {} finally { setTsUploading(false); }
  };

  const clearStaticFile = () => {
    setStaticFile(""); setAvailableFeats([]); setSelectedFeats([]);
    if (staticInputRef.current) staticInputRef.current.value = "";
  };
  const clearTsFile = () => {
    setTsFile(""); setAvailableTsFeats([]); setSelectedTsFeats([]);
    if (tsInputRef.current) tsInputRef.current.value = "";
  };

  const toggleModel = (id) => {
    setSelected(prev => prev.includes(id) ? prev.filter(m => m !== id) : [...prev, id]);
  };
  const toggleFeat = (f) => {
    setSelectedFeats(prev => prev.includes(f) ? prev.filter(x => x !== f) : [...prev, f]);
  };
  const toggleConcatBlock = (id) => {
    setConcatBlocks(prev => prev.includes(id) ? prev.filter(x => x !== id) : [...prev, id]);
  };

  const classifyLog = (text) => {
    if (text.includes("✅") || text.includes("complete") || text.includes("Saved")) return "green";
    if (text.includes("❌") || text.includes("Error")) return "red";
    if (text.startsWith("▶")) return "yellow";
    if (text.startsWith("═")) return "divider";
    if (text.startsWith("[preprocess]")) return "muted";
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
        model             : modelId,
        use_pipeline      : pipelineType !== "none",
        static_file       : staticFile || null,
        ts_file           : tsFile     || null,
        selected_cols     : selectedFeats.length   > 0 ? selectedFeats   : null,
        selected_ts_cols  : selectedTsFeats.length > 0 ? selectedTsFeats : null,
        embedding_model   : pipelineType !== "none" ? pipelineType   : null,
        other_blocks      : pipelineType !== "none" ? concatBlocks   : null,
        pooling_method    : pipelineType !== "none" ? poolingMethod  : null,
        use_pca           : pipelineType !== "none" ? pcaEnabled     : null,
        pca_dim           : pipelineType !== "none" && pcaEnabled ? pcaDim : null,
        notation          : pipelineType !== "none" ? "smiles"       : null,
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

  // ── Input dimension ──
  // pipelineDims 구조: { rdkit: { mean: {...}, multi_stat: {...} }, chemberta: {...} }
  // PCA가 켜져있으면 최종 차원은 pcaDim(사용자가 조절), 꺼져있으면 pooling 직후 차원(pooled_dim)
  // concatBlocks + poolingMethod + pcaEnabled를 모두 반영해서 최종 차원 계산
  const baseDimInfo = pipelineDims[pipelineType]?.[poolingMethod];
  const embDim   = baseDimInfo?.embedding ?? 0;
  const physDim  = concatBlocks.includes("metal_physchem") ? 5 : 0;
  const concDim  = concatBlocks.includes("log_conc") ? 1 : 0;
  const gemDim   = concatBlocks.includes("gem") ? 7 : 0;
  const extraDim = physDim + concDim + gemDim;
  const pooledDim = poolingMethod === "multi_stat"
    ? embDim * 3 + extraDim + 1
    : embDim + extraDim;

  const inputDim = pipelineType !== "none"
    ? (pcaEnabled ? pcaDim : (baseDimInfo ? pooledDim : "…"))
    : selectedFeats.length + selectedTsFeats.length;

  const usesStaticTime = selected.some(id => models.static_time.some(m => m.id === id));
  const usesStaticOnly = selected.some(id => models.static.some(m => m.id === id));
  const usesTsOnly     = selected.some(id => models.timeseries.some(m => m.id === id));
  const staticColor = usesStaticTime ? "st" : usesStaticOnly ? "static" : null;
  const tsColor      = usesStaticTime ? "st" : usesTsOnly     ? "ts"     : null;

  return (
    <div className="train-page">

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
            : logs.map((l, i) => <span key={i} className={`log-line ${l.cls}`}>{l.text}<br /></span>)}
        </div>
      </div>

      <div className="train-main-grid">
        <div className="train-left">
          <FeatureSelectionSection
            staticFile={staticFile} availableFeats={availableFeats} selectedFeats={selectedFeats}
            setSelectedFeats={setSelectedFeats} toggleFeat={toggleFeat} staticColor={staticColor}
            tsFile={tsFile} availableTsFeats={availableTsFeats} selectedTsFeats={selectedTsFeats}
            setSelectedTsFeats={setSelectedTsFeats} tsColor={tsColor}
          />

          <HeterogeneitySection
            status={status}
            pipelineType={pipelineType} setPipelineType={setPipelineType} pipelineDims={pipelineDims}
            concatBlocks={concatBlocks} toggleConcatBlock={toggleConcatBlock}
            poolingMethod={poolingMethod} setPoolingMethod={setPoolingMethod}
            pcaEnabled={pcaEnabled} setPcaEnabled={setPcaEnabled}
            pcaDim={pcaDim} setPcaDim={setPcaDim}
          />

          <ModelSections models={models} selected={selected} status={status}
            onToggle={toggleModel} allResults={allResults} />
        </div>

        <SummaryPanel
          selectedFeats={selectedFeats} availableFeats={availableFeats}
          selectedTsFeats={selectedTsFeats} availableTsFeats={availableTsFeats}
          pipelineType={pipelineType} inputDim={inputDim}
          poolingMethod={poolingMethod} pcaEnabled={pcaEnabled} pcaDim={pcaDim}
          selected={selected} allModels={allModels} models={models}
          status={status} canTrain={canTrain} onTrain={handleTrain}
        />
      </div>

      <ResultsSection
        resultRows={resultRows} bestR2Model={bestR2Model} allImages={allImages}
        allModels={allModels} selectedTab={selectedTab} setSelectedTab={setSelectedTab}
      />
    </div>
  );
}