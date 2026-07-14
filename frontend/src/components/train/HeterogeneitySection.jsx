const PIPELINE_OPTIONS = [
  { id: "none",      icon: "🚫", name: "No Pipeline",   desc: "Raw concentration" },
  { id: "rdkit",     icon: "🧪", name: "SM-RD-GEM",     desc: "SMILES · RDKit · GEM" },
  { id: "chemberta", icon: "🤖", name: "SM-BERTA-GEM",  desc: "SMILES · ChemBERTa · GEM" },
  { id: "unimol",    icon: "🌐", name: "SM-UniMol-GEM", desc: "SMILES · UniMol · GEM" },
];

const CONCAT_OPTIONS = [
  { id: "log_conc",       name: "Conc. (log)" },
  { id: "metal_physchem", name: "Physical properties" },
  { id: "gem",             name: "GEM" },
];

const POOLING_OPTIONS = [
  { id: "mean",       name: "Mean" },
  { id: "multi_stat", name: "Mean + Weighted + Max + Count" },
];

export { PIPELINE_OPTIONS, CONCAT_OPTIONS, POOLING_OPTIONS };

export default function HeterogeneitySection({
  status,
  pipelineType, setPipelineType, pipelineDims,
  concatBlocks, toggleConcatBlock,
  poolingMethod, setPoolingMethod,
  pcaEnabled, setPcaEnabled,
  pcaDim, setPcaDim,
}) {
  const disabled = status === "running";

  return (
    <div className="train-section">
      <div className="train-section-head het">
        <span className="train-section-title">Heterogeneity</span>
      </div>

      {/* Pipeline (택1) */}
      <div className="model-grid" style={{ gridTemplateColumns: "repeat(4, 1fr)" }}>
        {PIPELINE_OPTIONS.map(opt => {
          const base = pipelineDims[opt.id]?.[poolingMethod];
          // concatBlocks(사용자 선택)에 따라 실제 차원을 프론트에서 직접 계산
          // (백엔드 pipelineDims는 "전체 블록 포함" 기준 값이므로 그대로 쓰면 안 됨)
          const embDim   = base?.embedding ?? 0;
          const physDim  = concatBlocks.includes("metal_physchem") ? 5 : 0;
          const concDim  = concatBlocks.includes("log_conc") ? 1 : 0;
          const gemDim   = concatBlocks.includes("gem") ? 7 : 0;
          const extraDim = physDim + concDim + gemDim;
          const pooledDim = poolingMethod === "multi_stat"
            ? embDim * 3 + extraDim + 1
            : embDim + extraDim;

          return (
            <div key={opt.id}
              className={"model-card gray" + (pipelineType === opt.id ? " selected gray" : "")}
              onClick={() => !disabled && setPipelineType(opt.id)}>
              {pipelineType === opt.id && <span className="model-check gray">✓</span>}
              <span className="model-icon">{opt.icon}</span>
              <span className="model-name">{opt.name}</span>
              <span className="model-desc">{opt.desc}</span>
              {opt.id !== "none" && (
                <div style={{ fontSize: 9, color: "#999", marginTop: 6, paddingTop: 6, borderTop: "0.5px solid #eee", width: "100%", textAlign: "center" }}>
                  {base ? (
                    <>
                      <div style={{ fontWeight: 700, color: "#555" }}>{pooledDim} dim</div>
                      <div>emb {embDim} · phys {physDim} · conc {concDim} · gem {gemDim}</div>
                    </>
                  ) : "로딩중…"}
                </div>
              )}
            </div>
          );
        })}
      </div>

      {pipelineType !== "none" && (
        <>
          {/* Feature concat (다중선택) */}
          <div style={{ marginTop: "1rem" }}>
            <div style={{ fontSize: 11, fontWeight: 700, color: "#1D9E75", textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: 8 }}>
              Feature concat <span style={{ color: "#888", fontWeight: 500, textTransform: "none" }}>(다중선택)</span>
            </div>
            <div className="model-grid" style={{ gridTemplateColumns: "repeat(3, 1fr)" }}>
              {CONCAT_OPTIONS.map(opt => (
                <div key={opt.id}
                  className={"model-card gray" + (concatBlocks.includes(opt.id) ? " selected gray" : "")}
                  onClick={() => !disabled && toggleConcatBlock(opt.id)}>
                  {concatBlocks.includes(opt.id) && <span className="model-check gray">✓</span>}
                  <span className="model-name" style={{ fontSize: 12 }}>{opt.name}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Pooling (택1) */}
          <div style={{ marginTop: "1rem" }}>
            <div style={{ fontSize: 11, fontWeight: 700, color: "#534AB7", textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: 8 }}>
              Pooling <span style={{ color: "#888", fontWeight: 500, textTransform: "none" }}>(택 1)</span>
            </div>
            <div className="model-grid" style={{ gridTemplateColumns: "repeat(2, 1fr)" }}>
              {POOLING_OPTIONS.map(opt => (
                <div key={opt.id}
                  className={"model-card gray" + (poolingMethod === opt.id ? " selected gray" : "")}
                  onClick={() => !disabled && setPoolingMethod(opt.id)}>
                  {poolingMethod === opt.id && <span className="model-check gray">✓</span>}
                  <span className="model-name" style={{ fontSize: 12 }}>{opt.name}</span>
                </div>
              ))}
            </div>
          </div>

          {/* PCA (택1) — On 카드 안, "On" 텍스트 옆에 목표 차원 입력 인라인 배치 */}
          <div style={{ marginTop: "1rem" }}>
            <div style={{ fontSize: 11, fontWeight: 700, color: "#D85A30", textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: 8 }}>
              PCA <span style={{ color: "#888", fontWeight: 500, textTransform: "none" }}>(택 1)</span>
            </div>
            <div style={{ display: "flex", gap: "0.6rem" }}>

              <div
                className={"model-card gray" + (pcaEnabled === false ? " selected gray" : "")}
                onClick={() => !disabled && setPcaEnabled(false)}
                style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center" }}>
                <span style={{ fontSize: 12, fontWeight: 700, lineHeight: 1 }}>Off</span>
              </div>

              <div
                className={"model-card gray" + (pcaEnabled === true ? " selected gray" : "")}
                onClick={() => !disabled && setPcaEnabled(true)}
                style={{ flex: 1, position: "relative", display: "flex", alignItems: "center", justifyContent: "center" }}>
                {pcaEnabled === true && <span className="model-check gray">✓</span>}
                <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 16 }}>
                  <span style={{ fontSize: 12, fontWeight: 700, lineHeight: 1 }}>On</span>
                  <input
                    type="number" min="2" max="100" step="1" value={pcaDim}
                    onClick={(e) => e.stopPropagation()}
                    onWheel={(e) => e.target.blur()}
                    onChange={(e) => {
                      const v = Number(e.target.value);
                      if (!Number.isNaN(v)) setPcaDim(v);
                    }}
                    disabled={disabled}
                    style={{
                      width: 38, height: 16, lineHeight: "16px", fontSize: 11, fontWeight: 700,
                      color: "#993C1D", textAlign: "center", border: "1px solid #D85A30",
                      borderRadius: 5, padding: 0, background: "#fff", boxSizing: "content-box",
                    }}
                  />
                </div>
              </div>

            </div>
          </div>

          <div className="st-note" style={{ marginTop: "0.75rem" }}>
            ⚠ 데이터가 적을 경우 성능 저하 가능 (임베딩·pooling에 따라 차원 상이)
          </div>
        </>
      )}
    </div>
  );
}