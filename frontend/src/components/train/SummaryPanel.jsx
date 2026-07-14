import { MODEL_ICONS } from "./ModelCard";

const PIPELINE_LABEL = {
  none: "🚫 No Pipeline", rdkit: "🧪 SM-RD-GEM",
  chemberta: "🤖 SM-BERTA-GEM", unimol: "🌐 SM-UniMol-GEM",
};

const POOLING_LABEL = {
  mean: "Mean",
  multi_stat: "Mean+Weighted+Max+Count",
};

export default function SummaryPanel({
  selectedFeats, availableFeats, selectedTsFeats, availableTsFeats,
  pipelineType, inputDim, poolingMethod, pcaEnabled, pcaDim,
  selected, allModels, models,
  status, canTrain, onTrain,
}) {
  return (
    <div className="train-right">
      <div className="summary-wrap">

        <div className="sum-block feat-block">
          <div className="sum-block-header">
            <span className="sum-block-label">Features</span>
            <span className="sum-feat-count">
              {selectedFeats.length + selectedTsFeats.length} / {availableFeats.length + availableTsFeats.length}
            </span>
          </div>

          <div className="sum-feat-group">
            <div className="sum-block-header">
              <span className="sum-feat-sublabel">Static</span>
              {availableFeats.length > 0 && (
                <span className="sum-feat-subcount">{selectedFeats.length} / {availableFeats.length}</span>
              )}
            </div>
            <div className="sum-feat-tags">
              {selectedFeats.length > 0
                ? selectedFeats.map(f => <span key={f} className="sum-feat-tag">{f.replace("_0", "")}</span>)
                : <span style={{ fontSize: 10, color: "var(--text-muted)" }}>None selected</span>}
            </div>
          </div>

          <div className="sum-feat-group">
            <div className="sum-block-header">
              <span className="sum-feat-sublabel">Timeseries</span>
              {availableTsFeats.length > 0 && (
                <span className="sum-feat-subcount">{selectedTsFeats.length} / {availableTsFeats.length}</span>
              )}
            </div>
            <div className="sum-feat-tags">
              {selectedTsFeats.length > 0
                ? selectedTsFeats.map(f => <span key={f} className="sum-feat-tag">{f}</span>)
                : <span style={{ fontSize: 10, color: "var(--text-muted)" }}>None selected</span>}
            </div>
          </div>
        </div>

        <div className="sum-block het-block">
          <span className="sum-block-label">Heterogeneity</span>
          <span className="sum-het-val">{PIPELINE_LABEL[pipelineType]}</span>
          {pipelineType !== "none" && (
            <div style={{ fontSize: 10, color: "var(--text-muted)", marginTop: 4 }}>
              Pooling: {POOLING_LABEL[poolingMethod]} · PCA: {pcaEnabled ? `On (${pcaDim}dim)` : "Off"}
            </div>
          )}
        </div>

        <div className="sum-block dim-block">
          <span className="sum-block-label">Input dimension</span>
          <span className="sum-dim-val">{inputDim}</span>
          <span className="sum-dim-sub">
            {pipelineType !== "none"
              ? `${selectedFeats.length + selectedTsFeats.length} features → embedded (${PIPELINE_LABEL[pipelineType]})`
              : `${selectedFeats.length + selectedTsFeats.length} features × raw value`}
          </span>
        </div>

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
              : <span style={{ fontSize: 10, color: "var(--text-muted)" }}>None</span>}
          </div>
        </div>

        <div className="sum-block run-block">
          <button className="sum-run-btn" onClick={onTrain} disabled={status === "running" || !canTrain}>
            {status === "running" ? "⏳ Training..." : "▶ Train selected"}
          </button>
        </div>

      </div>
    </div>
  );
}