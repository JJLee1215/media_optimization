import ModelCard from "./ModelCard";

export default function ModelSections({ models, selected, status, onToggle, allResults }) {
  return (
    <>
      <div className="train-section">
        <div className="train-section-head static">
          <span className="train-section-title">Static models</span>
          <span className="section-badge static" style={{ marginLeft: 4 }}>batch_table_syn.csv</span>
        </div>
        <div className="model-grid">
          {models.static.map(m => (
            <ModelCard key={m.id} m={m} type="static" selected={selected} status={status}
              onToggle={onToggle} result={allResults[m.id]} />
          ))}
        </div>
      </div>

      <div className="train-section">
        <div className="train-section-head ts">
          <span className="train-section-title">Time Series models</span>
          <span className="section-badge ts" style={{ marginLeft: 4 }}>timeseries_syn.csv</span>
        </div>
        <div className="model-grid">
          {models.timeseries.map(m => (
            <ModelCard key={m.id} m={m} type="ts" selected={selected} status={status}
              onToggle={onToggle} result={allResults[m.id]} />
          ))}
        </div>
      </div>

      <div className="train-section">
        <div className="train-section-head st">
          <span className="train-section-title">Static &amp; Time Series models</span>
        </div>
        <div className="st-note">두 파일이 모두 선택되어야 학습 가능합니다</div>
        <div className="model-grid" style={{ marginTop: "0.5rem" }}>
          {models.static_time.map(m => (
            <ModelCard key={m.id} m={m} type="st" selected={selected} status={status}
              onToggle={onToggle} result={allResults[m.id]} />
          ))}
        </div>
      </div>
    </>
  );
}