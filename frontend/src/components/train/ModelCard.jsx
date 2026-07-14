const MODEL_ICONS = {
  gaussian_process : "📈", xgboost: "⚡", random_forest: "🌲", mlp: "🧠",
  rnn: "🔁", lstm: "⏱️", transformer: "⚙️", tcn: "🌊", static_time_gnn: "🕸️",
};

export { MODEL_ICONS };

export default function ModelCard({ m, type, selected, status, onToggle, result }) {
  const isSelected = selected.includes(m.id);
  return (
    <div className={"model-card" + (isSelected ? ` selected ${type}` : "")}
      onClick={() => status !== "running" && onToggle(m.id)}>
      {isSelected && <span className={`model-check ${type}`}>✓</span>}
      <span className="model-icon">{MODEL_ICONS[m.id]}</span>
      <span className="model-name">{m.name}</span>
      <span className="model-desc">{m.desc}</span>
      {m.has_model
        ? <span className="model-badge saved">{m.model_file}</span>
        : <span className="model-badge none">no saved model</span>}
      {result && (
        <span className="model-r2">
          {result.r2 !== undefined
            ? `R² ${result.r2.toFixed(3)}`
            : `RMSE ${result.titer_rmse ?? result.rmse ?? "—"}`}
        </span>
      )}
    </div>
  );
}