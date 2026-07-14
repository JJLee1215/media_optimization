import { MODEL_ICONS } from "./ModelCard";

export default function ResultsSection({ resultRows, bestR2Model, allImages, allModels, selectedTab, setSelectedTab }) {
  const API = "http://localhost:8000";
  return (
    <>
      {resultRows.length > 0 && (
        <div className="card">
          <h3 style={{ fontSize: 13, fontWeight: 600, marginBottom: "0.75rem" }}>Model Comparison</h3>
          <table className="result-table">
            <thead><tr><th>Model</th><th>R²</th><th>RMSE</th><th>CV R² (mean ± std)</th></tr></thead>
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
    </>
  );
}