export default function FeatureSelectionSection({
  staticFile, availableFeats, selectedFeats, setSelectedFeats, toggleFeat, staticColor,
  tsFile, availableTsFeats, selectedTsFeats, setSelectedTsFeats, tsColor,
}) {
  return (
    <div className="train-section">
      <div className="train-section-head feat">
        <span className="train-section-title">Feature Selection</span>
      </div>

      {/* Static 서브섹션 */}
      <div className={"feat-sub-section" + (staticColor ? ` lit ${staticColor}` : "")}>
        <div className="feat-sub-head">
          <span className={"feat-sub-dot" + (staticColor ? " active" : "")} />
          <span className={"feat-sub-title" + (staticColor ? " active" : "")}>Static</span>
          {staticFile && (
            <span className={"feat-sub-count" + (staticColor ? " in-use" : "")}>
              {selectedFeats.length} / {availableFeats.length}
            </span>
          )}
          {staticFile && (
            <div style={{ marginLeft: "auto", display: "flex", gap: 4 }}>
              <button className="log-clear" onClick={() => setSelectedFeats(availableFeats)}>All</button>
              <button className="log-clear" onClick={() => setSelectedFeats([])}>None</button>
            </div>
          )}
        </div>
        {!staticFile ? (
          <div className="feat-empty-note">파일을 업로드하면 feature가 표시됩니다</div>
        ) : (
          <div className="feat-tag-area">
            {availableFeats.map(f => (
              <div key={f}
                className={"feat-tag" + (selectedFeats.includes(f) ? " on" : "")}
                onClick={() => toggleFeat(f)}>
                {selectedFeats.includes(f) && <span style={{ color: "#1D9E75" }}>✓ </span>}{f}
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Timeseries 서브섹션 */}
      <div className={"feat-sub-section" + (tsColor ? ` lit ${tsColor}` : "")} style={{ marginTop: "0.5rem" }}>
        <div className="feat-sub-head">
          <span className={"feat-sub-dot" + (tsColor ? " active" : "")} />
          <span className={"feat-sub-title" + (tsColor ? " active" : "")}>Timeseries</span>
          {tsFile && (
            <span className={"feat-sub-count" + (tsColor ? " in-use" : "")}>
              {selectedTsFeats.length} / {availableTsFeats.length}
            </span>
          )}
          {tsFile && (
            <div style={{ marginLeft: "auto", display: "flex", gap: 4 }}>
              <button className="log-clear" onClick={() => setSelectedTsFeats(availableTsFeats)}>All</button>
              <button className="log-clear" onClick={() => setSelectedTsFeats([])}>None</button>
            </div>
          )}
        </div>
        {!tsFile ? (
          <div className="feat-empty-note">파일을 업로드하면 feature가 표시됩니다</div>
        ) : (
          <div className="feat-tag-area">
            {availableTsFeats.map(f => (
              <div key={f}
                className={"feat-tag" + (selectedTsFeats.includes(f) ? " on" : "")}
                onClick={() => setSelectedTsFeats(prev =>
                  prev.includes(f) ? prev.filter(x => x !== f) : [...prev, f]
                )}>
                {selectedTsFeats.includes(f) && <span style={{ color: "#1D9E75" }}>✓ </span>}{f}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}