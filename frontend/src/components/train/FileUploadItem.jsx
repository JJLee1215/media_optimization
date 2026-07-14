export default function FileUploadItem({ label, file, uploading, inputRef, onUpload, onClear, color, id }) {
  return (
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
}