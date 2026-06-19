import { useState } from "react";
import StaticPanel from "../components/Staticpanel";
import TimePanel   from "../components/Timepanel";
import "./DataPage.css";

export default function DataPage() {
  const [activeTab, setActiveTab] = useState("static");

  return (
    <div className="data-page">
      <div className="tab-bar">
        <button
          className={`tab ${activeTab === "static" ? "active" : ""}`}
          onClick={() => setActiveTab("static")}
        >
          📊 Static Data
        </button>
        <button
          className={`tab ${activeTab === "ts" ? "active" : ""}`}
          onClick={() => setActiveTab("ts")}
        >
          📈 Time Series Data
        </button>
      </div>

      <div className="panel-wrap">
        {activeTab === "static" ? <StaticPanel /> : <TimePanel />}
      </div>
    </div>
  );
}