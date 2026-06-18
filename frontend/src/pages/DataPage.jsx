import Split from "react-split";
import StaticPanel from "../components/Staticpanel";
import TimePanel   from "../components/Timepanel";
import "./DataPage.css";

export default function DataPage() {
  return (
    <div className="data-page">
      <h1 className="page-title">Data Analysis</h1>
      <Split
        className="data-layout"
        sizes={[50, 50]}
        minSize={280}
        gutterSize={8}
        direction="horizontal"
      >
        <div className="panel-wrap"><StaticPanel /></div>
        <div className="panel-wrap"><TimePanel /></div>
      </Split>
    </div>
  );
}