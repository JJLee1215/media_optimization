import { BrowserRouter, Routes, Route, NavLink } from "react-router-dom";
import DataPage     from "./pages/DataPage";
import TrainPage    from "./pages/TrainPage";
import PredictPage  from "./pages/PredictPage";
import OptimizePage from "./pages/Optimizepage";
import "./App.css";

export default function App() {
  return (
    <BrowserRouter>
      <div className="app">
        <nav className="sidebar">
          <div className="sidebar-title">🧬 Bioprocess ML</div>
          <NavLink to="/"         end className={({ isActive }) => isActive ? "nav-link active" : "nav-link"}>📊 Data Analysis</NavLink>
          <NavLink to="/train"        className={({ isActive }) => isActive ? "nav-link active" : "nav-link"}>🏋️ Model Train</NavLink>
          <NavLink to="/predict"      className={({ isActive }) => isActive ? "nav-link active" : "nav-link"}>🔮 Prediction</NavLink>
          <NavLink to="/optimize"     className={({ isActive }) => isActive ? "nav-link active" : "nav-link"}>⚡ Optimization</NavLink>
        </nav>
        <main className="main">
          <Routes>
            <Route path="/"        element={<DataPage />} />
            <Route path="/train"   element={<TrainPage />} />
            <Route path="/predict" element={<PredictPage />} />
            <Route path="/optimize" element={<OptimizePage />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}