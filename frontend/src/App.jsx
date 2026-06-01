import { BrowserRouter, Routes, Route, NavLink } from "react-router-dom";
import TrainPage from "./pages/TrainPage";
import TestPage from "./pages/TestPage";
import "./App.css";

export default function App() {
  return (
    <BrowserRouter>
      <div className="app">
        <nav className="navbar">
          <span className="nav-title">🧬 Bioprocess Engineering Tool</span>
          <div className="nav-links">
            <NavLink to="/" end className={({ isActive }) => isActive ? "nav-link active" : "nav-link"}>
              Training
            </NavLink>
            <NavLink to="/test" className={({ isActive }) => isActive ? "nav-link active" : "nav-link"}>
              Test
            </NavLink>
          </div>
        </nav>
        <main className="main">
          <Routes>
            <Route path="/" element={<TrainPage />} />
            <Route path="/test" element={<TestPage />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}