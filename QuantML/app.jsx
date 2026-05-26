/**
 * QuantML React Frontend
 * ======================
 * Connects to the FastAPI backend (main.py) at VITE_API_URL.
 *
 * Tabs: Dashboard · Model Compare · Analytics · Forecast · About
 *
 * Deploy to Vercel:
 *   1. `npm create vite@latest quantml -- --template react`
 *   2. Replace src/App.jsx with this file
 *   3. npm install recharts lucide-react
 *   4. Add VITE_API_URL=https://your-api.com to Vercel env vars
 *   5. `vercel deploy`
 */

import { useState, useCallback, useRef } from "react";
import {
  LineChart, Line, BarChart, Bar, XAxis, YAxis,
  CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  ReferenceLine, ComposedChart, Area,
} from "recharts";
import {
  Upload, TrendingUp, TrendingDown, BarChart2,
  Activity, Cpu, Info, ChevronRight, Loader2,
  RefreshCw, Zap,
} from "lucide-react";

// ── Config ────────────────────────────────────────────────────────────────────
const API = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

const COLOURS = {
  "XGBoost":           "#00d4aa",
  "Random Forest":     "#3b82f6",
  "Decision Tree":     "#f59e0b",
  "Linear Regression": "#6b7a8a",
  actual:              "#e2e8f4",
  grid:                "#1e2a3a",
  bg:                  "#080c15",
  surface:             "#0d1220",
};

const MODEL_LIST = ["xgboost", "random_forest", "decision_tree", "linear_regression"];
const MODEL_LABEL = {
  xgboost:           "XGBoost",
  random_forest:     "Random Forest",
  decision_tree:     "Decision Tree",
  linear_regression: "Linear Regression",
};

// ── Styles (injected once) ────────────────────────────────────────────────────
const GLOBAL_CSS = `
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap');

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg:       #0a0e17;
    --surface:  #0d1220;
    --surface2: #111827;
    --border:   #1e2a3a;
    --teal:     #00d4aa;
    --teal-dim: rgba(0,212,170,0.12);
    --blue:     #3b82f6;
    --amber:    #f59e0b;
    --red:      #ef4444;
    --green:    #22c55e;
    --text:     #e2e8f4;
    --muted:    #4a5a6a;
    --font-display: 'Syne', sans-serif;
    --font-mono:    'JetBrains Mono', monospace;
  }

  html, body, #root {
    height: 100%;
    background: var(--bg);
    color: var(--text);
    font-family: var(--font-mono);
    font-size: 13px;
    -webkit-font-smoothing: antialiased;
  }

  ::-webkit-scrollbar { width: 4px; }
  ::-webkit-scrollbar-track { background: var(--bg); }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

  .layout { display: flex; height: 100vh; overflow: hidden; }

  /* Sidebar */
  .sidebar {
    width: 220px;
    min-width: 220px;
    background: var(--surface);
    border-right: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    padding: 24px 0;
    gap: 4px;
    overflow-y: auto;
  }
  .sidebar-logo {
    padding: 0 20px 20px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 12px;
  }
  .sidebar-logo h1 {
    font-family: var(--font-display);
    font-size: 22px;
    font-weight: 800;
    color: var(--teal);
    letter-spacing: -0.5px;
  }
  .sidebar-logo p {
    font-size: 10px;
    color: var(--muted);
    margin-top: 2px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
  }
  .nav-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 20px;
    cursor: pointer;
    color: var(--muted);
    font-size: 12px;
    font-family: var(--font-mono);
    transition: all 0.15s;
    border-left: 2px solid transparent;
    letter-spacing: 0.03em;
  }
  .nav-item:hover { color: var(--text); background: rgba(255,255,255,0.03); }
  .nav-item.active {
    color: var(--teal);
    border-left-color: var(--teal);
    background: var(--teal-dim);
  }
  .sidebar-section {
    padding: 16px 20px 8px;
    font-size: 9px;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.12em;
  }

  /* Upload zone */
  .upload-zone {
    margin: 0 12px;
    border: 1px dashed var(--border);
    border-radius: 8px;
    padding: 14px 10px;
    text-align: center;
    cursor: pointer;
    transition: all 0.2s;
    color: var(--muted);
    font-size: 11px;
    line-height: 1.5;
  }
  .upload-zone:hover, .upload-zone.drag { border-color: var(--teal); color: var(--teal); background: var(--teal-dim); }
  .upload-zone input { display: none; }

  /* Controls */
  .ctrl { margin: 0 12px; }
  .ctrl label { display: block; font-size: 10px; color: var(--muted); margin-bottom: 5px; text-transform: uppercase; letter-spacing: 0.08em; }
  .ctrl select, .ctrl input[type=range] { width: 100%; background: var(--surface2); border: 1px solid var(--border); color: var(--text); border-radius: 5px; padding: 6px 8px; font-family: var(--font-mono); font-size: 11px; outline: none; }
  .ctrl select:focus { border-color: var(--teal); }
  input[type=range] { -webkit-appearance: none; height: 4px; border-radius: 2px; background: var(--border) !important; padding: 0 !important; cursor: pointer; }
  input[type=range]::-webkit-slider-thumb { -webkit-appearance: none; width: 12px; height: 12px; border-radius: 50%; background: var(--teal); }
  .range-val { text-align: right; font-size: 11px; color: var(--teal); margin-top: 4px; }

  /* Buttons */
  .btn {
    display: flex; align-items: center; justify-content: center; gap: 7px;
    width: 100%; padding: 9px; border-radius: 6px; border: none; cursor: pointer;
    font-family: var(--font-mono); font-size: 12px; font-weight: 500;
    transition: all 0.15s;
  }
  .btn-primary { background: var(--teal); color: #041511; }
  .btn-primary:hover { background: #00bfa0; }
  .btn-primary:disabled { opacity: 0.4; cursor: not-allowed; }
  .btn-outline { background: transparent; color: var(--teal); border: 1px solid var(--teal); }
  .btn-outline:hover { background: var(--teal-dim); }
  .ctrl-stack { display: flex; flex-direction: column; gap: 8px; margin: 0 12px; }

  /* Main content */
  .main { flex: 1; overflow-y: auto; display: flex; flex-direction: column; }
  .tab-content { flex: 1; padding: 28px 32px; }

  /* KPI cards */
  .kpi-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; margin-bottom: 24px; }
  .kpi {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 14px 16px;
    position: relative;
    overflow: hidden;
  }
  .kpi::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, var(--teal), transparent);
  }
  .kpi-label { font-size: 10px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 6px; }
  .kpi-value { font-family: var(--font-display); font-size: 20px; font-weight: 700; color: var(--text); }
  .kpi-delta { font-size: 11px; margin-top: 3px; }
  .kpi-delta.pos { color: var(--green); }
  .kpi-delta.neg { color: var(--red); }

  /* Chart card */
  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 20px 24px;
    margin-bottom: 20px;
  }
  .card-title {
    font-family: var(--font-display);
    font-size: 13px;
    font-weight: 600;
    color: var(--text);
    margin-bottom: 16px;
    letter-spacing: 0.02em;
  }
  .card-title span { color: var(--muted); font-weight: 400; font-size: 11px; margin-left: 8px; }

  /* Table */
  .data-table { width: 100%; border-collapse: collapse; font-size: 12px; }
  .data-table th { padding: 8px 12px; text-align: left; color: var(--muted); font-size: 10px; text-transform: uppercase; letter-spacing: 0.08em; border-bottom: 1px solid var(--border); }
  .data-table td { padding: 10px 12px; border-bottom: 1px solid rgba(30,42,58,0.5); }
  .data-table tr:last-child td { border-bottom: none; }
  .data-table tr:hover td { background: rgba(255,255,255,0.02); }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 10px; font-weight: 600; }
  .badge-teal  { background: var(--teal-dim); color: var(--teal); }
  .badge-blue  { background: rgba(59,130,246,0.12); color: var(--blue); }
  .badge-amber { background: rgba(245,158,11,0.12); color: var(--amber); }
  .badge-grey  { background: rgba(107,122,138,0.15); color: #6b7a8a; }

  /* Status */
  .status-bar {
    padding: 10px 32px;
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    font-size: 11px;
    color: var(--muted);
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .status-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--muted); }
  .status-dot.ready { background: var(--green); box-shadow: 0 0 6px var(--green); }
  .status-dot.loading { background: var(--amber); animation: pulse 1s infinite; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.3} }

  /* Empty / Info */
  .empty {
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    height: 260px; gap: 12px; color: var(--muted);
  }
  .empty svg { opacity: 0.3; }
  .empty p { font-size: 12px; }

  /* Forecast side metrics */
  .fc-metrics { display: flex; flex-direction: column; gap: 12px; }
  .fc-metric { background: var(--surface2); border: 1px solid var(--border); border-radius: 8px; padding: 12px 14px; }
  .fc-metric-label { font-size: 10px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 4px; }
  .fc-metric-value { font-family: var(--font-display); font-size: 18px; font-weight: 700; }

  /* About */
  .about-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
  .about-section h3 { font-family: var(--font-display); font-size: 13px; font-weight: 700; color: var(--teal); margin-bottom: 10px; text-transform: uppercase; letter-spacing: 0.05em; }
  .about-section p, .about-section li { font-size: 12px; color: #8a9ab0; line-height: 1.7; }
  .about-section ul { list-style: none; }
  .about-section ul li::before { content: '→ '; color: var(--teal); }
  .code-block { background: var(--surface2); border: 1px solid var(--border); border-radius: 6px; padding: 12px 16px; font-family: var(--font-mono); font-size: 11px; color: var(--teal); margin-top: 8px; white-space: pre; overflow-x: auto; }

  /* Divider */
  .divider { border: none; border-top: 1px solid var(--border); margin: 20px 0; }

  /* Two-col layout */
  .two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }

  /* Spinner */
  .spin { animation: spin 1s linear infinite; }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* Tooltip override for recharts */
  .recharts-tooltip-wrapper .recharts-default-tooltip {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    font-family: var(--font-mono) !important;
    font-size: 11px !important;
  }
`;

// ── Inject CSS ─────────────────────────────────────────────────────────────────
if (!document.getElementById("qml-styles")) {
  const s = document.createElement("style");
  s.id = "qml-styles";
  s.textContent = GLOBAL_CSS;
  document.head.appendChild(s);
}

// ── Helpers ───────────────────────────────────────────────────────────────────
const fmt   = (n, d = 2) => n == null ? "—" : Number(n).toFixed(d);
const fmtUSD = n => n == null ? "—" : `$${Number(n).toLocaleString("en-US", { minimumFractionDigits: 2 })}`;

function modelBadgeClass(name) {
  if (name.includes("XGBoost") || name.includes("xgboost")) return "badge-teal";
  if (name.includes("Forest")  || name.includes("forest"))  return "badge-blue";
  if (name.includes("Tree")    || name.includes("tree"))    return "badge-amber";
  return "badge-grey";
}

const CHART_PROPS = {
  margin: { top: 8, right: 16, left: 0, bottom: 0 },
};

const TooltipStyle = {
  contentStyle: {
    background: "#111827", border: "1px solid #1e2a3a",
    borderRadius: 6, fontFamily: "'JetBrains Mono', monospace", fontSize: 11,
  },
  labelStyle: { color: "#4a5a6a" },
};

// ── Components ────────────────────────────────────────────────────────────────

function KPI({ label, value, delta, deltaClass }) {
  return (
    <div className="kpi">
      <div className="kpi-label">{label}</div>
      <div className="kpi-value">{value}</div>
      {delta && <div className={`kpi-delta ${deltaClass ?? ""}`}>{delta}</div>}
    </div>
  );
}

function Card({ title, sub, children }) {
  return (
    <div className="card">
      {title && <div className="card-title">{title}{sub && <span>{sub}</span>}</div>}
      {children}
    </div>
  );
}

function Empty({ icon: Icon = BarChart2, text = "Upload data to get started" }) {
  return (
    <div className="empty">
      <Icon size={40} />
      <p>{text}</p>
    </div>
  );
}

function Spinner() {
  return <Loader2 size={16} className="spin" />;
}

// Custom Recharts tick
function MonoTick({ x, y, payload }) {
  return (
    <text x={x} y={y + 12} fill="#4a5a6a" fontSize={10}
          fontFamily="JetBrains Mono, monospace" textAnchor="middle">
      {payload.value}
    </text>
  );
}

// ── API calls ─────────────────────────────────────────────────────────────────

async function apiUploadPredict(file, modelName) {
  const fd = new FormData();
  fd.append("file", file);
  fd.append("model_name", modelName);
  const res = await fetch(`${API}/upload-and-predict`, { method: "POST", body: fd });
  if (!res.ok) throw new Error((await res.json()).detail ?? res.statusText);
  return res.json();
}

async function apiCompareAll(file) {
  const fd = new FormData();
  fd.append("file", file);
  const res = await fetch(`${API}/compare-all`, { method: "POST", body: fd });
  if (!res.ok) throw new Error((await res.json()).detail ?? res.statusText);
  return res.json();
}

async function apiForecast(file, modelName, nDays) {
  const fd = new FormData();
  fd.append("file", file);
  fd.append("model_name", modelName);
  fd.append("n_days", nDays);
  const res = await fetch(`${API}/forecast`, { method: "POST", body: fd });
  if (!res.ok) throw new Error((await res.json()).detail ?? res.statusText);
  return res.json();
}

async function apiMetrics() {
  const res = await fetch(`${API}/metrics`);
  if (!res.ok) throw new Error("No saved metrics found");
  return res.json();
}

// ── Main App ──────────────────────────────────────────────────────────────────

export default function App() {
  const [tab, setTab]             = useState("dashboard");
  const [file, setFile]           = useState(null);
  const [modelName, setModelName] = useState("xgboost");
  const [forecastDays, setFDays]  = useState(30);
  const [drag, setDrag]           = useState(false);
  const [loading, setLoading]     = useState(false);
  const [status, setStatus]       = useState("Upload a CSV to begin");

  // Data state
  const [prediction,  setPrediction]  = useState(null);
  const [allResults,  setAllResults]  = useState(null);
  const [forecastData, setForecast]   = useState(null);
  const [error,       setError]       = useState(null);

  const fileRef = useRef();

  // ── File handling ───────────────────────────────────────────────────────────
  const handleFile = useCallback(f => {
    if (!f || !f.name.endsWith(".csv")) { setError("Only CSV files supported."); return; }
    setFile(f);
    setStatus(`File ready: ${f.name} (${(f.size / 1024).toFixed(1)} KB)`);
    setError(null);
    setPrediction(null);
    setAllResults(null);
    setForecast(null);
  }, []);

  const onDrop = useCallback(e => {
    e.preventDefault(); setDrag(false);
    handleFile(e.dataTransfer.files[0]);
  }, [handleFile]);

  // ── Actions ─────────────────────────────────────────────────────────────────
  const runPredict = async () => {
    if (!file) return;
    setLoading(true); setError(null);
    setStatus(`Running ${MODEL_LABEL[modelName]}...`);
    try {
      const res = await apiUploadPredict(file, modelName);
      setPrediction(res);
      setStatus(`Done — R² ${fmt(res.metrics.R2, 4)}`);
    } catch (e) {
      setError(e.message);
      setStatus("Error during prediction");
    } finally { setLoading(false); }
  };

  const runCompare = async () => {
    if (!file) return;
    setLoading(true); setError(null);
    setStatus("Comparing all 4 models...");
    try {
      const res = await apiCompareAll(file);
      setAllResults(res);
      setStatus("All models compared");
    } catch (e) {
      setError(e.message);
      setStatus("Error during comparison");
    } finally { setLoading(false); }
  };

  const runForecast = async () => {
    if (!file) return;
    setLoading(true); setError(null);
    setStatus(`Forecasting ${forecastDays} days...`);
    try {
      const res = await apiForecast(file, modelName, forecastDays);
      setForecast(res);
      setStatus(`Forecast ready — ${res.trend}`);
    } catch (e) {
      setError(e.message);
      setStatus("Error during forecast");
    } finally { setLoading(false); }
  };

  // ── Chart data builders ─────────────────────────────────────────────────────
  const predChartData = prediction
    ? prediction.dates.map((d, i) => ({
        date: d.slice(5),
        Actual:    +prediction.actual[i].toFixed(2),
        Predicted: +prediction.predicted[i].toFixed(2),
      }))
    : [];

  const compareBarData = allResults
    ? Object.entries(allResults).map(([k, v]) => ({
        name:  MODEL_LABEL[k] ?? k,
        R2:    v.metrics.R2,
        RMSE:  v.metrics.RMSE,
        MAE:   v.metrics.MAE,
        MAPE:  v.metrics.MAPE,
        DirAcc: v.metrics.Directional_Accuracy,
      }))
    : [];

  const overlayData = allResults && (() => {
    const first = Object.values(allResults)[0];
    return first.dates.map((d, i) => {
      const row = { date: d.slice(5), Actual: +first.actual[i].toFixed(2) };
      Object.entries(allResults).forEach(([k, v]) => {
        row[MODEL_LABEL[k] ?? k] = +v.predicted[i].toFixed(2);
      });
      return row;
    });
  })();

  const fcChartData = forecastData
    ? forecastData.dates.map((d, i) => ({
        date:     d.slice(5),
        Forecast: forecastData.forecast[i],
        Upper:    forecastData.upper[i],
        Lower:    forecastData.lower[i],
      }))
    : [];

  // ── Nav ─────────────────────────────────────────────────────────────────────
  const TABS = [
    { id: "dashboard", label: "Dashboard",     icon: TrendingUp },
    { id: "compare",   label: "Model Compare", icon: BarChart2  },
    { id: "analytics", label: "Analytics",     icon: Activity   },
    { id: "forecast",  label: "Forecast",      icon: Zap        },
    { id: "about",     label: "About",         icon: Info       },
  ];

  return (
    <div className="layout">
      {/* ── Sidebar ─────────────────────────────────────────────────── */}
      <aside className="sidebar">
        <div className="sidebar-logo">
          <h1>QuantML</h1>
          <p>Stock Predictor</p>
        </div>

        {TABS.map(({ id, label, icon: Icon }) => (
          <div
            key={id}
            className={`nav-item ${tab === id ? "active" : ""}`}
            onClick={() => setTab(id)}
          >
            <Icon size={14} />
            {label}
            {tab === id && <ChevronRight size={12} style={{ marginLeft: "auto" }} />}
          </div>
        ))}

        <div style={{ flex: 1 }} />
        <div className="sidebar-section">Data Source</div>

        {/* Upload zone */}
        <div
          className={`upload-zone ${drag ? "drag" : ""}`}
          onDragOver={e => { e.preventDefault(); setDrag(true); }}
          onDragLeave={() => setDrag(false)}
          onDrop={onDrop}
          onClick={() => fileRef.current.click()}
        >
          <input ref={fileRef} type="file" accept=".csv"
                 onChange={e => handleFile(e.target.files[0])} />
          <Upload size={18} style={{ margin: "0 auto 6px", display: "block" }} />
          {file ? file.name : "Drop CSV or click"}
        </div>

        <div style={{ height: 16 }} />
        <div className="sidebar-section">Model</div>

        <div className="ctrl">
          <label>Algorithm</label>
          <select value={modelName} onChange={e => setModelName(e.target.value)}>
            {MODEL_LIST.map(m => (
              <option key={m} value={m}>{MODEL_LABEL[m]}</option>
            ))}
          </select>
        </div>

        <div style={{ height: 12 }} />
        <div className="ctrl">
          <label>Forecast days</label>
          <input type="range" min={7} max={90} step={7}
                 value={forecastDays} onChange={e => setFDays(+e.target.value)} />
          <div className="range-val">{forecastDays}d</div>
        </div>

        <div style={{ height: 12 }} />
        <div className="ctrl-stack">
          <button className="btn btn-primary" onClick={runPredict} disabled={!file || loading}>
            {loading ? <Spinner /> : <TrendingUp size={13} />}
            Run Prediction
          </button>
          <button className="btn btn-outline" onClick={runCompare} disabled={!file || loading}>
            <BarChart2 size={13} />
            Compare Models
          </button>
        </div>
        <div style={{ height: 16 }} />
      </aside>

      {/* ── Main ────────────────────────────────────────────────────── */}
      <div className="main">

        {/* Status bar */}
        <div className="status-bar">
          <div className={`status-dot ${loading ? "loading" : file ? "ready" : ""}`} />
          {status}
          {error && <span style={{ color: "var(--red)", marginLeft: 12 }}>⚠ {error}</span>}
        </div>

        <div className="tab-content">

          {/* ════════════════════════ DASHBOARD ════════════════════════ */}
          {tab === "dashboard" && (
            <>
              {prediction ? (
                <>
                  <div className="kpi-grid">
                    <KPI label="R² Score"
                         value={fmt(prediction.metrics.R2, 4)}
                         delta={prediction.metrics.R2 > 0.9 ? "Excellent" : prediction.metrics.R2 > 0.7 ? "Good" : "Fair"}
                         deltaClass={prediction.metrics.R2 > 0.9 ? "pos" : "neg"} />
                    <KPI label="RMSE"   value={fmtUSD(prediction.metrics.RMSE)} />
                    <KPI label="MAE"    value={fmtUSD(prediction.metrics.MAE)} />
                    <KPI label="MAPE"   value={`${fmt(prediction.metrics.MAPE)}%`} />
                    <KPI label="Dir. Accuracy"
                         value={`${fmt(prediction.metrics.Directional_Accuracy, 1)}%`}
                         delta={prediction.metrics.Directional_Accuracy > 55 ? "↑ above chance" : "↓ below 55%"}
                         deltaClass={prediction.metrics.Directional_Accuracy > 55 ? "pos" : "neg"} />
                  </div>

                  <Card title="Actual vs Predicted" sub={`— ${MODEL_LABEL[modelName]}`}>
                    <ResponsiveContainer width="100%" height={320}>
                      <LineChart data={predChartData} {...CHART_PROPS}>
                        <CartesianGrid stroke="#1e2a3a" strokeDasharray="3 3" />
                        <XAxis dataKey="date" tick={<MonoTick />} interval="preserveStartEnd" />
                        <YAxis tick={{ fill: "#4a5a6a", fontSize: 10, fontFamily: "JetBrains Mono" }}
                               tickFormatter={v => `$${v}`} width={64} />
                        <Tooltip {...TooltipStyle} formatter={v => [`$${v}`, ""]} />
                        <Legend wrapperStyle={{ fontFamily: "JetBrains Mono", fontSize: 11 }} />
                        <Line dataKey="Actual"    stroke="#e2e8f4" strokeWidth={1.5} dot={false} />
                        <Line dataKey="Predicted" stroke={COLOURS[MODEL_LABEL[modelName]] ?? "#00d4aa"}
                              strokeWidth={1.5} strokeDasharray="4 2" dot={false} />
                      </LineChart>
                    </ResponsiveContainer>
                  </Card>

                  <Card title="Dataset Info">
                    <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 12 }}>
                      {[
                        ["Total Rows",   prediction.dataset_info?.rows_total],
                        ["Date Start",   prediction.dataset_info?.date_start],
                        ["Date End",     prediction.dataset_info?.date_end],
                      ].map(([l, v]) => (
                        <div key={l}>
                          <div style={{ fontSize: 10, color: "var(--muted)", marginBottom: 3, textTransform: "uppercase" }}>{l}</div>
                          <div style={{ fontSize: 13 }}>{v ?? "—"}</div>
                        </div>
                      ))}
                    </div>
                  </Card>
                </>
              ) : (
                <Empty text="Upload a CSV and click Run Prediction to see results" />
              )}
            </>
          )}

          {/* ════════════════════════ MODEL COMPARE ════════════════════ */}
          {tab === "compare" && (
            <>
              {allResults ? (
                <>
                  <Card title="Model Performance" sub="— R², RMSE, MAE">
                    <ResponsiveContainer width="100%" height={280}>
                      <BarChart data={compareBarData} {...CHART_PROPS}>
                        <CartesianGrid stroke="#1e2a3a" strokeDasharray="3 3" />
                        <XAxis dataKey="name" tick={<MonoTick />} />
                        <YAxis tick={{ fill: "#4a5a6a", fontSize: 10, fontFamily: "JetBrains Mono" }} />
                        <Tooltip {...TooltipStyle} />
                        <Legend wrapperStyle={{ fontFamily: "JetBrains Mono", fontSize: 11 }} />
                        <Bar dataKey="R2"   fill="#00d4aa" radius={[3,3,0,0]} />
                        <Bar dataKey="RMSE" fill="#3b82f6" radius={[3,3,0,0]} />
                        <Bar dataKey="MAE"  fill="#f59e0b" radius={[3,3,0,0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </Card>

                  <Card title="Metrics Table">
                    <table className="data-table">
                      <thead>
                        <tr>
                          <th>Model</th><th>R²</th><th>RMSE</th><th>MAE</th>
                          <th>MAPE %</th><th>Dir. Acc %</th>
                        </tr>
                      </thead>
                      <tbody>
                        {compareBarData.map(row => (
                          <tr key={row.name}>
                            <td>
                              <span className={`badge ${modelBadgeClass(row.name)}`}>
                                {row.name}
                              </span>
                            </td>
                            <td>{fmt(row.R2, 4)}</td>
                            <td>{fmt(row.RMSE, 4)}</td>
                            <td>{fmt(row.MAE, 4)}</td>
                            <td>{fmt(row.MAPE, 2)}</td>
                            <td>{fmt(row.DirAcc, 1)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </Card>

                  {overlayData && (
                    <Card title="All Models vs Actual">
                      <ResponsiveContainer width="100%" height={340}>
                        <LineChart data={overlayData} {...CHART_PROPS}>
                          <CartesianGrid stroke="#1e2a3a" strokeDasharray="3 3" />
                          <XAxis dataKey="date" tick={<MonoTick />} interval="preserveStartEnd" />
                          <YAxis tick={{ fill: "#4a5a6a", fontSize: 10, fontFamily: "JetBrains Mono" }}
                                 tickFormatter={v => `$${v}`} width={64} />
                          <Tooltip {...TooltipStyle} formatter={v => [`$${v}`, ""]} />
                          <Legend wrapperStyle={{ fontFamily: "JetBrains Mono", fontSize: 11 }} />
                          <Line dataKey="Actual" stroke="#e2e8f4" strokeWidth={1.8} dot={false} />
                          {Object.keys(allResults).map(k => (
                            <Line key={k} dataKey={MODEL_LABEL[k] ?? k}
                                  stroke={COLOURS[MODEL_LABEL[k]] ?? "#888"}
                                  strokeWidth={1.2} strokeDasharray="4 2" dot={false} />
                          ))}
                        </LineChart>
                      </ResponsiveContainer>
                    </Card>
                  )}
                </>
              ) : (
                <Empty icon={BarChart2} text="Click Compare Models in the sidebar to run all 4 models" />
              )}
            </>
          )}

          {/* ════════════════════════ ANALYTICS ════════════════════════ */}
          {tab === "analytics" && (
            <>
              {prediction ? (
                <>
                  <Card title="Prediction Distribution">
                    <ResponsiveContainer width="100%" height={300}>
                      <ComposedChart data={predChartData} {...CHART_PROPS}>
                        <CartesianGrid stroke="#1e2a3a" strokeDasharray="3 3" />
                        <XAxis dataKey="date" tick={<MonoTick />} interval="preserveStartEnd" />
                        <YAxis tick={{ fill: "#4a5a6a", fontSize: 10, fontFamily: "JetBrains Mono" }}
                               tickFormatter={v => `$${v}`} width={64} />
                        <Tooltip {...TooltipStyle} formatter={v => [`$${v}`, ""]} />
                        <Legend wrapperStyle={{ fontFamily: "JetBrains Mono", fontSize: 11 }} />
                        <Area dataKey="Actual" fill="rgba(226,232,244,0.05)"
                              stroke="#e2e8f4" strokeWidth={1} dot={false} />
                        <Line dataKey="Predicted"
                              stroke={COLOURS[MODEL_LABEL[modelName]] ?? "#00d4aa"}
                              strokeWidth={1.5} strokeDasharray="3 2" dot={false} />
                      </ComposedChart>
                    </ResponsiveContainer>
                  </Card>

                  <Card title="Residuals (Actual − Predicted)">
                    <ResponsiveContainer width="100%" height={220}>
                      <BarChart
                        data={predChartData.map(r => ({
                          date:     r.date,
                          residual: +(r.Actual - r.Predicted).toFixed(2),
                        }))}
                        {...CHART_PROPS}
                      >
                        <CartesianGrid stroke="#1e2a3a" strokeDasharray="3 3" />
                        <XAxis dataKey="date" tick={<MonoTick />} interval="preserveStartEnd" />
                        <YAxis tick={{ fill: "#4a5a6a", fontSize: 10, fontFamily: "JetBrains Mono" }}
                               tickFormatter={v => `$${v}`} width={60} />
                        <Tooltip {...TooltipStyle} formatter={v => [`$${v}`, "Residual"]} />
                        <ReferenceLine y={0} stroke="#4a5a6a" strokeDasharray="3 3" />
                        <Bar dataKey="residual"
                             fill="#00d4aa"
                             radius={[2,2,0,0]}
                             label={false}
                        />
                      </BarChart>
                    </ResponsiveContainer>
                  </Card>

                  <Card title="Directional Accuracy Breakdown">
                    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 16 }}>
                      {(() => {
                        const actual = prediction.actual;
                        const pred   = prediction.predicted;
                        let correct = 0, incorrect = 0;
                        for (let i = 1; i < actual.length; i++) {
                          const aUp = actual[i] > actual[i-1];
                          const pUp = pred[i]   > pred[i-1];
                          if (aUp === pUp) correct++; else incorrect++;
                        }
                        const total = correct + incorrect;
                        return [
                          ["Correct Direction",   correct,             "var(--green)"],
                          ["Wrong Direction",     incorrect,           "var(--red)"],
                          ["Total Days Compared", total,               "var(--teal)"],
                        ].map(([l, v, c]) => (
                          <div key={l} className="fc-metric">
                            <div className="fc-metric-label">{l}</div>
                            <div className="fc-metric-value" style={{ color: c }}>{v}</div>
                          </div>
                        ));
                      })()}
                    </div>
                  </Card>
                </>
              ) : (
                <Empty icon={Activity} text="Run a prediction first to see analytics" />
              )}
            </>
          )}

          {/* ════════════════════════ FORECAST ═════════════════════════ */}
          {tab === "forecast" && (
            <>
              <div style={{ marginBottom: 16 }}>
                <button className="btn btn-primary"
                        style={{ width: "auto", padding: "9px 20px" }}
                        onClick={runForecast} disabled={!file || loading}>
                  {loading ? <Spinner /> : <Zap size={13} />}
                  Generate {forecastDays}-Day Forecast
                </button>
              </div>

              {forecastData ? (
                <div style={{ display: "grid", gridTemplateColumns: "1fr 280px", gap: 20 }}>
                  <div>
                    <Card title={`${forecastDays}-Day Price Forecast`}
                          sub={`— ${forecastData.trend} (${forecastData.price_change_pct > 0 ? "+" : ""}${forecastData.price_change_pct}%)`}>
                      <ResponsiveContainer width="100%" height={340}>
                        <ComposedChart data={fcChartData} {...CHART_PROPS}>
                          <CartesianGrid stroke="#1e2a3a" strokeDasharray="3 3" />
                          <XAxis dataKey="date" tick={<MonoTick />} interval="preserveStartEnd" />
                          <YAxis tick={{ fill: "#4a5a6a", fontSize: 10, fontFamily: "JetBrains Mono" }}
                                 tickFormatter={v => `$${v}`} width={64} />
                          <Tooltip {...TooltipStyle} formatter={v => [`$${v}`, ""]} />
                          <Legend wrapperStyle={{ fontFamily: "JetBrains Mono", fontSize: 11 }} />
                          <Area dataKey="Upper" fill="rgba(0,212,170,0.08)"
                                stroke="rgba(0,212,170,0.3)" strokeWidth={1} dot={false} name="Upper Band" />
                          <Area dataKey="Lower" fill="rgba(0,212,170,0.04)"
                                stroke="rgba(0,212,170,0.2)" strokeWidth={1} dot={false} name="Lower Band" />
                          <Line dataKey="Forecast" stroke="#00d4aa" strokeWidth={2}
                                strokeDasharray="6 3" dot={false} />
                        </ComposedChart>
                      </ResponsiveContainer>
                    </Card>

                    <Card title="First 7 Days">
                      <table className="data-table">
                        <thead>
                          <tr><th>Date</th><th>Forecast</th><th>Low (95%)</th><th>High (95%)</th></tr>
                        </thead>
                        <tbody>
                          {forecastData.dates.slice(0, 7).map((d, i) => (
                            <tr key={d}>
                              <td>{d}</td>
                              <td style={{ color: "var(--teal)" }}>{fmtUSD(forecastData.forecast[i])}</td>
                              <td style={{ color: "var(--red)" }}>{fmtUSD(forecastData.lower[i])}</td>
                              <td style={{ color: "var(--green)" }}>{fmtUSD(forecastData.upper[i])}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </Card>
                  </div>

                  <div className="fc-metrics">
                    <div className="fc-metric">
                      <div className="fc-metric-label">Starting Price</div>
                      <div className="fc-metric-value">{fmtUSD(forecastData.last_known_price)}</div>
                    </div>
                    <div className="fc-metric">
                      <div className="fc-metric-label">Day-{forecastDays} Target</div>
                      <div className="fc-metric-value" style={{ color: forecastData.price_change_pct > 0 ? "var(--green)" : "var(--red)" }}>
                        {fmtUSD(forecastData.forecast[forecastData.forecast.length - 1])}
                      </div>
                    </div>
                    <div className="fc-metric">
                      <div className="fc-metric-label">Expected Change</div>
                      <div className="fc-metric-value" style={{ color: forecastData.price_change_pct > 0 ? "var(--green)" : "var(--red)" }}>
                        {forecastData.price_change_pct > 0 ? "+" : ""}{forecastData.price_change_pct}%
                      </div>
                    </div>
                    <div className="fc-metric">
                      <div className="fc-metric-label">Trend</div>
                      <div className="fc-metric-value" style={{ color: forecastData.trend === "Bullish" ? "var(--green)" : "var(--red)", display: "flex", alignItems: "center", gap: 8 }}>
                        {forecastData.trend === "Bullish" ? <TrendingUp size={18} /> : <TrendingDown size={18} />}
                        {forecastData.trend}
                      </div>
                    </div>
                    <div className="fc-metric">
                      <div className="fc-metric-label">Model Used</div>
                      <div style={{ marginTop: 4 }}>
                        <span className={`badge ${modelBadgeClass(MODEL_LABEL[modelName])}`}>
                          {MODEL_LABEL[modelName]}
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              ) : (
                <Empty icon={Zap} text="Select a model and click Generate Forecast" />
              )}
            </>
          )}

          {/* ════════════════════════ ABOUT ════════════════════════════ */}
          {tab === "about" && (
            <>
              <div style={{ marginBottom: 24 }}>
                <h2 style={{ fontFamily: "var(--font-display)", fontSize: 28, fontWeight: 800, color: "var(--teal)" }}>
                  QuantML
                </h2>
                <p style={{ color: "var(--muted)", marginTop: 4, fontSize: 12 }}>
                  ML-powered stock price prediction · React + FastAPI
                </p>
              </div>

              <div className="about-grid">
                <div className="about-section">
                  <h3>Models</h3>
                  <ul>
                    <li>XGBoost — gradient boosted trees, best accuracy</li>
                    <li>Random Forest — robust ensemble, handles noise</li>
                    <li>Decision Tree — interpretable, fast baseline</li>
                    <li>Linear Regression — transparent benchmark</li>
                  </ul>
                </div>

                <div className="about-section">
                  <h3>Features (28 total)</h3>
                  <ul>
                    <li>Lag prices: close_lag1/5/10/20</li>
                    <li>Moving averages: MA-20/50/200, EMA-12/26</li>
                    <li>MACD, RSI-14, Bollinger Bands</li>
                    <li>OBV, Volume ratio, Daily return</li>
                  </ul>
                </div>

                <div className="about-section">
                  <h3>API Endpoints</h3>
                  <ul>
                    <li>POST /upload-and-predict</li>
                    <li>POST /compare-all</li>
                    <li>POST /forecast</li>
                    <li>GET  /metrics</li>
                    <li>GET  /health</li>
                  </ul>
                </div>

                <div className="about-section">
                  <h3>Deploy to Vercel</h3>
                  <div className="code-block">{`npm create vite@latest quantml -- --template react
cd quantml
npm install recharts lucide-react
# replace src/App.jsx with this file
vercel deploy`}</div>
                </div>

                <div className="about-section" style={{ gridColumn: "1 / -1" }}>
                  <h3>Environment</h3>
                  <div className="code-block">{`# .env.local (React / Vercel)
VITE_API_URL=https://your-fastapi-backend.com

# Vercel dashboard → Settings → Environment Variables
VITE_API_URL = https://your-api.railway.app`}</div>
                </div>
              </div>
            </>
          )}

        </div>
      </div>
    </div>
  );
}