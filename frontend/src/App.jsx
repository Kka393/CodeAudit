import React, { useState, useEffect, useRef, useCallback } from "react";
import Navbar from "./components/Navbar.jsx";
import CodeAttributionViewer from "./components/CodeAttributionViewer.jsx";
import RealtimeProgress from "./components/RealtimeProgress.jsx";
import CouncilDebate from "./components/CouncilDebate.jsx";
import ArchitectureView from "./components/ArchitectureView.jsx";
import AboutView from "./components/AboutView.jsx";
import { checkHealth, predict, startDebateStream, explainPrediction } from "./api.js";

// ── Language Extension Map ──────────────────────────────────────────
const LANG_FROM_EXT = {
  py: "python", java: "java", js: "javascript", jsx: "javascript",
  ts: "javascript", tsx: "javascript", cpp: "cpp", cc: "cpp",
  cxx: "cpp", c: "c", h: "c", go: "go", cs: "csharp", php: "php",
};

const SUPPORTED_LANGUAGES = [
  "python", "java", "javascript", "cpp", "c", "go", "csharp", "php",
];

// Helper to normalize verdict classes to standard css class modifier
function classForVerdict(cls) {
  const l = (cls || "").toLowerCase();
  if (l === "machine" || l === "ai") return "machine";
  if (l === "human") return "human";
  if (l === "hybrid") return "hybrid";
  if (l === "adversarial") return "adversarial";
  return "human";
}

function pct(v) {
  return `${((v || 0) * 100).toFixed(1)}%`;
}

// ── SVG Confidence Gauge ────────────────────────────────────────────
function ConfidenceGauge({ value }) {
  const r = 54, cx = 64, cy = 64;
  const circ = 2 * Math.PI * r;
  const pctVal = Math.max(0, Math.min(1, value || 0));
  const offset = circ * (1 - pctVal * 0.75); // 270-degree arc
  const color = pctVal >= 0.7 ? "var(--success)" : pctVal >= 0.5 ? "var(--warning)" : "var(--danger)";

  return (
    <div className="gauge-box">
      <svg width="132" height="102" viewBox="0 0 128 100">
        <path
          d={`M ${cx - r} ${cy} A ${r} ${r} 0 1 1 ${cx + r} ${cy}`}
          fill="none" stroke="rgba(255,255,255,0.08)" strokeWidth="8"
          strokeLinecap="round"
        />
        <path
          d={`M ${cx - r} ${cy} A ${r} ${r} 0 1 1 ${cx + r} ${cy}`}
          fill="none" stroke={color} strokeWidth="8" strokeLinecap="round"
          strokeDasharray={circ} strokeDashoffset={offset}
          style={{ transition: "stroke-dashoffset 0.8s ease-out" }}
        />
        <text x={cx} y={cy - 5} textAnchor="middle" fill="var(--text-primary)"
          fontFamily="var(--font-mono)" fontSize="18" fontWeight="700">
          {pct(pctVal)}
        </text>
        <text x={cx} y={cy + 14} textAnchor="middle" fill="var(--text-muted)" fontSize="10">
          confidence
        </text>
      </svg>
    </div>
  );
}

// ── Probability Bars ────────────────────────────────────────────────
function ProbabilityBars({ probabilities, highlightClass }) {
  if (!probabilities) return null;
  const entries = Object.entries(probabilities).sort((a, b) => b[1] - a[1]);
  const maxVal = Math.max(...entries.map(([, v]) => v), 0.01);

  return (
    <div className="prob-list">
      {entries.map(([cls, prob]) => {
        const isHighlight = cls.toLowerCase() === (highlightClass || "").toLowerCase();
        return (
          <div className="prob-row" key={cls}>
            <span className="prob-label">{cls === "AI" ? "Machine" : cls}</span>
            <div className="prob-track">
              <div
                className={`prob-fill ${isHighlight ? "prob-fill--highlight" : "prob-fill--normal"}`}
                style={{ width: `${(prob / maxVal) * 100}%` }}
              />
            </div>
            <span className={`prob-val ${isHighlight ? "prob-val--highlight" : ""}`}>
              {pct(prob)}
            </span>
          </div>
        );
      })}
    </div>
  );
}

// ── Agent Card Component ────────────────────────────────────────────
function AgentCard({ title, agent }) {
  if (!agent) return null;
  const v = classForVerdict(agent.predicted_class);
  const displayClass = agent.predicted_class === "AI" ? "Machine" : agent.predicted_class;

  return (
    <div className="glass glass--card">
      <div className="glass__title">{title}</div>
      {agent.failed && (
        <div className="gate-callout gate-callout--fail mb-sm">
          <span>Agent failed evidence validation — fallback neutral score used.</span>
        </div>
      )}
      <div className={`agent-card__verdict verdict-card__badge--${v}`}>
        {displayClass} · {pct(agent.confidence)}
      </div>
      {agent.rationale && (
        <p className="agent-card__rationale">{agent.rationale}</p>
      )}
      {agent.evidence && agent.evidence.length > 0 && (
        <ul className="evidence-list">
          {agent.evidence.map((ev, i) => (
            <li key={i} className="evidence-item">
              <em>"{ev.quote}"</em> — {ev.reason}
            </li>
          ))}
        </ul>
      )}
      <div className="agent-meta">
        Forensic Steps: {agent.num_steps || 0} tool call{agent.num_steps !== 1 ? "s" : ""}
        {agent.tool_calls_made?.length > 0 && (
          <> · Tools: {agent.tool_calls_made.join(", ")}</>
        )}
      </div>
    </div>
  );
}

// =====================================================================
// Main Application Component
// =====================================================================
export default function App() {
  // ── Persistent Navigation State ─────────────────────────────────
  const [activeTab, setActiveTab] = useState("home");
  const [serverOnline, setServerOnline] = useState(true);

  // ── Persistent Input State ──────────────────────────────────────
  const [code, setCode] = useState("");
  const [language, setLanguage] = useState("python");
  const [threshold, setThreshold] = useState(0.70);

  // ── Persistent Analysis & XAI State ─────────────────────────────
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isExplaining, setIsExplaining] = useState(false);
  const [error, setError] = useState(null);
  const [predictionResult, setPredictionResult] = useState(null);
  const [explanationResult, setExplanationResult] = useState(null);

  // ── Persistent LLM Council Debate State ─────────────────────────
  const [debateEvents, setDebateEvents] = useState([]);
  const [debating, setDebating] = useState(false);
  const [debateDone, setDebateDone] = useState(false);
  const debateCancelRef = useRef(null);

  // ── Server Health Ping ──────────────────────────────────────────
  useEffect(() => {
    let mounted = true;
    checkHealth().then((ok) => {
      if (mounted) setServerOnline(ok);
    });
    const interval = setInterval(() => {
      checkHealth().then((ok) => {
        if (mounted) setServerOnline(ok);
      });
    }, 15000);
    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, []);

  // ── File Upload Handler ─────────────────────────────────────────
  const handleFileUpload = useCallback((e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const ext = file.name.split(".").pop()?.toLowerCase();
    const detectedLang = LANG_FROM_EXT[ext];
    if (detectedLang) setLanguage(detectedLang);

    const reader = new FileReader();
    reader.onload = (ev) => setCode(ev.target.result || "");
    reader.readAsText(file);
  }, []);

  // ── Analyze Code Action ─────────────────────────────────────────
  const handleAnalyze = useCallback(async () => {
    if (!code.trim()) return;

    setIsAnalyzing(true);
    setError(null);
    setPredictionResult(null);
    setExplanationResult(null);
    setDebateEvents([]);
    setDebateDone(false);

    try {
      // 1. Run 4-Class Classification & Agent Forensics Pipeline
      const data = await predict(code, language, threshold);
      setPredictionResult(data);
      setIsAnalyzing(false);

      // 2. Automatically generate source-level XAI attribution for 4-Class Prediction
      setIsExplaining(true);
      try {
        const explData = await explainPrediction(code, language, data.prediction);
        setExplanationResult(explData);
      } catch (explErr) {
        console.warn("Attribution explanation failed:", explErr);
        setExplanationResult({ error: explErr.message });
      } finally {
        setIsExplaining(false);
      }
    } catch (err) {
      setError(err.message || "Failed to analyze code.");
      setIsAnalyzing(false);
      setIsExplaining(false);
    }
  }, [code, language, threshold]);

  // ── Start Debate Action ─────────────────────────────────────────
  const handleStartDebate = useCallback(() => {
    if (!code.trim()) return;

    // Transition to council view
    setActiveTab("council");
    setDebating(true);
    setDebateEvents([]);
    setDebateDone(false);

    // Target probability for debate
    const probs = predictionResult?.ml_probabilities || {};
    const aiProb = probs["Machine"] || probs["AI"] || predictionResult?.confidence || 0.5;

    const cancel = startDebateStream(
      code,
      aiProb,
      3,
      (event) => {
        setDebateEvents((prev) => [...prev, event]);
        if (event.event_type === "done" || event.event_type === "error") {
          setDebating(false);
          setDebateDone(true);
        }
      },
      (err) => {
        setDebating(false);
        setDebateEvents((prev) => [
          ...prev,
          { event_type: "error", content: err.message },
        ]);
      }
    );

    debateCancelRef.current = cancel;
  }, [code, predictionResult]);

  const handleStopDebate = useCallback(() => {
    if (debateCancelRef.current) {
      debateCancelRef.current();
      debateCancelRef.current = null;
    }
    setDebating(false);
  }, []);

  // ── Keyboard shortcut (Ctrl+Enter) ──────────────────────────────
  const handleKeyDown = useCallback(
    (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
        e.preventDefault();
        handleAnalyze();
      }
    },
    [handleAnalyze]
  );

  // Normalize final predicted class label (Strictly "Machine")
  const displayPrediction = predictionResult?.prediction === "AI" ? "Machine" : predictionResult?.prediction;
  const isMachinePredicted = displayPrediction?.toLowerCase() === "machine";

  return (
    <div className="app">
      {/* ── Persistent Navigation Bar ── */}
      <Navbar
        activeTab={activeTab}
        onSelectTab={setActiveTab}
        serverOnline={serverOnline}
      />

      {/* ── Page 1: Home (Code Audit & Detection) ── */}
      {activeTab === "home" && (
        <main>
          {/* Hero Header */}
          <section className="hero">
            <div className="hero__badge">Research Framework</div>
            <h1 className="hero__title">CODE AUDIT</h1>
            <p className="hero__tagline">
              Agentic framework for identifying AI generated code using Model Council and XAI
            </p>
          </section>

          {/* Input Section */}
          <section className="glass glass--card input-section" aria-label="Source code input">
            <div className="input-row">
              {/* Language Selector */}
              <div className="input-group">
                <label className="input-label" htmlFor="lang-select">Language</label>
                <select
                  id="lang-select"
                  className="select"
                  value={language}
                  onChange={(e) => setLanguage(e.target.value)}
                >
                  {SUPPORTED_LANGUAGES.map((l) => (
                    <option key={l} value={l}>{l}</option>
                  ))}
                </select>
              </div>

              {/* File Upload */}
              <div className="input-group file-input-wrapper">
                <span className="input-label">File Upload</span>
                <label className="file-input-label" htmlFor="file-input">
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                    <polyline points="17 8 12 3 7 8"/>
                    <line x1="12" y1="3" x2="12" y2="15"/>
                  </svg>
                  Browse Code File
                </label>
                <input
                  id="file-input"
                  className="file-input"
                  type="file"
                  accept=".py,.java,.js,.jsx,.ts,.tsx,.cpp,.cc,.c,.h,.go,.cs,.php"
                  onChange={handleFileUpload}
                />
              </div>

              {/* Confidence Threshold Slider */}
              <div className="input-group" style={{ flex: 1, minWidth: "220px" }}>
                <div style={{ display: "flex", justifyContent: "space-between" }}>
                  <label className="input-label" htmlFor="threshold-range">Confidence Gate</label>
                  <span className="threshold-badge">{pct(threshold)}</span>
                </div>
                <div className="threshold-slider-box">
                  <input
                    id="threshold-range"
                    className="range-slider"
                    type="range"
                    min="0.30"
                    max="0.95"
                    step="0.05"
                    value={threshold}
                    onChange={(e) => setThreshold(parseFloat(e.target.value))}
                    style={{ width: "100%" }}
                  />
                </div>
              </div>
            </div>

            {/* Code Textarea */}
            <div className="textarea-container">
              <textarea
                id="code-editor"
                className="textarea"
                value={code}
                onChange={(e) => setCode(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Paste code snippet here or upload a file above...&#10;&#10;Press Ctrl + Enter to analyze"
                spellCheck={false}
              />
            </div>

            {/* Input Action Controls */}
            <div className="input-actions">
              <span className="hint-text">Supported: Python, Java, JS, C++, Go, C#, C, PHP</span>
              <button
                id="analyze-btn"
                className={`btn btn--primary ${isAnalyzing ? "btn--loading" : ""}`}
                onClick={handleAnalyze}
                disabled={isAnalyzing || isExplaining || !code.trim()}
              >
                {isAnalyzing ? (
                  <><span className="spinner" /> Processing Pipeline...</>
                ) : (
                  "Analyze Code"
                )}
              </button>
            </div>
          </section>

          {/* Real-Time Processing Multi-Stage Indicator */}
          <RealtimeProgress isAnalyzing={isAnalyzing} isExplaining={isExplaining} />

          {/* Error Message */}
          {error && (
            <div className="error-message" role="alert">
              <strong>Error:</strong> {error}
            </div>
          )}

          {/* Results View */}
          {predictionResult && !isAnalyzing && (
            <div className="results-grid" aria-live="polite">
              {/* 1. Final Aggregated Verdict Card */}
              <section className="glass glass--card verdict-card">
                <span className="verdict-card__label">Final Aggregated Prediction</span>
                <div className={`verdict-card__badge verdict-card__badge--${classForVerdict(displayPrediction)}`}>
                  {displayPrediction}
                </div>
                <ConfidenceGauge value={predictionResult.confidence} />
                <div className="verdict-card__conf-text">
                  Consensus Confidence: <strong>{pct(predictionResult.confidence)}</strong>
                </div>
              </section>

              {/* 2. 4-Class ML Model Output */}
              <section className="glass glass--card">
                <div className="glass__title">Primary 4-Class Classifier Output</div>
                <div className="agent-card__verdict verdict-card__badge--machine mb-sm" style={{ fontSize: "1rem" }}>
                  Top ML Prediction: <strong>{predictionResult.ml_prediction === "AI" ? "Machine" : predictionResult.ml_prediction}</strong>
                </div>
                <ProbabilityBars
                  probabilities={predictionResult.ml_probabilities}
                  highlightClass={predictionResult.ml_prediction}
                />
              </section>

              {/* 3. Source-Code Level XAI Attribution (4-Class Classifier) */}
              <section>
                <CodeAttributionViewer
                  explanation={explanationResult}
                  targetClass={displayPrediction}
                  confidence={predictionResult.confidence}
                />
              </section>

              {/* 4. Forensic Agent Outputs (Agent A & Agent B) */}
              <section className="agents-grid">
                <AgentCard
                  title="Agent A · Comment & Style Forensics"
                  agent={predictionResult.comment_style_agent}
                />
                <AgentCard
                  title="Agent B · Structural & AST Forensics"
                  agent={predictionResult.structural_agent}
                />
              </section>

              {/* 5. Aggregation & Confidence Gate Status */}
              <section className="glass glass--card">
                <div className="glass__title">Aggregation Consensus & Confidence Gate</div>

                <div className="agg-weights">
                  <div className="agg-weight-box">
                    <span className="agg-weight__val">{(predictionResult.aggregation.ml_weight * 100).toFixed(0)}%</span>
                    <span className="agg-weight__lbl">4-Class ML Model</span>
                  </div>
                  <div className="agg-weight-box">
                    <span className="agg-weight__val">{(predictionResult.aggregation.agent_a_weight * 100).toFixed(0)}%</span>
                    <span className="agg-weight__lbl">Comment Agent</span>
                  </div>
                  <div className="agg-weight-box">
                    <span className="agg-weight__val">{(predictionResult.aggregation.agent_b_weight * 100).toFixed(0)}%</span>
                    <span className="agg-weight__lbl">Structural Agent</span>
                  </div>
                </div>

                <div>
                  <span className={`agreement-pill ${predictionResult.aggregation.agreement ? "agreement-pill--yes" : "agreement-pill--no"}`}>
                    {predictionResult.aggregation.agreement ? "✓ Sources Agree" : "✗ Evidentiary Disagreement"}
                  </span>
                </div>

                <div className={`gate-callout ${predictionResult.confidence_gate.passed ? "gate-callout--pass" : "gate-callout--fail"}`}>
                  <div>
                    {predictionResult.confidence_gate.passed
                      ? `Confidence gate passed (threshold: ${pct(predictionResult.confidence_gate.threshold)}) — direct verdict accepted.`
                      : `Confidence gate flagged uncertainty (threshold: ${pct(predictionResult.confidence_gate.threshold)}) — LLM Model Council recommended.`}
                  </div>

                  {predictionResult.confidence_gate.needs_debate && (
                    <button
                      className="btn btn--primary"
                      onClick={handleStartDebate}
                      style={{ padding: "6px 14px", fontSize: "0.85rem" }}
                    >
                      Run Debate in Council →
                    </button>
                  )}
                </div>
              </section>

              {/* 6. Classifier 2: LLM-Family Classifier (ONLY WHEN MACHINE IS PREDICTED) */}
              {isMachinePredicted && predictionResult.family_prediction && (
                <section className="glass glass--card family-card">
                  <div className="glass__title">Classifier 2 · LLM Model Family Identification</div>
                  <p className="caption-text mb-sm" style={{ color: "var(--text-secondary)" }}>
                    Activated exclusively when the 4-class classifier predicts Machine. Identifies the most probable LLM family responsible.
                  </p>

                  <div className={`family-badge ${predictionResult.family_prediction.is_unknown ? "family-badge--unknown" : ""}`}>
                    {predictionResult.family_prediction.is_unknown
                      ? "⚠️ Unknown Family"
                      : predictionResult.family_prediction.predicted_family}
                  </div>

                  <ProbabilityBars
                    probabilities={predictionResult.family_prediction.probabilities}
                    highlightClass={predictionResult.family_prediction.predicted_family}
                  />
                </section>
              )}
            </div>
          )}
        </main>
      )}

      {/* ── Page 2: LLM Council (Multi-Agent Debate) ── */}
      {activeTab === "council" && (
        <CouncilDebate
          code={code}
          predictionResult={predictionResult}
          debateEvents={debateEvents}
          debating={debating}
          debateDone={debateDone}
          onStartDebate={handleStartDebate}
          onStopDebate={handleStopDebate}
        />
      )}

      {/* ── Page 3: Architecture Documentation ── */}
      {activeTab === "architecture" && (
        <ArchitectureView />
      )}

      {/* ── Page 4: About (Team G1272) ── */}
      {activeTab === "about" && (
        <AboutView />
      )}
    </div>
  );
}
