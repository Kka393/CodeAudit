import React, { useEffect, useState } from "react";

export default function RealtimeProgress({ isAnalyzing, isExplaining }) {
  const [currentStage, setCurrentStage] = useState(0);

  const stages = [
    { id: "clean", label: "Preprocessing & View Splitting", detail: "Normalizing code and extracting AST views (main, sub-functions, comments)" },
    { id: "embed", label: "ModernBERT Encoding", detail: "Generating 768-d masked-mean pooled embeddings across the 3 views" },
    { id: "classify", label: "4-Class Classification", detail: "Projecting views (768→256), concatenating, and running MLP head" },
    { id: "agent_a", label: "Agent A: Comment & Style Forensics", detail: "Analyzing comment density, docstrings, and identifier naming style" },
    { id: "agent_b", label: "Agent B: Structural Forensics", detail: "Analyzing cyclomatic complexity, nesting depth, and control flow" },
    { id: "aggregate", label: "Aggregator & Confidence Gate", detail: "Evaluating 50% ML + 25% Agent A + 25% Agent B consensus" },
  ];

  useEffect(() => {
    if (!isAnalyzing && !isExplaining) {
      setCurrentStage(0);
      return;
    }

    const interval = setInterval(() => {
      setCurrentStage((prev) => {
        if (prev < stages.length - 1) {
          return prev + 1;
        }
        return prev;
      });
    }, 900);

    return () => clearInterval(interval);
  }, [isAnalyzing, isExplaining]);

  if (!isAnalyzing && !isExplaining) return null;

  return (
    <div className="glass glass--card progress-card" aria-live="polite">
      <div className="progress-header">
        <div className="spinner" />
        <div>
          <div className="glass__title">
            {isExplaining ? "Computing Source-Level XAI Attribution..." : "Running Multi-View Analysis..."}
          </div>
          <div className="progress-current-text">
            {stages[currentStage]?.label}
          </div>
        </div>
      </div>

      <div className="progress-track">
        <div
          className="progress-fill"
          style={{ width: `${((currentStage + 1) / stages.length) * 100}%` }}
        />
      </div>

      <div className="progress-steps-list">
        {stages.map((st, i) => {
          const isDone = i < currentStage;
          const isActive = i === currentStage;
          return (
            <div
              key={st.id}
              className={`progress-step ${isDone ? "progress-step--done" : ""} ${isActive ? "progress-step--active" : ""}`}
            >
              <div className="progress-step__dot">
                {isDone ? "✓" : i + 1}
              </div>
              <div className="progress-step__info">
                <span className="progress-step__title">{st.label}</span>
                {isActive && <span className="progress-step__desc">{st.detail}</span>}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
