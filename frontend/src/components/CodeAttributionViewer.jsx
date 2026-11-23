import React, { useState } from "react";

export default function CodeAttributionViewer({ explanation, targetClass, confidence }) {
  const [filterMode, setFilterMode] = useState("all"); // "all" | "significant" | "positive" | "negative"

  if (!explanation) return null;

  if (explanation.error) {
    return (
      <div className="glass glass--card attribution-error">
        <div className="glass__title">Source-Level XAI Attribution</div>
        <p className="error-text">⚠️ {explanation.error}</p>
      </div>
    );
  }

  const lines = explanation.lines || [];
  const summary = explanation.summary || {};
  const explainedClass = explanation.target_class || targetClass || "Unknown";
  const explainedConf = explanation.baseline_confidence != null 
    ? `${(explanation.baseline_confidence * 100).toFixed(1)}%` 
    : confidence != null ? `${(confidence * 100).toFixed(1)}%` : "--";

  const filteredLines = lines.filter((line) => {
    if (filterMode === "significant") return line.is_significant;
    if (filterMode === "positive") return line.direction === "positive";
    if (filterMode === "negative") return line.direction === "negative";
    return true;
  });

  const maxAbsAttr = Math.max(...lines.map((l) => Math.abs(l.attribution || 0)), 0.01);

  return (
    <div className="glass glass--card attribution-container">
      {/* ── Header ── */}
      <div className="attribution-header">
        <div>
          <div className="glass__title">Source-Code Attribution (XAI)</div>
          <div className="attribution-subtitle">
            Explaining 4-Class Prediction: <strong className="highlight-text">{explainedClass}</strong> · Confidence: <strong>{explainedConf}</strong>
          </div>
        </div>

        <div className="attribution-filters">
          <button
            className={`filter-btn ${filterMode === "all" ? "filter-btn--active" : ""}`}
            onClick={() => setFilterMode("all")}
          >
            All Lines ({lines.length})
          </button>
          <button
            className={`filter-btn ${filterMode === "significant" ? "filter-btn--active" : ""}`}
            onClick={() => setFilterMode("significant")}
          >
            Significant Only ({lines.filter((l) => l.is_significant).length})
          </button>
          <button
            className={`filter-btn filter-btn--positive ${filterMode === "positive" ? "filter-btn--active" : ""}`}
            onClick={() => setFilterMode("positive")}
          >
            ↑ Toward ({lines.filter((l) => l.direction === "positive").length})
          </button>
          <button
            className={`filter-btn filter-btn--negative ${filterMode === "negative" ? "filter-btn--active" : ""}`}
            onClick={() => setFilterMode("negative")}
          >
            ↓ Against ({lines.filter((l) => l.direction === "negative").length})
          </button>
        </div>
      </div>

      {/* ── Legend Banner ── */}
      <div className="attribution-legend">
        <div className="legend-item">
          <span className="legend-badge legend-badge--positive">↑ +Score</span>
          <span className="legend-desc">
            <strong>Contributes toward {explainedClass}</strong> (masking this line causes probability to drop)
          </span>
        </div>
        <div className="legend-item">
          <span className="legend-badge legend-badge--negative">↓ -Score</span>
          <span className="legend-desc">
            <strong>Contributes against {explainedClass}</strong> (masking this line causes probability to rise)
          </span>
        </div>
      </div>

      {/* ── Key Highlights Summary ── */}
      {(summary.top_positive_snippet || summary.top_negative_snippet) && (
        <div className="attribution-key-insights">
          {summary.top_positive_snippet && (
            <div className="insight-chip insight-chip--positive">
              <span className="insight-chip__label">Most decisive line for {explainedClass}:</span>
              <code className="insight-chip__code">
                L#{summary.positive_contributing_lines?.[0]}: "{summary.top_positive_snippet}" (+{summary.max_positive_attribution})
              </code>
            </div>
          )}
          {summary.top_negative_snippet && (
            <div className="insight-chip insight-chip--negative">
              <span className="insight-chip__label">Strongest counter-signal:</span>
              <code className="insight-chip__code">
                L#{summary.negative_contributing_lines?.[0]}: "{summary.top_negative_snippet}" ({summary.max_negative_attribution})
              </code>
            </div>
          )}
        </div>
      )}

      {/* ── Code Viewer ── */}
      <div className="code-viewer">
        <div className="code-viewer__scroll">
          {filteredLines.map((line) => {
            const attr = line.attribution || 0;
            const absAttr = Math.abs(attr);
            const intensity = Math.min(absAttr / maxAbsAttr, 1.0);
            
            let rowClass = "code-line";
            let bgStyle = {};

            if (line.is_significant) {
              if (line.direction === "positive") {
                rowClass += " code-line--positive";
                bgStyle = { backgroundColor: `rgba(16, 185, 129, ${0.08 + intensity * 0.22})` };
              } else if (line.direction === "negative") {
                rowClass += " code-line--negative";
                bgStyle = { backgroundColor: `rgba(239, 68, 68, ${0.08 + intensity * 0.22})` };
              }
            }

            return (
              <div key={line.line_number} className={rowClass} style={bgStyle}>
                <span className="code-line__num">{line.line_number}</span>
                <span className="code-line__content">
                  <code>{line.content || " "}</code>
                </span>
                <div className="code-line__impact">
                  {line.is_significant && (
                    <span className={`impact-badge impact-badge--${line.direction}`}>
                      {line.direction === "positive" ? "↑ +" : "↓ "}
                      {attr.toFixed(3)}
                    </span>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
