import React, { useMemo } from "react";

export default function CouncilDebate({
  code,
  predictionResult,
  debateEvents,
  debating,
  debateDone,
  onStartDebate,
  onStopDebate,
}) {
  // Parse rounds from streaming events
  const { rounds, judgeVerdict, isConverged, errorEvent } = useMemo(() => {
    const rList = [];
    let currentR = null;
    let verdict = null;
    let converged = false;
    let err = null;

    debateEvents.forEach((ev) => {
      if (ev.event_type === "round_start") {
        if (ev.converged) {
          converged = true;
        } else {
          currentR = { roundNum: ev.round_num ?? rList.length, entries: [] };
          rList.push(currentR);
        }
      } else if (ev.event_type === "advocate" || ev.event_type === "skeptic") {
        if (!currentR) {
          currentR = { roundNum: ev.round_num ?? 0, entries: [] };
          rList.push(currentR);
        }
        currentR.entries.push(ev);
      } else if (ev.event_type === "judge") {
        try {
          verdict = JSON.parse(ev.content);
        } catch {
          verdict = { verdict: "Unknown", reasoning: ev.content, confidence: ev.confidence || 50 };
        }
      } else if (ev.event_type === "error") {
        err = ev.content;
      }
    });

    return { rounds: rList, judgeVerdict: verdict, isConverged: converged, errorEvent: err };
  }, [debateEvents]);

  // Parse structured arguments from model text
  function parseArgumentText(text) {
    if (!text) return { evidence: "", argument: text || "" };

    const confMatch = text.match(/Confidence:\s*(\d+)\/100/i);
    const cleanText = text.replace(/Confidence:\s*\d+\/100/gi, "").trim();

    return {
      argument: cleanText,
      confidenceScore: confMatch ? confMatch[1] : null,
    };
  }

  const aiProb = useMemo(() => {
    if (!predictionResult) return 0.5;
    const probs = predictionResult.ml_probabilities || {};
    return probs["Machine"] || probs["AI"] || predictionResult.confidence || 0.5;
  }, [predictionResult]);

  return (
    <div className="council-container">
      {/* ── Council Hero Banner ── */}
      <div className="glass glass--card council-hero">
        <div className="council-hero__content">
          <div className="council-badge">Multi-Agent Dispute Resolution</div>
          <h2 className="council-hero__title">The LLM Model Council</h2>
          <p className="council-hero__desc">
            When the multi-view classifier and local agents detect evidentiary deadlock or low confidence,
            three independent external LLMs convene in a multi-round forensic debate to cross-examine
            code artifacts and determine authorship.
          </p>
        </div>

        {/* Action Controls */}
        <div className="council-hero__actions">
          {!debating && !debateDone && (
            <button
              className="btn btn--primary"
              onClick={onStartDebate}
              disabled={!code || debating}
            >
              {rounds.length > 0 ? "Restart Debate" : "Convene Council & Begin Debate"}
            </button>
          )}

          {debating && (
            <button className="btn btn--danger" onClick={onStopDebate}>
              Stop Debate Session
            </button>
          )}

          {debateDone && (
            <button className="btn btn--secondary" onClick={onStartDebate}>
              Re-run Debate
            </button>
          )}
        </div>
      </div>

      {/* ── Council Participants Card ── */}
      <div className="glass glass--card participants-grid">
        <div className="participant-card participant-card--advocate">
          <div className="participant-card__header">
            <span className="participant-role">🔴 ADVOCATE</span>
            <span className="participant-model">Groq (gpt-oss-20b)</span>
          </div>
          <p className="participant-card__mandate">
            Tasked with discovering structural markers, algorithmic uniformity, and synthetic fingerprints that argue the code is <strong>Machine-generated</strong>.
          </p>
        </div>

        <div className="participant-card participant-card--skeptic">
          <div className="participant-card__header">
            <span className="participant-role">🟢 SKEPTIC</span>
            <span className="participant-model">Gemini (gemini-3.5-flash-lite)</span>
          </div>
          <p className="participant-card__mandate">
            Tasked with identifying idiomatic developer quirks, irregular spacing, and human-typical shortcuts that argue the code is <strong>Human-written</strong>.
          </p>
        </div>

        <div className="participant-card participant-card--judge">
          <div className="participant-card__header">
            <span className="participant-role">⚖️ PRESIDING JUDGE</span>
            <span className="participant-model">Gemini (gemini-3.5-flash-lite)</span>
          </div>
          <p className="participant-card__mandate">
            Impartially weighs claims against ground-truth source lines and delivers the final verdict: <em>Human, Machine, Hybrid, or Adversarial</em>.
          </p>
        </div>
      </div>

      {/* ── Status Bar ── */}
      {debating && (
        <div className="glass council-status-bar">
          <span className="spinner" />
          <span className="council-status-text">
            Forensic agents are cross-examining code evidence in real-time...
          </span>
        </div>
      )}

      {errorEvent && (
        <div className="error-message">
          <strong>Council Session Error:</strong> {errorEvent}
        </div>
      )}

      {/* ── Debate Rounds ── */}
      {rounds.length > 0 && (
        <div className="debate-rounds-list">
          {rounds.map((round) => (
            <div key={round.roundNum} className="glass glass--card debate-round-block">
              <div className="debate-round-block__header">
                <span className="debate-round-pill">Round {round.roundNum + 1}</span>
                <span className="debate-round-subtitle">Cross-Examination & Rebuttal</span>
              </div>

              <div className="debate-entries-grid">
                {round.entries.map((entry, idx) => {
                  const isAdvocate = entry.role === "advocate";
                  const parsed = parseArgumentText(entry.content);

                  return (
                    <div
                      key={idx}
                      className={`debate-card ${isAdvocate ? "debate-card--advocate" : "debate-card--skeptic"}`}
                    >
                      <div className="debate-card__top">
                        <div className="debate-card__speaker">
                          <span className="speaker-icon">{isAdvocate ? "🔴" : "🟢"}</span>
                          <span className="speaker-name">{isAdvocate ? "Advocate" : "Skeptic"}</span>
                        </div>
                        {entry.confidence != null && (
                          <span className="speaker-conf">
                            Confidence: <strong>{entry.confidence}%</strong>
                          </span>
                        )}
                      </div>

                      <div className="debate-card__body">
                        <p className="debate-card__arg">{parsed.argument}</p>
                      </div>

                      {entry.gap != null && (
                        <div className="debate-card__footer">
                          <span>Convergence Gap: <strong>{entry.gap} pts</strong></span>
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* ── Convergence Alert ── */}
      {isConverged && (
        <div className="glass convergence-banner">
          <span>✓ Agents reached evidentiary convergence. Referring to Presiding Judge.</span>
        </div>
      )}

      {/* ── Final Council Verdict ── */}
      {judgeVerdict && (
        <div className="glass glass--card verdict-verdict-card">
          <div className="verdict-verdict-card__header">
            <span className="verdict-label">FINAL COUNCIL DECISION</span>
            <div className={`final-verdict-badge final-verdict-badge--${judgeVerdict.verdict?.toLowerCase()}`}>
              {judgeVerdict.verdict}
            </div>
          </div>

          <div className="verdict-verdict-card__meta">
            <span>Deliberation Confidence: <strong>{judgeVerdict.confidence}%</strong></span>
          </div>

          {judgeVerdict.reasoning && (
            <div className="verdict-verdict-card__reasoning">
              <strong>Decisive Evidence:</strong>
              <p>{judgeVerdict.reasoning}</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
