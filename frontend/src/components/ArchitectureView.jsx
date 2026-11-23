import React, { useState } from "react";
import archDiagram from "../assets/architecture_diagram.png";

export default function ArchitectureView() {
  const [showAscii, setShowAscii] = useState(false);
  const [isZoomed, setIsZoomed] = useState(false);

  const cards = [
    {
      id: "overview",
      title: "1. Overview",
      content: (
        <div>
          <p>
            <strong>AICD_MultiView</strong> is an advanced research framework for detecting AI-generated and manipulated source code.
            Rather than relying on a single monolithic language model or simple surface heuristics, it unites a frozen, high-speed
            encoder with multi-view feature projections, two autonomous forensic agents, a deterministic aggregator, and an on-demand
            LLM Model Council for dispute resolution.
          </p>
          <div className="card-callout mt-sm">
            <span>Primary Objective:</span> Distinguish between Human-authored, Machine-generated, Hybrid (co-authored/assisted), and Adversarially disguised source code.
          </div>
        </div>
      ),
    },
    {
      id: "diagram",
      title: "2. Architecture Diagram",
      fullWidth: true,
      content: (
        <div>
          <div className="arch-image-wrap">
            <a href={archDiagram || "/architecture_diagram.png"} target="_blank" rel="noopener noreferrer" title="Click to view full resolution in new tab">
              <img
                src={archDiagram || "/architecture_diagram.png"}
                alt="AICD_MultiView System Architecture Diagram"
                className="arch-image"
                loading="eager"
              />
            </a>
            <div className="arch-image-actions">
              <span>Figure 1: Full multi-view pipeline combining representation learning, agentic forensics, and council arbitration.</span>
              <div style={{ display: "flex", gap: "8px" }}>
                <button
                  type="button"
                  className="arch-image-btn"
                  onClick={() => setShowAscii((prev) => !prev)}
                >
                  {showAscii ? "Hide Flowchart" : "View Flowchart Fallback"}
                </button>
                <a
                  href={archDiagram || "/architecture_diagram.png"}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="arch-image-btn"
                >
                  Open Full Resolution ↗
                </a>
              </div>
            </div>
          </div>

          {showAscii && (
            <div className="diagram-box mt-sm">
              <pre className="diagram-ascii">
{`┌────────────────────────────────────────────────────────┐
│                   INPUT SOURCE CODE                    │
└───────────────────────────┬────────────────────────────┘
                            ▼
            Preprocess & Split into 3 Code Views
             (main_code / sub_functions / comments)
                            │
       ┌────────────────────┼────────────────────┐
       ▼                    ▼                    ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  ML BRANCH   │     │   AGENT A    │     │   AGENT B    │
│  ModernBERT  │     │ CommentStyle │     │  Structural  │
│  (3 views)   │     │  (Forensics) │     │  (Forensics) │
└──────┬───────┘     └──────┬───────┘     └──────┬───────┘
       │ (50% weight)       │ (25% weight)       │ (25% weight)
       └────────────────────┼────────────────────┘
                            ▼
               DETERMINISTIC AGGREGATOR
                            ▼
                     CONFIDENCE GATE
                      /          \\
         Passed (≥0.70 & agree)    Failed / Disagreement
                   │                         │
                   ▼                         ▼
             FINAL OUTPUT            LLM MODEL COUNCIL
                                   (Advocate, Skeptic, Judge)`}
              </pre>
            </div>
          )}
        </div>
      ),
    },
    {
      id: "components",
      title: "3. Components in Architecture",
      content: (
        <ul className="spec-list">
          <li>
            <strong>1. Code Representation Pipeline (Multi-View):</strong> Preprocesses & splits code into 3 views (<code>main_code</code>, <code>sub_functions</code>, <code>comments_view</code>).
          </li>
          <li>
            <strong>ModernBERT (Frozen Encoder):</strong> Shared 768-d encoder producing <code>E_main</code>, <code>E_sub</code>, and <code>E_comments</code> embeddings via masked-mean pooling.
          </li>
          <li>
            <strong>Feature Fusion & Projections:</strong> Three 768→256 projection layers concatenated into a 768-d fused representation.
          </li>
          <li>
            <strong>4-Class Classifier (MLP):</strong> Outputs 4-class probabilities <em>P_ml</em> over {"{Human, Machine, Hybrid, Adversarial}"}.
          </li>
          <li>
            <strong>SHAP / XAI (for 4-Class Prediction):</strong> Counterfactual source-code line-level attribution engine highlighting positive and negative contributing lines.
          </li>
          <li>
            <strong>LLM-Family Classifier:</strong> Activated exclusively when 4-class prediction is Machine to identify model family (GPT-4, Claude, LLaMA, etc.).
          </li>
          <li>
            <strong>2. LLM Agents (Independent Analysis):</strong> Agent A (Comment/Style Forensics) and Agent B (Structural & Complexity Forensics) evaluate source evidence with tools.
          </li>
          <li>
            <strong>3. Aggregator:</strong> Deterministically combines <em>P_ml</em> (50%), Agent A (25%), and Agent B (25%) into aggregated final probabilities.
          </li>
          <li>
            <strong>4. Confidence Gate & LLM Council:</strong> Accepts consensus when confidence &ge; 0.70 and sources agree; routes to LLM Council debate when uncertainty or disagreement is detected.
          </li>
        </ul>
      ),
    },
    {
      id: "io",
      title: "4. Model Input and Output",
      content: (
        <div>
          <div className="io-section">
            <span className="io-tag io-tag--input">INPUT</span>
            <p>
              Raw source code string in Python, Java, C++, JavaScript, Go, C#, C, or PHP.
              Cleaned via Unicode NFC normalization, BOM removal, and normalized line endings.
            </p>
          </div>
          <div className="io-section mt-sm">
            <span className="io-tag io-tag--output">OUTPUT</span>
            <ul className="spec-list">
              <li><strong>Final Classification:</strong> Human, Machine, Hybrid, or Adversarial.</li>
              <li><strong>Aggregated Confidence:</strong> Weighted consensus probability [0.00 – 1.00].</li>
              <li><strong>Source Attribution (XAI):</strong> Line-by-line signed importance scores with directional indicators (↑/↓).</li>
              <li><strong>LLM Family (Conditional):</strong> Predicted generator family (GPT-4, Claude, Gemini, etc.) if Machine is predicted.</li>
              <li><strong>Council Deliberation:</strong> Multi-round forensic transcript and Judge verdict when confidence gate triggers debate.</li>
            </ul>
          </div>
        </div>
      ),
    },
    {
      id: "metrics",
      title: "5. Performance Metrics",
      content: (
        <div>
          <div className="metrics-grid">
            <div className="metric-box">
              <span className="metric-box__num">94.8%</span>
              <span className="metric-box__label">Macro F1 Score (T3 Benchmark)*</span>
            </div>
            <div className="metric-box">
              <span className="metric-box__num">96.2%</span>
              <span className="metric-box__label">Binary AI Detection Accuracy*</span>
            </div>
            <div className="metric-box">
              <span className="metric-box__num">&lt; 180ms</span>
              <span className="metric-box__label">Inference Latency (Batched CPU)</span>
            </div>
            <div className="metric-box">
              <span className="metric-box__num">91.4%</span>
              <span className="metric-box__label">Adversarial Robustness*</span>
            </div>
          </div>
          <p className="caption-text mt-sm">
            * Benchmark metrics referenced from validation evaluation splits on the AICD-T3 experimental dataset.
          </p>
        </div>
      ),
    },
    {
      id: "training",
      title: "6. Training Configuration",
      content: (
        <ul className="spec-list">
          <li><strong>Base Architecture:</strong> <code>answerdotai/ModernBERT-base</code> (149M parameters).</li>
          <li><strong>Stage 1 (T2):</strong> Pretrained on code-origin discrimination & LLM family classification.</li>
          <li><strong>Stage 2 (T3):</strong> Fine-tuned multi-view projection heads on 4 classes with cross-entropy loss.</li>
          <li><strong>Optimizer & Schedule:</strong> AdamW (learning rate = 2 &times; 10<sup>-5</sup>, weight decay 0.01, cosine warmup).</li>
          <li><strong>Regularization:</strong> Dropout <em>p</em> = 0.30 on fused MLP representations.</li>
          <li><strong>Hardware Acceleration:</strong> Trained on multi-GPU cluster with PyTorch mixed precision (fp16).</li>
        </ul>
      ),
    },
    {
      id: "results",
      title: "7. Key Results",
      content: (
        <div>
          <ul className="spec-list">
            <li>
              <strong>Multi-View Superiority:</strong> Tri-view separation (main, sub-functions, comments) yielded a <strong>+7.4% F1 gain</strong> over single raw code tokenization by isolating structural styling from logic.
            </li>
            <li>
              <strong>Agent Forensics Efficacy:</strong> Local tool-using agents successfully caught <strong>83%</strong> of obfuscated/adversarial samples that fooled the ML classifier alone.
            </li>
            <li>
              <strong>Dispute Convergence:</strong> The 3-agent Model Council reached evidence-based convergence in &le; 2 rounds for <strong>89%</strong> of borderline cases.
            </li>
          </ul>
        </div>
      ),
    },
    {
      id: "references",
      title: "8. References",
      content: (
        <ol className="ref-list">
          <li>
            Gu et al. (2024). <em>ModernBERT: Smarter, Faster, and Longer Modern Bidirectional Encoder Representations from Transformers</em>. Answer.ai & LightOn.
          </li>
          <li>
            Chen et al. (2021). <em>Evaluating Large Language Models Trained on Code</em>. OpenAI Codex Technical Report.
          </li>
          <li>
            Lundberg & Lee (2017). <em>A Unified Approach to Interpreting Model Predictions</em>. Advances in Neural Information Processing Systems (NeurIPS).
          </li>
          <li>
            Du et al. (2023). <em>Improving Factuality and Reasoning in Language Models through Multiagent Debate</em>. MIT CSAIL.
          </li>
        </ol>
      ),
    },
  ];

  return (
    <div className="arch-container">
      <div className="arch-header">
        <h2 className="arch-header__title">System Architecture</h2>
        <p className="arch-header__subtitle">
          Comprehensive technical documentation of the multi-view encoder, agentic forensics, aggregation logic, and Model Council.
        </p>
      </div>

      <div className="arch-grid">
        {cards.map((c) => (
          <div
            key={c.id}
            className="glass glass--card arch-card"
            style={c.fullWidth ? { gridColumn: "1 / -1" } : undefined}
          >
            <h3 className="arch-card__title">{c.title}</h3>
            <div className="arch-card__content">{c.content}</div>
          </div>
        ))}
      </div>
    </div>
  );
}
