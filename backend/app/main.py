"""
app/main.py
============
Application entry point AND the visible, single place where the
full pipeline is orchestrated:

    input
      |
      v
   preprocess + split code
      |
      +----------------+----------------+
      |                |                |
      v                v                v
   ML branch      Agent A branch   Agent B branch
      |                |                |
      +-------+--------+----------------+
              |
              v
          aggregator
              |
              v
        confidence gate
              |
              v
    (needs_debate ? debate_stub : nothing further)
              |
              v
          final result

Individual modules (ml/, agents/, aggregation/, llm/) only DEFINE
their classes/functions -- importing them never runs the pipeline.
main.py is what calls them, in this order. FastAPI route handlers in
api/predict.py call run_full_pipeline() defined here; they do not
reimplement any of this orchestration.

Run the API with:
    uvicorn app.main:app --reload --port 8000
"""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from typing import Optional

from app.agents.comment_style_agent import run_comment_style_agent
from app.agents.debate_stub import run_debate_loop
from app.agents.structural_agent import run_structural_agent
from app.aggregation.aggregator import aggregate
from app.aggregation.confidence_gate import apply_confidence_gate
from app.config import CONFIDENCE_THRESHOLD, SUPPORTED_LANGUAGES, USE_MOCK_LLM
from app.ml.classifier import Classifier
from app.ml.code_splitter import split_code
from app.ml.family_classifier import FamilyClassifier
from app.ml.feature_extractor import FeatureExtractor
from app.ml.modernbert_encoder import ModernBertEncoder

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)-7s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


class PipelineError(RuntimeError):
    pass


class Pipeline:
    """
    Holds the heavy, load-once components (frozen ModernBERT encoder
    + trained classifier + LLM client) so they are constructed a
    single time at process startup, not once per request.
    """

    def __init__(self):
        logger.info("Loading ML branch (ModernBERT encoder + trained classifier)...")
        self.encoder = ModernBertEncoder()
        self.feature_extractor = FeatureExtractor(self.encoder)
        self.classifier = Classifier()
        logger.info(
            "ML branch ready | encoder match_ratio=%.3f | labels=%s",
            self.encoder.match_ratio, self.classifier.labels,
        )

        # Family classifier (Stage 6 head) — activated only when ML predicts AI/Machine
        try:
            self.family_classifier = FamilyClassifier()
            logger.info(
                "Family classifier ready | labels=%s",
                self.family_classifier.labels,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Family classifier could not be loaded (%s). "
                "LLM-family predictions will be disabled.", exc,
            )
            self.family_classifier = None

        if USE_MOCK_LLM:
            from app.llm.mock_llm_client import MockLLMClient
            self.llm_client = MockLLMClient()
            logger.info("USE_MOCK_LLM=true -- agents will use MockLLMClient (no Gemini calls).")
        else:
            from app.llm.gemini_client_manager import GeminiClientManager
            self.llm_client = GeminiClientManager()
            logger.info("USE_MOCK_LLM=false -- agents will call Gemini via GeminiClientManager.")


_pipeline: Pipeline = None  # constructed lazily on first request or at FastAPI startup


def get_pipeline() -> Pipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = Pipeline()
    return _pipeline


def run_full_pipeline(
    code: str,
    language: str,
    confidence_threshold: Optional[float] = None,
) -> dict:
    """The single application-level orchestration function. Every
    step below is a call into a module that only defines behavior;
    this function decides the order.

    Args:
        code: Raw source code to analyze.
        language: Programming language ("python", "java", etc.).
        confidence_threshold: Per-request override for the confidence
            gate threshold. If None, uses CONFIDENCE_THRESHOLD from config.
    """
    if not code or not code.strip():
        raise PipelineError("`code` must be a non-empty string.")

    language = (language or "").lower()
    if language and language not in SUPPORTED_LANGUAGES:
        raise PipelineError(
            f"Unsupported language '{language}'. Supported: {SUPPORTED_LANGUAGES}"
        )

    threshold = confidence_threshold if confidence_threshold is not None else CONFIDENCE_THRESHOLD
    pipeline = get_pipeline()

    # 1. Preprocess + split code (reused for BOTH the ML branch and
    #    both agents' toolsets -- never re-parsed independently).
    views = split_code(code, language=language)

    # 2. ML branch.
    embeddings = pipeline.feature_extractor.extract(views)
    ml_prediction = pipeline.classifier.predict(embeddings)

    # 2b. Family classifier (only when ML predicts AI/Machine).
    family_prediction = None
    _ai_labels = {"ai", "machine"}
    fc = getattr(pipeline, "family_classifier", None)
    if (fc is not None
            and ml_prediction.predicted_class.lower() in _ai_labels):
        try:
            family_pred = fc.predict(embeddings.e_main)
            family_prediction = {
                "predicted_family": family_pred.predicted_family,
                "confidence": family_pred.confidence,
                "probabilities": family_pred.probabilities,
                "is_unknown": family_pred.is_unknown,
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning("Family classifier failed at inference time: %s", exc)
            family_prediction = None

    # 3. Agent branches.
    agent_a_result = run_comment_style_agent(views, language, pipeline.llm_client)
    agent_b_result = run_structural_agent(views, language, pipeline.llm_client)

    # 4. Aggregator (deterministic, no LLM call inside it).
    aggregation = aggregate(ml_prediction, agent_a_result, agent_b_result)

    # 5. Confidence gate (uses per-request threshold if provided).
    gate = apply_confidence_gate(aggregation, threshold=threshold)

    # 6. Debate stub (never actually debates in this phase —
    #    the real debate is triggered separately via /debate/stream).
    evidence_packet = {
        "ml_prediction": ml_prediction,
        "agent_a_result": agent_a_result,
        "agent_b_result": agent_b_result,
        "aggregation": aggregation,
        "needs_debate": gate.needs_debate,
    }
    if gate.needs_debate:
        evidence_packet = run_debate_loop(evidence_packet)

    # 7. Build the final structured result.
    return {
        "prediction": aggregation.final_class,
        "confidence": aggregation.final_confidence,

        "ml_prediction": ml_prediction.predicted_class,
        "ml_probabilities": ml_prediction.probabilities,

        # Family prediction (null if ML class is not AI/Machine)
        "family_prediction": family_prediction,

        "comment_style_agent": {
            "predicted_class": agent_a_result.predicted_class,
            "confidence": agent_a_result.confidence,
            "evidence": agent_a_result.evidence,
            "rationale": agent_a_result.rationale,
            "tool_calls_made": agent_a_result.tool_calls_made,
            "num_steps": agent_a_result.num_steps,
            "failed": agent_a_result.failed,
        },
        "structural_agent": {
            "predicted_class": agent_b_result.predicted_class,
            "confidence": agent_b_result.confidence,
            "evidence": agent_b_result.evidence,
            "rationale": agent_b_result.rationale,
            "tool_calls_made": agent_b_result.tool_calls_made,
            "num_steps": agent_b_result.num_steps,
            "failed": agent_b_result.failed,
        },

        "aggregation": {
            "ml_weight": aggregation.ml_weight,
            "agent_a_weight": aggregation.agent_a_weight,
            "agent_b_weight": aggregation.agent_b_weight,
            "agreement": aggregation.agreement,
        },

        "confidence_gate": {
            "threshold": threshold,
            "passed": gate.passed,
            "needs_debate": gate.needs_debate,
        },

        "debate": {
            "triggered": gate.needs_debate,
            "debate_applied": False,
        },
    }


def create_app() -> FastAPI:
    app = FastAPI(
        title="CodeAudit",
        description="Multi-view AI-generated-code detector: ML branch + two tool-using agents + deterministic aggregator + confidence gate.",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    from app.api.predict import router as predict_router
    app.include_router(predict_router)

    from app.api.debate import router as debate_router
    app.include_router(debate_router)

    from app.api.explain import router as explain_router
    app.include_router(explain_router)

    return app


app = create_app()
