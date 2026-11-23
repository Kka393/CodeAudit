"""
aggregation/aggregator.py
============================
Deterministic aggregation of P_ml, Agent A, and Agent B into one
final probability vector. NO Gemini call happens in this module.
"""

from dataclasses import dataclass
from typing import Dict

from app.agents.base_agent import AgentResult
from app.config import AGGREGATOR_WEIGHTS, ALL_CLASSES
from app.ml.classifier import MLPrediction


@dataclass
class AggregationResult:
    final_class: str
    final_confidence: float
    final_probabilities: Dict[str, float]
    agreement: bool
    ml_weight: float
    agent_a_weight: float
    agent_b_weight: float


def agent_result_to_probs(result: AgentResult) -> Dict[str, float]:
    """Selected class gets `confidence`; the remaining 1-confidence is
    split equally across the other three classes."""
    confidence = min(max(result.confidence, 0.0), 1.0)
    remainder = (1.0 - confidence) / (len(ALL_CLASSES) - 1)

    return {
        label: confidence if _norm_cls(label) == _norm_cls(result.predicted_class) else remainder
        for label in ALL_CLASSES
    }


def _get_prob(p_dict: Dict[str, float], label: str) -> float:
    if label in p_dict:
        return p_dict[label]
    if label.lower() in ("machine", "ai"):
        for alt in ("Machine", "AI", "machine", "ai"):
            if alt in p_dict:
                return p_dict[alt]
    return 0.0


def _norm_cls(cls: str) -> str:
    l = (cls or "").lower()
    return "ai" if l in ("ai", "machine") else l


def aggregate(
    ml_prediction: MLPrediction,
    agent_a_result: AgentResult,
    agent_b_result: AgentResult,
) -> AggregationResult:
    p_ml = ml_prediction.probabilities
    p_agent_a = agent_result_to_probs(agent_a_result)
    p_agent_b = agent_result_to_probs(agent_b_result)

    w_ml = AGGREGATOR_WEIGHTS["ml"]
    w_a = AGGREGATOR_WEIGHTS["agent_a"]
    w_b = AGGREGATOR_WEIGHTS["agent_b"]

    final_probs = {
        label: (
            w_ml * _get_prob(p_ml, label)
            + w_a * _get_prob(p_agent_a, label)
            + w_b * _get_prob(p_agent_b, label)
        )
        for label in ALL_CLASSES
    }

    final_class = max(final_probs, key=final_probs.get)
    final_confidence = final_probs[final_class]

    agreement = (
        _norm_cls(ml_prediction.predicted_class)
        == _norm_cls(agent_a_result.predicted_class)
        == _norm_cls(agent_b_result.predicted_class)
    )

    return AggregationResult(
        final_class=final_class,
        final_confidence=final_confidence,
        final_probabilities=final_probs,
        agreement=agreement,
        ml_weight=w_ml,
        agent_a_weight=w_a,
        agent_b_weight=w_b,
    )
