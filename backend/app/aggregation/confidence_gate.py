"""
aggregation/confidence_gate.py
=================================
needs_debate = final_confidence < CONFIDENCE_THRESHOLD OR agreement == false

This phase NEVER runs an actual debate -- see agents/debate_stub.py.
"""

from dataclasses import dataclass

from app.aggregation.aggregator import AggregationResult
from app.config import CONFIDENCE_THRESHOLD


@dataclass
class ConfidenceGateResult:
    threshold: float
    passed: bool
    needs_debate: bool


def apply_confidence_gate(aggregation: AggregationResult, threshold: float = CONFIDENCE_THRESHOLD) -> ConfidenceGateResult:
    passed = aggregation.final_confidence >= threshold and aggregation.agreement
    needs_debate = not passed

    return ConfidenceGateResult(threshold=threshold, passed=passed, needs_debate=needs_debate)
