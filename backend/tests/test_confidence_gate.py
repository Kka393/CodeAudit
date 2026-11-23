"""
Test 13: confidence gate.
"""

from app.aggregation.aggregator import AggregationResult
from app.aggregation.confidence_gate import apply_confidence_gate


def _aggregation(confidence, agreement):
    return AggregationResult(
        final_class="AI", final_confidence=confidence, final_probabilities={},
        agreement=agreement, ml_weight=0.5, agent_a_weight=0.25, agent_b_weight=0.25,
    )


def test_high_confidence_and_agreement_passes():
    gate = apply_confidence_gate(_aggregation(0.85, True), threshold=0.70)
    assert gate.passed is True
    assert gate.needs_debate is False


def test_low_confidence_fails_even_with_agreement():
    gate = apply_confidence_gate(_aggregation(0.50, True), threshold=0.70)
    assert gate.passed is False
    assert gate.needs_debate is True


def test_disagreement_fails_even_with_high_confidence():
    gate = apply_confidence_gate(_aggregation(0.90, False), threshold=0.70)
    assert gate.passed is False
    assert gate.needs_debate is True
