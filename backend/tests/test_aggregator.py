"""
Test 12: aggregator.
"""

from app.agents.base_agent import AgentResult
from app.aggregation.aggregator import aggregate, agent_result_to_probs
from app.ml.classifier import MLPrediction


def _agent(predicted_class, confidence):
    return AgentResult(
        predicted_class=predicted_class, confidence=confidence, evidence=[],
        rationale="", tool_calls_made=[], num_steps=2, failed=False,
    )


def test_agent_result_to_probs_splits_remainder_equally():
    result = _agent("AI", 0.80)
    probs = agent_result_to_probs(result)

    ai_key = "AI" if "AI" in probs else "Machine"
    assert probs[ai_key] == 0.80
    assert abs(probs["Human"] - probs["Hybrid"]) < 1e-9
    assert abs(sum(probs.values()) - 1.0) < 1e-9


def test_aggregate_agreement_true_when_all_three_match():
    ml = MLPrediction(
        predicted_class="AI", confidence=0.9,
        probabilities={"Human": 0.02, "AI": 0.9, "Hybrid": 0.04, "Adversarial": 0.04},
    )
    agent_a = _agent("AI", 0.87)
    agent_b = _agent("AI", 0.82)

    result = aggregate(ml, agent_a, agent_b)
    assert result.agreement is True
    assert result.final_class in ("AI", "Machine")
    assert result.final_confidence > 0.7


def test_aggregate_agreement_false_on_disagreement():
    ml = MLPrediction(
        predicted_class="AI", confidence=0.9,
        probabilities={"Human": 0.02, "AI": 0.9, "Hybrid": 0.04, "Adversarial": 0.04},
    )
    agent_a = _agent("Human", 0.6)
    agent_b = _agent("Hybrid", 0.55)

    result = aggregate(ml, agent_a, agent_b)
    assert result.agreement is False
