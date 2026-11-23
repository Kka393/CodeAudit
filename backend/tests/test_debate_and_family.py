"""
test_debate_and_family.py
==========================
Tests for:
  - Custom confidence_threshold parameter in POST /predict
  - Family classifier integration in POST /predict
  - GET /debate/stream SSE endpoint
  - POST /explain endpoint
"""

import types
import pytest
import torch
from fastapi.testclient import TestClient

import app.main as main_module
from app.ml.classifier import MLPrediction
from app.ml.family_classifier import FamilyPrediction
from app.ml.feature_extractor import ViewEmbeddings


class _StubEncoder:
    match_ratio = 1.0

    def is_frozen(self):
        return True


class _StubFeatureExtractor:
    def extract(self, views):
        return ViewEmbeddings(
            e_main=torch.zeros(768),
            e_sub=torch.zeros(768),
            e_comments=torch.zeros(768),
        )


class _StubClassifierAI:
    labels = ["Human", "AI", "Hybrid", "Adversarial"]

    def predict(self, embeddings):
        return MLPrediction(
            predicted_class="AI",
            confidence=0.92,
            probabilities={"Human": 0.02, "AI": 0.92, "Hybrid": 0.03, "Adversarial": 0.03},
        )


class _StubClassifierHuman:
    labels = ["Human", "AI", "Hybrid", "Adversarial"]

    def predict(self, embeddings):
        return MLPrediction(
            predicted_class="Human",
            confidence=0.95,
            probabilities={"Human": 0.95, "AI": 0.02, "Hybrid": 0.02, "Adversarial": 0.01},
        )


class _StubFamilyClassifier:
    labels = ["GPT-4", "Claude", "Gemini", "LLaMA"]

    def predict(self, embedding):
        return FamilyPrediction(
            predicted_family="GPT-4",
            confidence=0.88,
            probabilities={"GPT-4": 0.88, "Claude": 0.07, "Gemini": 0.03, "LLaMA": 0.02},
            is_unknown=False,
        )


class _StubLLMClient:
    def decide_next_step(self, agent_name, available_tools, steps_taken, max_steps, **kwargs):
        return {
            "action": "final_answer",
            "predicted_class": "AI",
            "confidence": 0.9,
            "evidence": [],
            "rationale": "stub",
        }


def test_custom_confidence_threshold():
    stub = types.SimpleNamespace(
        encoder=_StubEncoder(),
        feature_extractor=_StubFeatureExtractor(),
        classifier=_StubClassifierAI(),
        family_classifier=None,
        llm_client=_StubLLMClient(),
    )
    main_module._pipeline = stub
    client = TestClient(main_module.app)

    # Set very high threshold (0.99) -> gate should fail / needs_debate=True
    resp = client.post(
        "/predict",
        json={"code": "def foo(): pass", "language": "python", "confidence_threshold": 0.99},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["confidence_gate"]["threshold"] == 0.99
    assert data["confidence_gate"]["needs_debate"] is True

    # Set very low threshold (0.40) -> gate should pass
    resp = client.post(
        "/predict",
        json={"code": "def foo(): pass", "language": "python", "confidence_threshold": 0.40},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["confidence_gate"]["threshold"] == 0.40
    assert data["confidence_gate"]["passed"] is True


def test_family_prediction_when_ai():
    stub = types.SimpleNamespace(
        encoder=_StubEncoder(),
        feature_extractor=_StubFeatureExtractor(),
        classifier=_StubClassifierAI(),
        family_classifier=_StubFamilyClassifier(),
        llm_client=_StubLLMClient(),
    )
    main_module._pipeline = stub
    client = TestClient(main_module.app)

    resp = client.post("/predict", json={"code": "def foo(): pass", "language": "python"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["family_prediction"] is not None
    assert data["family_prediction"]["predicted_family"] == "GPT-4"
    assert data["family_prediction"]["confidence"] == 0.88


def test_family_prediction_none_when_human():
    stub = types.SimpleNamespace(
        encoder=_StubEncoder(),
        feature_extractor=_StubFeatureExtractor(),
        classifier=_StubClassifierHuman(),
        family_classifier=_StubFamilyClassifier(),
        llm_client=_StubLLMClient(),
    )
    main_module._pipeline = stub
    client = TestClient(main_module.app)

    resp = client.post("/predict", json={"code": "def foo(): pass", "language": "python"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["family_prediction"] is None


def test_debate_stream_endpoint(monkeypatch):
    from app.agents.debate_engine import DebateEvent

    class _MockDebateEngine:
        def run_multi_stream(self, code, ai_prob, max_rounds=3):
            yield DebateEvent(event_type="round_start", round_num=0, role=None, content="")
            yield DebateEvent(event_type="advocate", round_num=0, role="advocate", content="AI code pattern", confidence=85)
            yield DebateEvent(event_type="skeptic", round_num=0, role="skeptic", content="Human style", confidence=70, gap=15)
            yield DebateEvent(event_type="judge", round_num=None, role="judge", content='{"verdict": "AI", "confidence": 80}')
            yield DebateEvent(event_type="done", round_num=None, role=None, content='{"verdict": "AI", "confidence": 80}')

    import app.api.debate as debate_module
    monkeypatch.setattr(debate_module, "_debate_engine", _MockDebateEngine())

    client = TestClient(main_module.app)
    resp = client.get("/debate/stream?code=def+foo():+pass&ai_prob=0.7&max_rounds=2")
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]
    text = resp.text
    assert "round_start" in text
    assert "advocate" in text
    assert "skeptic" in text
    assert "judge" in text
    assert "done" in text


def test_explain_endpoint_success(monkeypatch):
    from app.ml.shap_explainer import CodeExplanation, LineAttribution

    class _MockLineExplainer:
        def explain(self, code, language="python", target_class=None):
            return CodeExplanation(
                target_class="Machine",
                baseline_confidence=0.88,
                lines=[
                    LineAttribution(1, "def foo():", 0.0, "neutral", False),
                    LineAttribution(2, "    return 42", 0.15, "positive", True),
                ],
                summary={"total_lines": 2, "evaluated_lines": 2},
            )

    import app.api.explain as explain_module
    monkeypatch.setattr(explain_module, "_line_explainer", _MockLineExplainer())

    client = TestClient(main_module.app)
    resp = client.post("/explain", json={"code": "def foo():\n    return 42", "language": "python"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["target_class"] == "Machine"
    assert data["baseline_confidence"] == 0.88
    assert len(data["lines"]) == 2
    assert data["lines"][1]["direction"] == "positive"
    assert data["lines"][1]["attribution"] == 0.15


def test_explain_endpoint_empty_code():
    client = TestClient(main_module.app)
    resp = client.post("/explain", json={"code": "", "language": "python"})
    assert resp.status_code == 400


