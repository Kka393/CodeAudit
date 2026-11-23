"""
Test 14: POST /predict. Stubs out the ML branch and LLM client via
main.Pipeline so this test needs neither the real ModernBERT
checkpoint nor the real classifier checkpoint nor Gemini access.
"""

import types

import pytest
import torch
from fastapi.testclient import TestClient

import app.main as main_module
from app.agents.base_agent import AgentResult
from app.ml.classifier import MLPrediction
from app.ml.feature_extractor import ViewEmbeddings


class _StubEncoder:
    match_ratio = 1.0

    def is_frozen(self):
        return True


class _StubFeatureExtractor:
    def extract(self, views):
        return ViewEmbeddings(e_main=torch.zeros(768), e_sub=torch.zeros(768), e_comments=torch.zeros(768))


class _StubClassifier:
    labels = ["Human", "AI", "Hybrid", "Adversarial"]

    def predict(self, embeddings):
        return MLPrediction(
            predicted_class="AI", confidence=0.9,
            probabilities={"Human": 0.02, "AI": 0.9, "Hybrid": 0.04, "Adversarial": 0.04},
        )


class _StubLLMClient:
    def decide_next_step(self, agent_name, available_tools, steps_taken, max_steps, **kwargs):
        if len(steps_taken) < max_steps:
            return {"action": "call_tool", "tool": available_tools[0]}
        return {
            "action": "final_answer", "predicted_class": "AI", "confidence": 0.85,
            "evidence": [], "rationale": "stubbed",
        }


@pytest.fixture(autouse=True)
def stub_pipeline(monkeypatch):
    stub = types.SimpleNamespace(
        encoder=_StubEncoder(),
        feature_extractor=_StubFeatureExtractor(),
        classifier=_StubClassifier(),
        llm_client=_StubLLMClient(),
    )
    monkeypatch.setattr(main_module, "_pipeline", stub)
    yield


def test_predict_returns_structured_result():
    client = TestClient(main_module.app)

    response = client.post("/predict", json={"code": "def f(x):\n    return x", "language": "python"})
    assert response.status_code == 200

    body = response.json()
    assert body["prediction"] in ["Human", "AI", "Machine", "Hybrid", "Adversarial"]
    assert "ml_probabilities" in body
    assert "comment_style_agent" in body
    assert "structural_agent" in body
    assert "confidence_gate" in body
    assert body["debate"]["debate_applied"] is False


def test_predict_rejects_empty_code():
    client = TestClient(main_module.app)
    response = client.post("/predict", json={"code": "", "language": "python"})
    assert response.status_code == 400


def test_predict_rejects_unsupported_language():
    client = TestClient(main_module.app)
    response = client.post("/predict", json={"code": "print(1)", "language": "cobol"})
    assert response.status_code == 400
