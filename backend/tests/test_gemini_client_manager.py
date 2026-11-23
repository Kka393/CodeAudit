"""
Test 11: GeminiClientManager. Exercises the retry/failover logic
using a stubbed _call_once so no real network/API access is needed.
"""

import json

from app.llm.api_key_manager import APIKeyManager
from app.llm.gemini_client_manager import GeminiClientManager


def test_decide_next_step_parses_json_and_retries_across_keys(monkeypatch):
    key_manager = APIKeyManager(keys=["KEY1", "KEY2"], cooldown_seconds=60)
    manager = GeminiClientManager(model="fake-model", key_manager=key_manager)

    calls = {"count": 0}

    def fake_call_once(self, key, prompt):
        calls["count"] += 1
        if key == "KEY1":
            raise RuntimeError("429 rate limit exceeded")
        return json.dumps({"action": "call_tool", "tool": "get_comments"})

    monkeypatch.setattr(GeminiClientManager, "_call_once", fake_call_once)

    decision = manager.decide_next_step(
        agent_name="comment_style_agent",
        available_tools=["get_comments"],
        steps_taken=[],
        max_steps=2,
        tool_descriptions={"get_comments": "..."},
        system_prompt="test",
    )

    assert decision == {"action": "call_tool", "tool": "get_comments"}
    assert calls["count"] == 2  # KEY1 failed, KEY2 succeeded
    assert key_manager.available_key_count() == 1  # KEY1 in cooldown


def test_decide_next_step_extracts_json_from_markdown_fence(monkeypatch):
    key_manager = APIKeyManager(keys=["KEY1"], cooldown_seconds=60)
    manager = GeminiClientManager(model="fake-model", key_manager=key_manager)

    def fake_call_once(self, key, prompt):
        return '```json\n{"action": "final_answer", "predicted_class": "AI", "confidence": 0.9, "evidence": [], "rationale": "x"}\n```'

    monkeypatch.setattr(GeminiClientManager, "_call_once", fake_call_once)

    decision = manager.decide_next_step(
        agent_name="structural_agent",
        available_tools=[],
        steps_taken=[],
        max_steps=0,
        tool_descriptions={},
        system_prompt="test",
    )

    assert decision["predicted_class"] == "AI"
    assert decision["confidence"] == 0.9
