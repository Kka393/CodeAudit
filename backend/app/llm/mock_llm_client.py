"""
llm/mock_llm_client.py
========================
Deterministic mock LLM used when USE_MOCK_LLM=true (the default).
Lets the entire pipeline -- including the two "genuinely tool-using"
agents -- be exercised locally with zero API cost and zero network
dependency.

The mock plays the same role a real Gemini function-calling session
would: given a system prompt describing available tools and a
transcript of what's happened so far, it decides either to call one
more tool or to produce the agent's final JSON verdict. Its
decisions are deterministic (hash-seeded on the input) so repeated
runs on the same code produce the same result, which is convenient
for the mock-pipeline tests in tests/test_ml_pipeline.py.
"""

import hashlib
import json
from typing import Any, Dict, List


def _stable_seed(*parts: str) -> int:
    joined = "||".join(parts)
    return int(hashlib.sha256(joined.encode("utf-8")).hexdigest(), 16)


class MockLLMClient:
    """Drop-in stand-in for GeminiClientManager's decision interface."""

    def decide_next_step(
        self,
        agent_name: str,
        available_tools: List[str],
        steps_taken: List[Dict[str, Any]],
        max_steps: int,
    ) -> Dict[str, Any]:
        """
        Returns either:
            {"action": "call_tool", "tool": "<name>"}
        or:
            {"action": "final_answer", "predicted_class": ..., "confidence": ...,
             "evidence": [...], "rationale": "..."}
        """
        step_number = len(steps_taken)

        if step_number < min(2, max_steps):
            # Simple, fixed exploration order per agent -- a real
            # Gemini call would choose adaptively; the mock explores
            # the two most informative tools for each agent.
            preferred_order = {
                "comment_style_agent": ["get_comments", "get_identifier_names"],
                "structural_agent": ["get_ast_stats", "get_function_signatures"],
            }
            order = preferred_order.get(agent_name, available_tools)
            next_tool = order[step_number] if step_number < len(order) else available_tools[0]
            return {"action": "call_tool", "tool": next_tool}

        return self._final_answer(agent_name, steps_taken)

    def _final_answer(self, agent_name: str, steps_taken: List[Dict[str, Any]]) -> Dict[str, Any]:
        from app.config import ALL_CLASSES
        classes = ALL_CLASSES
        seed = _stable_seed(agent_name, json.dumps(steps_taken, default=str))
        predicted_class = classes[seed % len(classes)]
        # Deterministic confidence in [0.55, 0.95] from the seed.
        confidence = round(0.55 + (seed % 41) / 100.0, 2)

        evidence = []
        rationale_parts = []

        for step in steps_taken:
            tool = step["tool"]
            output = step["output"]

            if tool == "get_comments" and output:
                first_comment = output[0]
                evidence.append({"quote": first_comment, "reason": "Comment phrasing considered in the verdict."})
                rationale_parts.append(f"Inspected {len(output)} comment(s).")
            elif tool == "get_identifier_names":
                names = output.get("function_names", [])
                if names:
                    evidence.append({"quote": names[0], "reason": "Identifier naming pattern considered."})
                rationale_parts.append("Inspected identifier naming conventions.")
            elif tool == "get_ast_stats":
                rationale_parts.append(
                    f"AST stats: nesting={output.get('max_nesting_depth')}, "
                    f"cyclomatic={output.get('cyclomatic_complexity')}."
                )
            elif tool == "get_function_signatures" and output:
                evidence.append({"quote": output[0], "reason": "Function signature style considered."})
                rationale_parts.append(f"Inspected {len(output)} function signature(s).")

        if not rationale_parts:
            rationale_parts.append("Insufficient structural/style signal; low-confidence default.")

        return {
            "action": "final_answer",
            "predicted_class": predicted_class,
            "confidence": confidence,
            "evidence": evidence,
            "rationale": " ".join(rationale_parts),
        }
