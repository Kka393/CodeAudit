"""
agents/structural_agent.py
=============================
Agent B: decides Human/AI/Hybrid/Adversarial from structural/
complexity evidence ONLY (never raw source text or comments), by
genuinely choosing which StructuralToolset methods to call (up to
MAX_AGENT_STEPS).
"""

import logging

from app.agents.base_agent import AgentResult, run_agent_loop
from app.agents.tools_structural import StructuralToolset
from app.config import MAX_AGENT_STEPS
from app.ml.code_splitter import CodeViews

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a code-forensics agent whose ONLY job is to judge whether a code "
    "sample's STRUCTURE and COMPLEXITY (nesting, cyclomatic complexity, function "
    "signatures, control-flow shape) look Human-written, AI-generated, a Hybrid "
    "of both, or Adversarially disguised. You must NOT use raw source text or "
    "comments as evidence -- only structural/complexity metrics from your tools."
)

TOOL_DESCRIPTIONS = {
    "get_ast_stats": "Returns max_nesting_depth, cyclomatic_complexity, num_functions, avg_function_length.",
    "get_function_signatures": "Returns function/method headers only (no bodies).",
    "get_control_flow_summary": "Returns counts of if/for/while/try across extracted functions.",
    "compare_complexity_to_typical": "Given a metric name and value, returns a 0-1 typicality heuristic score.",
}


def run_structural_agent(views: CodeViews, language: str, llm_client) -> AgentResult:
    toolset = StructuralToolset(views, language=language)

    def _compare_default():
        stats = toolset.get_ast_stats()
        return toolset.compare_complexity_to_typical("cyclomatic_complexity", stats["cyclomatic_complexity"])

    tool_registry = {
        "get_ast_stats": toolset.get_ast_stats,
        "get_function_signatures": toolset.get_function_signatures,
        "get_control_flow_summary": toolset.get_control_flow_summary,
        "compare_complexity_to_typical": _compare_default,
    }

    try:
        return run_agent_loop(
            agent_name="structural_agent",
            toolset=toolset,
            tool_registry=tool_registry,
            tool_descriptions=TOOL_DESCRIPTIONS,
            system_prompt=SYSTEM_PROMPT,
            llm_client=llm_client,
            max_steps=MAX_AGENT_STEPS,
        )
    except Exception as exc:  # noqa: BLE001 - one failed agent must never crash the pipeline
        logger.exception("structural_agent failed: %s", exc)
        return AgentResult(
            predicted_class="Human",
            confidence=0.0,
            evidence=[],
            rationale=f"Agent failed: {exc}",
            tool_calls_made=[],
            num_steps=0,
            failed=True,
        )
