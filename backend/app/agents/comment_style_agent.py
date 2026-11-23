"""
agents/comment_style_agent.py
================================
Agent A: decides Human/AI/Hybrid/Adversarial from comment and
identifier-naming style evidence, by genuinely choosing which
CommentStyleToolset methods to call (up to MAX_AGENT_STEPS) rather
than being handed a pre-built evidence packet.
"""

import logging

from app.agents.base_agent import AgentResult, run_agent_loop
from app.agents.tools_comment_style import CommentStyleToolset
from app.config import MAX_AGENT_STEPS
from app.ml.code_splitter import CodeViews

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a code-forensics agent whose ONLY job is to judge whether a code "
    "sample's COMMENTS and IDENTIFIER NAMING look Human-written, AI-generated, "
    "a Hybrid of both, or Adversarially disguised. Use the tools available to "
    "gather evidence before answering; do not invent facts not returned by a tool."
)

TOOL_DESCRIPTIONS = {
    "get_comments": "Returns the extracted comments from this code sample, one per item.",
    "get_identifier_names": "Returns function/variable/parameter names used in the code.",
    "get_docstrings": "Returns docstring-like comments, if any (may be empty).",
    "compare_style_to_reference": "Given a text string, returns a 0-1 stylistic-consistency heuristic score.",
}


def run_comment_style_agent(views: CodeViews, language: str, llm_client) -> AgentResult:
    toolset = CommentStyleToolset(views, language=language)

    tool_registry = {
        "get_comments": toolset.get_comments,
        "get_identifier_names": toolset.get_identifier_names,
        "get_docstrings": toolset.get_docstrings,
        "compare_style_to_reference": lambda: toolset.compare_style_to_reference(views.comments_view),
    }

    try:
        return run_agent_loop(
            agent_name="comment_style_agent",
            toolset=toolset,
            tool_registry=tool_registry,
            tool_descriptions=TOOL_DESCRIPTIONS,
            system_prompt=SYSTEM_PROMPT,
            llm_client=llm_client,
            max_steps=MAX_AGENT_STEPS,
        )
    except Exception as exc:  # noqa: BLE001 - one failed agent must never crash the pipeline
        logger.exception("comment_style_agent failed: %s", exc)
        return AgentResult(
            predicted_class="Human",
            confidence=0.0,
            evidence=[],
            rationale=f"Agent failed: {exc}",
            tool_calls_made=[],
            num_steps=0,
            failed=True,
        )
