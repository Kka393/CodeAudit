"""
agents/base_agent.py
======================
Shared agentic tool-use loop used by both Agent A (Comment/Style) and
Agent B (Structural). Each agent:

  1. Asks the LLM (real Gemini via GeminiClientManager, or
     MockLLMClient) what to do next: call one more tool, or give a
     final answer.
  2. Executes any requested tool call against ITS OWN toolset and
     appends the tool + its exact output to the running transcript.
  3. Stops after at most MAX_AGENT_STEPS tool calls and forces a
     final answer.
  4. Validates the final answer's evidence quotes against the exact
     tool outputs collected during the run. One corrective retry is
     allowed; if evidence still doesn't validate, the agent is
     marked failed with confidence=0.0 -- it does NOT crash the
     pipeline.

CLASS LABEL NORMALIZATION
--------------------------
Real LLMs do not reliably answer with the exact string in
config.ALL_CLASSES ("Human", "Machine", "Hybrid", "Adversarial" or
whatever CLASS_LABEL_ORDER is set to). _normalize_class_label() maps
many common phrasings (including the OLD "AI" wording, in case the
prompt or the model drifts) onto the CURRENT ALL_CLASSES by index:

    ALL_CLASSES[0] = Human      (or whatever index 0 is set to)
    ALL_CLASSES[1] = Machine/AI (or whatever index 1 is set to)
    ALL_CLASSES[2] = Hybrid     (or whatever index 2 is set to)
    ALL_CLASSES[3] = Adversarial (or whatever index 3 is set to)

It is deliberately driven by ALL_CLASSES rather than hardcoded
strings, so it stays correct if CLASS_LABEL_ORDER ever changes.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from app.config import ALL_CLASSES, MAX_AGENT_STEPS

logger = logging.getLogger(__name__)


@dataclass
class AgentResult:
    predicted_class: str
    confidence: float
    evidence: List[Dict[str, str]]
    rationale: str
    tool_calls_made: List[str]
    num_steps: int
    failed: bool = False


def _stringify(output: Any) -> str:
    """Flatten a tool output into a single string for substring
    checks, regardless of whether it's a str, list, or dict.

    IMPORTANT: for dicts, this includes the KEYS (as "key: value"),
    not just the values -- Gemini is shown tool outputs as JSON
    (which includes keys), so if we silently dropped keys here, any
    quote that legitimately referenced a metric name (e.g.
    "cyclomatic_complexity: 2") would fail validation even though it
    was copied verbatim from what the model actually saw.
    """
    if isinstance(output, str):
        return output
    if isinstance(output, (list, tuple)):
        return " ".join(_stringify(o) for o in output)
    if isinstance(output, dict):
        return " ".join(f"{k}: {_stringify(v)}" for k, v in output.items())
    return str(output)


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _validate_evidence(evidence: List[Dict[str, str]], steps_taken: List[Dict[str, Any]]) -> bool:
    if not evidence:
        return True  # no evidence cited is valid (just unsupported), not invalid

    # Two representations of the same tool outputs, since the model
    # is shown a JSON view (in the prompt's history_block) but might
    # instead be quoting from a flattened reading of it -- accept a
    # quote that matches EITHER, so a verbatim copy from what the
    # model actually saw is never unfairly rejected. Whitespace is
    # normalized because json.dumps(..., indent=2) introduces
    # newlines/indentation that a copied quote may not preserve.
    import json as _json

    stringified_haystack = _normalize_whitespace(
        " ".join(_stringify(step["output"]) for step in steps_taken)
    )
    json_haystack = _normalize_whitespace(
        " ".join(_json.dumps(step["output"], default=str) for step in steps_taken)
    )

    for item in evidence:
        quote = item.get("quote", "")
        if not quote:
            return False
        normalized_quote = _normalize_whitespace(quote)
        if normalized_quote not in stringified_haystack and normalized_quote not in json_haystack:
            return False
    return True


# ============================================================
# CLASS NORMALIZATION
# ============================================================

def _normalize_class_label(label: Any) -> Optional[str]:
    """
    Convert many different phrasings of the four classes into the
    CURRENT canonical ALL_CLASSES names (index-based, so this stays
    correct if CLASS_LABEL_ORDER changes in .env).
    """
    if label is None:
        return None

    value = str(label).strip()
    if not value:
        return None

    # Remove surrounding quotes/backticks
    value = value.strip("`'\"")

    # Normalize case and whitespace
    normalized = re.sub(r"\s+", " ", value.lower()).strip()

    # --------------------------------------------------------
    # Numeric labels
    # --------------------------------------------------------
    if normalized in {"0", "class_0", "class 0"}:
        return ALL_CLASSES[0]
    if normalized in {"1", "class_1", "class 1"}:
        return ALL_CLASSES[1]
    if normalized in {"2", "class_2", "class 2"}:
        return ALL_CLASSES[2]
    if normalized in {"3", "class_3", "class 3"}:
        return ALL_CLASSES[3]

    # --------------------------------------------------------
    # Human
    # --------------------------------------------------------
    human_labels = {
        "human", "human-written", "human written", "human-written code",
        "human code", "purely human", "pure human",
    }
    if normalized in human_labels:
        return ALL_CLASSES[0]

    # --------------------------------------------------------
    # Machine / AI-generated
    # --------------------------------------------------------
    machine_labels = {
        "machine", "machine-generated", "machine generated",
        "machine-generated code", "ai", "ai-generated", "ai generated",
        "ai-generated code", "fully ai-generated", "fully ai generated",
        "generated by ai", "artificial intelligence generated",
    }
    if normalized in machine_labels:
        return ALL_CLASSES[1]

    # --------------------------------------------------------
    # Hybrid
    # --------------------------------------------------------
    hybrid_labels = {
        "hybrid", "mixed", "mixed-origin", "mixed origin", "mixed code",
        "co-authored", "co authored", "ai-assisted", "ai assisted",
        "human-ai", "human ai", "human and ai",
    }
    if normalized in hybrid_labels:
        return ALL_CLASSES[2]

    # --------------------------------------------------------
    # Adversarial
    # --------------------------------------------------------
    adversarial_labels = {
        "adversarial", "adversarially disguised", "adversarial code",
        "obfuscated", "obfuscated code", "manipulated", "manipulated code",
        "deliberately modified", "deliberately modified code",
    }
    if normalized in adversarial_labels:
        return ALL_CLASSES[3]

    # --------------------------------------------------------
    # Controlled substring matching (only AFTER exact matching above)
    # --------------------------------------------------------
    if "adversarial" in normalized:
        return ALL_CLASSES[3]
    if "obfuscat" in normalized:
        return ALL_CLASSES[3]
    if "hybrid" in normalized:
        return ALL_CLASSES[2]
    if "co-authored" in normalized or "co authored" in normalized:
        return ALL_CLASSES[2]
    if "ai-assisted" in normalized or "ai assisted" in normalized:
        return ALL_CLASSES[2]
    if "machine-generated" in normalized or "machine generated" in normalized:
        return ALL_CLASSES[1]
    if "ai-generated" in normalized or "ai generated" in normalized:
        return ALL_CLASSES[1]
    if "human-written" in normalized or "human written" in normalized:
        return ALL_CLASSES[0]

    return None


def _accepts_kwargs(llm_client) -> bool:
    """MockLLMClient.decide_next_step() has a narrower signature than
    GeminiClientManager's; this lets run_agent_loop call either
    uniformly without inspecting types directly."""
    import inspect
    params = inspect.signature(llm_client.decide_next_step).parameters
    return "tool_descriptions" in params


# ============================================================
# MAIN AGENT LOOP
# ============================================================

def run_agent_loop(
    agent_name: str,
    toolset,
    tool_registry: Dict[str, Callable[[], Any]],
    tool_descriptions: Dict[str, str],
    system_prompt: str,
    llm_client,
    max_steps: int = MAX_AGENT_STEPS,
) -> AgentResult:
    steps_taken: List[Dict[str, Any]] = []
    tool_calls_made: List[str] = []

    decision = None
    for _ in range(max_steps):
        decision = llm_client.decide_next_step(
            agent_name=agent_name,
            available_tools=list(tool_registry.keys()),
            steps_taken=steps_taken,
            max_steps=max_steps,
            **(
                {"tool_descriptions": tool_descriptions, "system_prompt": system_prompt}
                if hasattr(llm_client, "decide_next_step") and _accepts_kwargs(llm_client)
                else {}
            ),
        )

        if decision.get("action") != "call_tool":
            break

        tool_name = decision.get("tool")
        if tool_name not in tool_registry:
            logger.warning("%s requested unknown tool '%s'; ignoring.", agent_name, tool_name)
            continue

        output = tool_registry[tool_name]()
        steps_taken.append({"tool": tool_name, "output": output})
        tool_calls_made.append(tool_name)

    if decision is None or decision.get("action") != "final_answer":
        decision = llm_client.decide_next_step(
            agent_name=agent_name,
            available_tools=list(tool_registry.keys()),
            steps_taken=steps_taken,
            max_steps=0,  # force final answer
            **(
                {"tool_descriptions": tool_descriptions, "system_prompt": system_prompt}
                if _accepts_kwargs(llm_client)
                else {}
            ),
        )

    raw_predicted_class = decision.get("predicted_class", None)
    confidence = float(decision.get("confidence", 0.0))
    evidence = decision.get("evidence", [])
    rationale = decision.get("rationale", "")

    # --------------------------------------------------------
    # NORMALIZE CLASS
    # --------------------------------------------------------
    predicted_class = _normalize_class_label(raw_predicted_class)

    logger.info(
        "%s raw class=%r -> normalized class=%r",
        agent_name, raw_predicted_class, predicted_class,
    )

    if predicted_class is None:
        logger.warning(
            "%s returned an unrecognized class '%s'.", agent_name, raw_predicted_class
        )
        return AgentResult(
            predicted_class=ALL_CLASSES[0],  # placeholder only; confidence=0.0 makes it neutral downstream
            confidence=0.0,
            evidence=[],
            rationale=(
                "Agent returned an unrecognized class label. "
                f"Raw label: {raw_predicted_class!r}"
            ),
            tool_calls_made=tool_calls_made,
            num_steps=len(steps_taken),
            failed=True,
        )

    # --------------------------------------------------------
    # Evidence validation
    # --------------------------------------------------------
    if not _validate_evidence(evidence, steps_taken):
        logger.warning("%s evidence failed validation; requesting one correction.", agent_name)
        corrected = llm_client.decide_next_step(
            agent_name=agent_name,
            available_tools=list(tool_registry.keys()),
            steps_taken=steps_taken + [{
                "tool": "_validation_error",
                "output": "Your previous evidence quotes were not exact substrings of "
                          "the tool outputs above. Reissue your final answer with "
                          "evidence quotes copied EXACTLY from a tool output.",
            }],
            max_steps=0,
            **({"tool_descriptions": tool_descriptions, "system_prompt": system_prompt} if _accepts_kwargs(llm_client) else {}),
        )

        evidence = corrected.get("evidence", [])
        corrected_raw_class = corrected.get("predicted_class", raw_predicted_class)
        corrected_class = _normalize_class_label(corrected_raw_class)
        if corrected_class is not None:
            predicted_class = corrected_class
        confidence = float(corrected.get("confidence", confidence))
        rationale = corrected.get("rationale", rationale)

        if not _validate_evidence(evidence, steps_taken):
            logger.error("%s evidence still failed validation after correction; marking failed.", agent_name)
            return AgentResult(
                predicted_class=predicted_class, confidence=0.0, evidence=[],
                rationale="Evidence could not be validated against tool outputs.",
                tool_calls_made=tool_calls_made, num_steps=len(steps_taken), failed=True,
            )

    return AgentResult(
        predicted_class=predicted_class,
        confidence=min(max(confidence, 0.0), 1.0),
        evidence=evidence,
        rationale=rationale,
        tool_calls_made=tool_calls_made,
        num_steps=len(steps_taken),
        failed=False,
    )