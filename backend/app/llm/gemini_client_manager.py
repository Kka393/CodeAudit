"""
llm/gemini_client_manager.py
==============================
The ONLY module that talks to Gemini. Agents never import
APIKeyManager or the google-genai SDK directly -- they only call
GeminiClientManager.decide_next_step(), exactly like they would call
MockLLMClient.decide_next_step() when USE_MOCK_LLM=true.

Architecture:

    Agent A / Agent B
          |
          v
    GeminiClientManager
          |
          v
    APIKeyManager  ->  google-genai client  ->  Gemini API

Responsible for: getting a key from APIKeyManager, creating the
Gemini client, making the request, reporting success/failure back to
APIKeyManager, and retrying through another available key on
rate-limit / quota / transient / auth failures (never hammering one
failed key).

We ask Gemini to respond with STRICT JSON (rather than relying on a
specific native function-calling schema shape, which drifts across
google-genai SDK versions) describing either the next tool call or
the agent's final verdict. This keeps the integration robust to SDK
version differences while still being a genuine per-step agentic
decision made by the model, not a scripted sequence.

IMPORTANT: the class names offered to Gemini in the prompt are built
from app.config.ALL_CLASSES (NOT hardcoded), so if CLASS_LABEL_ORDER
in .env is changed (e.g. to "Human,Machine,Hybrid,Adversarial"),
Gemini is asked for exactly those names -- it will never be told to
answer with a class name your config doesn't actually use.
"""

import json
import logging
import re
from typing import Any, Dict, List

from app.config import ALL_CLASSES, GEMINI_MAX_RETRIES_PER_REQUEST, GEMINI_MODEL
from app.llm.api_key_manager import APIKeyManager, NoAvailableKeyError

logger = logging.getLogger(__name__)


class GeminiRequestError(RuntimeError):
    pass


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _extract_json(text: str) -> Dict[str, Any]:
    match = _JSON_FENCE_RE.search(text)
    payload = match.group(1) if match else text
    return json.loads(payload)


def _build_prompt(
    agent_name: str,
    system_prompt: str,
    available_tools: List[str],
    tool_descriptions: Dict[str, str],
    steps_taken: List[Dict[str, Any]],
    max_steps: int,
) -> str:
    tools_block = "\n".join(f"- {name}: {tool_descriptions.get(name, '')}" for name in available_tools)
    history_block = json.dumps(steps_taken, default=str, indent=2) if steps_taken else "(none yet)"
    class_options = "|".join(ALL_CLASSES)

    return f"""{system_prompt}

You are the {agent_name}. You may call at most {max_steps} tools in total,
one per turn, before you MUST give a final answer.

Available tools:
{tools_block}

Tool calls made so far (with their exact outputs):
{history_block}

Respond with STRICT JSON only, no prose, no markdown fences, matching
exactly one of these two shapes:

To call a tool:
{{"action": "call_tool", "tool": "<one of the available tool names>"}}

To give your final answer (only once you have used your tool
budget, or you are confident no more tools are needed):
{{"action": "final_answer", "predicted_class": "{class_options}",
  "confidence": <float 0-1>,
  "evidence": [{{"quote": "<must be an exact substring of a tool output above>",
                 "reason": "<short reason>"}}],
  "rationale": "<short overall rationale>"}}

IMPORTANT: "predicted_class" MUST be exactly one of: {class_options}
(copy one of those strings verbatim -- do not use a synonym, a
different casing, or any other wording).
"""


class GeminiClientManager:
    def __init__(self, model: str = GEMINI_MODEL, key_manager: APIKeyManager = None):
        self.model = model
        self.key_manager = key_manager or APIKeyManager()

    def _call_once(self, key: str, prompt: str) -> str:
        from google import genai  # imported lazily so USE_MOCK_LLM=true never requires the SDK installed

        client = genai.Client(api_key=key)
        response = client.models.generate_content(model=self.model, contents=prompt)
        return response.text

    def decide_next_step(
        self,
        agent_name: str,
        available_tools: List[str],
        steps_taken: List[Dict[str, Any]],
        max_steps: int,
        tool_descriptions: Dict[str, str] = None,
        system_prompt: str = "",
    ) -> Dict[str, Any]:
        tool_descriptions = tool_descriptions or {}
        prompt = _build_prompt(
            agent_name, system_prompt, available_tools, tool_descriptions, steps_taken, max_steps
        )

        last_error = None
        for _attempt in range(GEMINI_MAX_RETRIES_PER_REQUEST):
            try:
                key = self.key_manager.get_next_key()
            except NoAvailableKeyError as exc:
                raise GeminiRequestError(str(exc)) from exc

            try:
                raw_text = self._call_once(key, prompt)
                self.key_manager.report_success(key)
                return _extract_json(raw_text)
            except json.JSONDecodeError as exc:
                # Malformed JSON is a model-quality problem, not a key
                # problem -- don't punish the key, just retry once.
                last_error = exc
                logger.warning("Gemini returned non-JSON output for %s: %s", agent_name, exc)
                continue
            except Exception as exc:  # noqa: BLE001 - broad on purpose: SDK errors vary
                error_text = str(exc).lower()
                is_key_problem = any(
                    token in error_text
                    for token in ("rate limit", "quota", "unavailable", "auth", "permission", "429", "503")
                )
                if is_key_problem:
                    self.key_manager.report_failure(key)
                    logger.warning("Gemini key failed for %s (%s); trying another key.", agent_name, exc)
                    last_error = exc
                    continue
                raise GeminiRequestError(f"Non-recoverable Gemini error: {exc}") from exc

        raise GeminiRequestError(
            f"Exhausted {GEMINI_MAX_RETRIES_PER_REQUEST} attempts calling Gemini for {agent_name}: {last_error}"
        )