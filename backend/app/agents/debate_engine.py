"""
agents/debate_engine.py
=========================
Production-ready multi-round LLM debate engine, extracted and cleaned
up from llm_debate_reorganized.py (a Colab notebook export).

ACTIVE MODE: run_multi_stream() — a generator that yields DebateEvent
objects as each LLM call (advocate / skeptic / judge) completes. This
is consumed by the /debate/stream SSE endpoint in api/debate.py.

SWAP MODE: run_single() is preserved as commented-out code at the
bottom. To switch to single-round debate, uncomment it and replace
the run_multi_stream call in api/debate.py with run_single.

Architecture:
    - Advocate (Groq) argues the code IS AI-generated
    - Skeptic (Gemini) argues the code IS human-written
    - Judge (Gemini) evaluates both arguments and renders verdict
    - Multi-round: up to MAX_ROUNDS rebuttals with convergence detection
    - All providers use OpenAI-compatible endpoints
"""

import json
import logging
import math
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Generator, Optional

try:
    from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI
except ImportError:
    APIConnectionError = APIStatusError = APITimeoutError = Exception
    OpenAI = None

from app.config import (
    DEBATE_ADVOCATE_MODEL,
    DEBATE_CONVERGENCE_GAP,
    DEBATE_GEMINI_API_KEY,
    DEBATE_GROQ_API_KEY,
    DEBATE_JUDGE_GEMINI_API_KEY,
    DEBATE_JUDGE_MODEL,
    DEBATE_MAX_ROUNDS,
    DEBATE_SKEPTIC_MODEL,
)

logger = logging.getLogger(__name__)


# ======================================================================
# Data classes
# ======================================================================

@dataclass
class DebateEvent:
    """Streamed event emitted per significant moment in the debate.
    Serialized to JSON and sent as an SSE data line."""
    event_type: str       # "round_start" | "advocate" | "skeptic" | "judge" | "done" | "error"
    round_num: Optional[int] = None
    role: Optional[str] = None
    content: str = ""
    confidence: Optional[int] = None
    gap: Optional[int] = None
    converged: Optional[bool] = None

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class DebateResult:
    """Final structured result of a completed debate."""
    verdict: str
    confidence: int
    reasoning: str
    rounds_run: int
    transcript: list
    debate_applied: bool = True


@dataclass
class _DebateState:
    """Internal state tracking across rounds."""
    code: str
    profile: dict
    ai_prob: float
    transcript: list = field(default_factory=list)
    round_num: int = 0

    def history_for(self) -> str:
        """Format the full transcript so an agent can see what has been argued."""
        if not self.transcript:
            return "No prior arguments yet -- this is the opening round."
        lines = []
        for entry in self.transcript:
            lines.append(
                f"[Round {entry['round']}] {entry['role'].upper()}: "
                f"{entry['content']}"
            )
        return "\n\n".join(lines)


# ======================================================================
# Prompts
# ======================================================================

_ADVOCATE_PROMPT = """You are a forensic code analyst arguing this snippet IS AI-generated.
Code:
{code}

Structural profile: {profile}
Classifier's own prediction: {ai_prob:.2%} probability of AI authorship.

Build the strongest evidence-based case for AI authorship. Cite specific lines,
formatting patterns, naming conventions, or structural metrics. Do not concede
uncertainty -- argue your position.

End your response with exactly this line, no other text after it:
Confidence: X/100"""

_SKEPTIC_PROMPT = """You are a forensic code analyst arguing this snippet IS human-written.
Code:
{code}

Structural profile: {profile}
Classifier's own prediction: {ai_prob:.2%} probability of AI authorship.

Build the strongest evidence-based case for human authorship -- messiness, personal
style quirks, unconventional choices a model would rarely produce. Cite specifics.

End your response with exactly this line, no other text after it:
Confidence: X/100"""

_REBUTTAL_ADVOCATE_PROMPT = """You are a forensic analyst arguing this code IS AI-generated.

Code:
{code}
Structural profile: {profile}

Debate so far:
{history}

The Skeptic just argued for human authorship. Directly rebut their strongest point
using specific evidence -- don't just restate your opening argument. If their point
is genuinely strong, you may lower your confidence rather than defend a weak position.
End with: Confidence: X/100"""

_REBUTTAL_SKEPTIC_PROMPT = """You are a forensic analyst arguing this code IS human-written.

Code:
{code}
Structural profile: {profile}

Debate so far:
{history}

The Advocate just argued for AI authorship. Directly rebut their strongest point
using specific evidence -- don't just restate your opening argument. If their point
is genuinely strong, you may lower your confidence rather than defend a weak position.
End with: Confidence: X/100"""

_JUDGE_PROMPT = """You are an impartial judge. Two analysts debated the authorship of this code.

Advocate (argues AI): {advocate_argument}

Skeptic (argues Human): {skeptic_argument}

Original classifier confidence: {ai_prob:.2%} AI.
Structural profile: {profile}

Weigh both arguments against the evidence. Do not simply average confidences --
evaluate which argument is better supported by the actual code.

Choose one of four verdicts:
- "AI": Evidence strongly and consistently indicates the code was generated by an LLM.
- "Human": Evidence strongly and consistently indicates the code was written by a person.
- "Human-AI Hybrid": Specific evidence points to mixed authorship -- for example, a
  human-written structure or logic with AI-generated boilerplate sections, or AI-generated
  code that shows signs of subsequent human editing (inconsistent style shifts, partial
  refactoring, mixed naming conventions across sections). Use this when different PARTS
  of the code point in different directions, not merely when you are uncertain overall.
- "Adversarial": The Advocate's and Skeptic's arguments are genuinely irreconcilable --
  both cite comparably strong, directly conflicting evidence, and the code itself does
  not contain enough signal to determine which interpretation is correct. This differs
  from Human-AI Hybrid: use Adversarial when the EVIDENCE is contradictory or ambiguous,
  not when authorship itself is mixed. Do not choose this just because the debate was
  close -- reserve it for genuine evidentiary deadlock.

Respond ONLY as JSON:
{{"verdict": "AI" or "Human" or "Human-AI Hybrid" or "Adversarial", "confidence": 0-100, "reasoning": "one sentence explaining which evidence was decisive"}}"""


# ======================================================================
# Helpers
# ======================================================================

def _extract_forensic_report(code: str) -> dict:
    """Extract structural signals injected into every agent's prompt."""
    lines = code.splitlines()
    total = max(len(lines), 1)
    blank = sum(1 for ln in lines if not ln.strip())
    r_void = blank / total

    chars = Counter(code)
    n = len(code) or 1
    entropy = -sum((c / n) * math.log2(c / n) for c in chars.values())

    identifiers = re.findall(r'\b[a-zA-Z_]\w*\b', code)
    avg_id_len = sum(len(i) for i in identifiers) / max(len(identifiers), 1)
    line_lengths = [len(ln) for ln in lines]

    return {
        "empty_line_ratio": round(r_void, 4),
        "char_entropy": round(entropy, 3),
        "avg_identifier_length": round(avg_id_len, 2),
        "avg_line_length": round(sum(line_lengths) / total, 2),
        "line_length_std": round(
            (sum((ll - sum(line_lengths) / total) ** 2 for ll in line_lengths) / total) ** 0.5,
            2,
        ),
    }


def _extract_confidence(text: str) -> int:
    """Extract confidence score from LLM response text."""
    if not text or not text.strip():
        logger.warning("extract_confidence received empty text")
        return 50

    patterns = [
        r"[Cc]onfidence[^\n\d]{0,40}?(\d{1,3})\s*(?:%|/\s*100|percent|\bout of 100\b)",
        r"[Cc]onfidence[^\n\d]{0,40}?(\d{1,3})\b",
        r"(\d{1,3})\s*(?:%|/\s*100)\s*(?:confiden\w*)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            val = int(matches[-1])
            if 0 <= val <= 100:
                return val

    all_matches = re.findall(r"(\d{1,3})\s*(?:%|/\s*100)", text)
    for val_str in reversed(all_matches):
        v = int(val_str)
        if 0 <= v <= 100:
            return v

    logger.warning("Could not extract confidence, defaulting to 50.")
    return 50


def _extract_json(raw: str) -> dict:
    """Extract and repair a JSON object from LLM output, tolerant of
    code fences, reasoning preambles, and common malformations."""
    # Strip <think>...</think> tags (some models emit these)
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)

    start = cleaned.find("{")
    if start == -1:
        raise json.JSONDecodeError("No JSON object found", cleaned, 0)
    candidate = cleaned[start:]

    # Try strict JSON first
    try:
        # Find matching closing brace
        depth = 0
        for i, ch in enumerate(candidate):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(candidate[:i + 1])
    except json.JSONDecodeError:
        pass

    # Fallback: try json_repair if available
    try:
        from json_repair import repair_json
        result = repair_json(candidate, return_objects=True)
        if isinstance(result, dict) and result:
            return result
    except ImportError:
        pass

    raise json.JSONDecodeError("No valid JSON object found", candidate, 0)


# ======================================================================
# LLM call wrapper
# ======================================================================

def _call_llm(
    client: OpenAI,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int = 1500,
    max_retries: int = 2,
    timeout: float = 30.0,
) -> str:
    """Single OpenAI-compatible call wrapper shared by every agent role."""
    resp = None
    for attempt in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
            if not resp or not getattr(resp, "choices", None):
                error_info = getattr(resp, "error", None) if resp else "no response"
                if attempt == max_retries:
                    logger.error(
                        "%s returned no choices after %d attempts. Error: %s",
                        model, max_retries + 1, error_info,
                    )
                    return ""
                wait = 5 * (attempt + 1)
                logger.warning(
                    "%s returned no choices (attempt %d/%d). Retrying in %ds...",
                    model, attempt + 1, max_retries + 1, wait,
                )
                time.sleep(wait)
                continue
            break
        except (APITimeoutError, APIConnectionError):
            if attempt == max_retries:
                logger.error("%s timed out after %d attempts.", model, max_retries + 1)
                return ""
            wait = 5 * (attempt + 1)
            logger.warning(
                "%s timed out (attempt %d/%d). Retrying in %ds...",
                model, attempt + 1, max_retries + 1, wait,
            )
            time.sleep(wait)
        except APIStatusError as e:
            if "temperature" in str(e).lower():
                logger.warning("%s rejected temperature param, retrying without.", model)
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    timeout=timeout,
                )
                break
            raise

    if not resp or not getattr(resp, "choices", None):
        logger.error("%s gave up with no valid response.", model)
        return ""

    message = resp.choices[0].message
    content = (message.content or "").strip()

    if not content:
        reasoning = getattr(message, "reasoning", None) or \
                    getattr(message, "reasoning_content", None)
        if reasoning:
            logger.warning(
                "%s returned empty content; using reasoning field.", model,
            )
            return reasoning.strip()
        logger.warning("%s returned a completely empty response.", model)

    return content


# ======================================================================
# DebateEngine
# ======================================================================

class DebateEngine:
    """Multi-round LLM debate engine using three independent providers."""

    def __init__(
        self,
        groq_key: str = DEBATE_GROQ_API_KEY,
        gemini_key: str = DEBATE_GEMINI_API_KEY,
        judge_gemini_key: str = DEBATE_JUDGE_GEMINI_API_KEY,
    ):
        if not groq_key or not gemini_key or not judge_gemini_key:
            missing = []
            if not groq_key:
                missing.append("GROQ_ADVOCATE_API_KEY")
            if not gemini_key:
                missing.append("GEMINI_SKEPTIC_API_KEY")
            if not judge_gemini_key:
                missing.append("GEMINI_JUDGE_API_KEY (or OPENROUTER_JUDGE_API_KEY)")
            raise ValueError(
                f"Debate engine requires API keys for all three providers. "
                f"Missing: {', '.join(missing)}. Set them in .env."
            )

        self._clients = {
            "groq": OpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key=groq_key,
                timeout=60.0,
            ),
            "gemini": OpenAI(
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                api_key=gemini_key,
                timeout=60.0,
            ),
            # Judge now uses Gemini instead of OpenRouter
            "gemini_judge": OpenAI(
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                api_key=judge_gemini_key,
                timeout=60.0,
            ),
        }

        self._models = {
            "groq":         DEBATE_ADVOCATE_MODEL,
            "gemini":       DEBATE_SKEPTIC_MODEL,
            "gemini_judge": DEBATE_JUDGE_MODEL,
        }

        logger.info(
            "DebateEngine initialized | advocate=%s | skeptic=%s | judge=%s",
            self._models["groq"], self._models["gemini"], self._models["gemini_judge"],
        )

    def _call(self, provider: str, prompt: str, temperature: float) -> str:
        return _call_llm(
            self._clients[provider],
            self._models[provider],
            prompt,
            temperature,
        )

    # ------------------------------------------------------------------
    # ACTIVE MODE: Multi-round streaming generator
    # ------------------------------------------------------------------
    def run_multi_stream(
        self,
        code: str,
        ai_prob: float,
        max_rounds: int = DEBATE_MAX_ROUNDS,
        convergence_gap: int = DEBATE_CONVERGENCE_GAP,
    ) -> Generator[DebateEvent, None, None]:
        """Yields a DebateEvent for each significant moment:
          - Round N starts             -> yield round_start
          - Advocate argument done     -> yield advocate event
          - Skeptic argument done      -> yield skeptic event
          - Convergence detected       -> yield round_start with converged=True
          - Judge verdict returned     -> yield judge event
          - All done                   -> yield done event
        """
        profile = _extract_forensic_report(code)
        state = _DebateState(code=code, profile=profile, ai_prob=ai_prob)

        for round_num in range(max_rounds):
            state.round_num = round_num
            yield DebateEvent(
                event_type="round_start",
                round_num=round_num,
                content=f"Starting round {round_num}",
            )

            # --- Build prompts ---
            if round_num == 0:
                adv_prompt = _ADVOCATE_PROMPT.format(
                    code=code, profile=profile, ai_prob=ai_prob,
                )
                skep_prompt = _SKEPTIC_PROMPT.format(
                    code=code, profile=profile, ai_prob=ai_prob,
                )
            else:
                adv_prompt = _REBUTTAL_ADVOCATE_PROMPT.format(
                    code=code, profile=profile, history=state.history_for(),
                )
                skep_prompt = _REBUTTAL_SKEPTIC_PROMPT.format(
                    code=code, profile=profile, history=state.history_for(),
                )

            # --- Advocate call (Groq) ---
            adv_response = self._call("groq", adv_prompt, temperature=0.7)
            adv_conf = _extract_confidence(adv_response)
            state.transcript.append({
                "role": "advocate", "round": round_num,
                "content": adv_response, "confidence": adv_conf,
            })
            yield DebateEvent(
                event_type="advocate",
                round_num=round_num,
                role="advocate",
                content=adv_response,
                confidence=adv_conf,
            )

            # --- Skeptic call (Gemini) ---
            skep_response = self._call("gemini", skep_prompt, temperature=0.7)
            skep_conf = _extract_confidence(skep_response)
            state.transcript.append({
                "role": "skeptic", "round": round_num,
                "content": skep_response, "confidence": skep_conf,
            })
            gap = abs(adv_conf - (100 - skep_conf))
            yield DebateEvent(
                event_type="skeptic",
                round_num=round_num,
                role="skeptic",
                content=skep_response,
                confidence=skep_conf,
                gap=gap,
            )

            # --- Check convergence ---
            if gap <= convergence_gap:
                logger.info(
                    "Debate converged after round %d (gap=%d <= %d)",
                    round_num, gap, convergence_gap,
                )
                yield DebateEvent(
                    event_type="round_start",
                    round_num=round_num,
                    content="Converged",
                    converged=True,
                )
                break

        # --- Judge call (Gemini) ---
        full_transcript = state.history_for()
        judge_prompt = _JUDGE_PROMPT.format(
            advocate_argument=full_transcript,
            skeptic_argument="(see above, interleaved)",
            ai_prob=ai_prob,
            profile=profile,
        )

        judge_raw = self._call("gemini_judge", judge_prompt, temperature=0.0)

        try:
            verdict = _extract_json(judge_raw)
        except json.JSONDecodeError:
            logger.warning("Judge JSON parse failed. Raw: %s", judge_raw[:500])
            verdict = {
                "verdict": "AI" if ai_prob > 0.5 else "Human",
                "confidence": 50,
                "reasoning": "Judge parse failure, defaulted to classifier",
            }

        yield DebateEvent(
            event_type="judge",
            role="judge",
            content=json.dumps(verdict),
            confidence=verdict.get("confidence"),
        )

        yield DebateEvent(
            event_type="done",
            content=json.dumps({
                "verdict": verdict.get("verdict", "Unknown"),
                "confidence": verdict.get("confidence", 50),
                "reasoning": verdict.get("reasoning", ""),
                "rounds_run": state.round_num + 1,
            }),
            confidence=verdict.get("confidence"),
        )

    # ------------------------------------------------------------------
    # SWAP: Single-round debate (commented out -- uncomment to use)
    # ------------------------------------------------------------------
    # def run_single(self, code: str, ai_prob: float) -> DebateResult:
    #     """SWAP: uncomment this and replace run_multi_stream calls with
    #     run_single to switch to single-round mode (no streaming).
    #
    #     Returns a DebateResult directly instead of streaming events.
    #     """
    #     profile = _extract_forensic_report(code)
    #
    #     adv_arg = self._call(
    #         "groq",
    #         _ADVOCATE_PROMPT.format(code=code, profile=profile, ai_prob=ai_prob),
    #         temperature=0.7,
    #     )
    #     skep_arg = self._call(
    #         "gemini",
    #         _SKEPTIC_PROMPT.format(code=code, profile=profile, ai_prob=ai_prob),
    #         temperature=0.7,
    #     )
    #     judge_raw = self._call(
    #         "gemini_judge",
    #         _JUDGE_PROMPT.format(
    #             advocate_argument=adv_arg,
    #             skeptic_argument=skep_arg,
    #             ai_prob=ai_prob,
    #             profile=profile,
    #         ),
    #         temperature=0.0,
    #     )
    #
    #     try:
    #         verdict = _extract_json(judge_raw)
    #     except json.JSONDecodeError:
    #         verdict = {
    #             "verdict": "AI" if ai_prob > 0.5 else "Human",
    #             "confidence": 50,
    #             "reasoning": "Judge parse failure",
    #         }
    #
    #     return DebateResult(
    #         verdict=verdict.get("verdict", "Unknown"),
    #         confidence=verdict.get("confidence", 50),
    #         reasoning=verdict.get("reasoning", ""),
    #         rounds_run=1,
    #         transcript=[
    #             {"role": "advocate", "round": 0, "content": adv_arg},
    #             {"role": "skeptic", "round": 0, "content": skep_arg},
    #         ],
    #     )
