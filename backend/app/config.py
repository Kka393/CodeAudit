"""
config.py
=========
Single source of truth for every tunable parameter used by the
AICD_MultiView application (model paths, agent limits, aggregation
weights, confidence threshold, Gemini settings).

Values are read from environment variables (populated from a local
.env file via python-dotenv) with safe, explicit defaults. Nothing
else in this project should hardcode these values.
"""

import os

from dotenv import load_dotenv

load_dotenv()


def _get_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _get_float(name: str, default: float) -> float:
    val = os.environ.get(name)
    if val is None or val.strip() == "":
        return default
    return float(val)


def _get_int(name: str, default: int) -> int:
    val = os.environ.get(name)
    if val is None or val.strip() == "":
        return default
    return int(val)


# ------------------------------------------------------------------
# Model paths (YOU place the actual files here; nothing is bundled)
# ------------------------------------------------------------------
# T2-trained ModernBERT checkpoint (the file you called
# "model_complete.pth"). This is the SHARED, FROZEN encoder used for
# all three views (main_code / sub_functions / comments_view).
MODERNBERT_MODEL_PATH = os.environ.get(
    "MODERNBERT_MODEL_PATH",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "models", "model_complete.pth"),
)

# Base HF checkpoint used ONLY to reconstruct the ModernBERT
# architecture under the currently-installed transformers version.
# The actual weights loaded into it come from MODERNBERT_MODEL_PATH,
# not from this pretrained checkpoint's own weights.
MODERNBERT_BASE_ARCHITECTURE = os.environ.get(
    "MODERNBERT_BASE_ARCHITECTURE", "answerdotai/ModernBERT-base"
)

# Trained 4-class classifier checkpoint (projections + MLP head).
# This is the ONLY classifier checkpoint the app will load. Do not
# point this at checkpoint_epoch_*.pt or stage6_classifier_head.pth.
CLASSIFIER_CHECKPOINT_PATH = os.environ.get(
    "CLASSIFIER_CHECKPOINT_PATH",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "models", "best_4class_classifier.pt"),
)

MAX_LENGTH = _get_int("MAX_LENGTH", 512)

import torch  # noqa: E402

_env_device = os.environ.get("AICD_DEVICE")
DEVICE = torch.device(_env_device) if _env_device else torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

EMPTY_VIEW_PLACEHOLDER = "[NONE]"
FUNCTION_JOIN_SEPARATOR = "\n\n"

RANDOM_SEED = 42


# ------------------------------------------------------------------
# Label mapping
# ------------------------------------------------------------------
# IMPORTANT: the training script that produced best_4class_classifier.pt
# saved its "classes" field as the generic placeholders
# ["Class_0", "Class_1", "Class_2", "Class_3"] -- it does NOT encode
# the semantic Human/AI/Hybrid/Adversarial mapping. classifier.py
# inspects the checkpoint at load time; if it only finds the generic
# placeholders, it falls back to CLASS_LABELS below AND logs a loud
# warning that the mapping is ASSUMED, not verified from the
# checkpoint, exactly per the "never silently guess" requirement.
#
# If you know the true index->class mapping used during T3 training
# (check your training notebook / dataset label encoder), set it here
# or via the CLASS_LABEL_ORDER env var (comma-separated, index order),
# e.g. CLASS_LABEL_ORDER=Human,AI,Hybrid,Adversarial
_default_order = ["Human", "Machine", "Hybrid", "Adversarial"]
_env_order = os.environ.get("CLASS_LABEL_ORDER", "").strip()
CLASS_LABELS = (
    [c.strip() for c in _env_order.split(",")] if _env_order else _default_order
)
LABEL_MAPPING_VERIFIED = bool(_env_order)  # True only if the user explicitly set it

ALL_CLASSES = list(CLASS_LABELS)  # convenience alias used across the app


# ------------------------------------------------------------------
# Agents
# ------------------------------------------------------------------
MAX_AGENT_STEPS = _get_int("MAX_AGENT_STEPS", 2)


# ------------------------------------------------------------------
# Gemini / LLM
# ------------------------------------------------------------------
USE_MOCK_LLM = _get_bool("USE_MOCK_LLM", True)
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")

GEMINI_API_KEYS = [
    (os.environ.get(f"API_KEY_{i}") or os.environ.get(f"GEMINI_API_KEY_{i}", "")).strip()
    for i in range(1, 6)
]
GEMINI_API_KEYS = [k for k in GEMINI_API_KEYS if k]  # drop empty slots

KEY_COOLDOWN_SECONDS = _get_int("KEY_COOLDOWN_SECONDS", 60)
GEMINI_MAX_RETRIES_PER_REQUEST = _get_int("GEMINI_MAX_RETRIES_PER_REQUEST", 5)


# ------------------------------------------------------------------
# Aggregation
# ------------------------------------------------------------------
AGGREGATOR_WEIGHTS = {
    "ml": _get_float("WEIGHT_ML", 0.50),
    "agent_a": _get_float("WEIGHT_AGENT_A", 0.25),
    "agent_b": _get_float("WEIGHT_AGENT_B", 0.25),
}

_weight_sum = sum(AGGREGATOR_WEIGHTS.values())
if abs(_weight_sum - 1.0) > 1e-6:
    raise ValueError(
        f"AGGREGATOR_WEIGHTS must sum to 1.0, got {_weight_sum} "
        f"({AGGREGATOR_WEIGHTS!r}). Fix WEIGHT_ML / WEIGHT_AGENT_A / "
        f"WEIGHT_AGENT_B in your .env."
    )

CONFIDENCE_THRESHOLD = _get_float("CONFIDENCE_THRESHOLD", 0.70)


# ------------------------------------------------------------------
# Supported languages
# ------------------------------------------------------------------
SUPPORTED_LANGUAGES = [
    "python", "java", "csharp", "cpp", "javascript", "go", "c", "php",
]


# ------------------------------------------------------------------
# Family classifier (Stage 6 head, T2-trained)
# ------------------------------------------------------------------
# The LLM-family classifier head trained on the T2 dataset.
# Architecture: Linear(768→256) + BatchNorm1d(256) + two heads:
#   binary_head: Linear(256→2) — binary Human/AI (NOT used; we use the 4-class)
#   family_head: Linear(256→12) — 12 LLM family classes
# We only use family_head, and only when the 4-class classifier predicts AI/Machine.
FAMILY_CLASSIFIER_PATH = os.environ.get(
    "FAMILY_CLASSIFIER_PATH",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "models", "stage6_classifier_head.pth"),
)

# 12 LLM family labels — set via FAMILY_CLASS_LABELS in .env once you
# confirm the true order from your T2 training notebook's label encoder.
# Default is a best-guess based on the AICD-T2 dataset's known model families.
_default_family_labels = [
    "GPT-4", "GPT-3.5", "Claude", "Gemini", "LLaMA", "CodeLlama",
    "Mistral", "Copilot", "DeepSeek", "Qwen", "StarCoder", "Other",
]
_env_family_labels = os.environ.get("FAMILY_CLASS_LABELS", "").strip()
FAMILY_CLASS_LABELS = (
    [c.strip() for c in _env_family_labels.split(",")]
    if _env_family_labels else _default_family_labels
)
FAMILY_LABELS_VERIFIED = bool(_env_family_labels)

# If the family classifier's max probability is below this threshold,
# the prediction is reported as "Unknown/Other" rather than a specific family.
FAMILY_UNKNOWN_THRESHOLD = _get_float("FAMILY_UNKNOWN_THRESHOLD", 0.35)


# ------------------------------------------------------------------
# Debate (LLM debate engine — Phase B)
# ------------------------------------------------------------------
# Three separate API keys for the three debate roles. These use
# OpenAI-compatible endpoints (Groq, Gemini, OpenRouter).
DEBATE_GROQ_API_KEY = os.environ.get("GROQ_ADVOCATE_API_KEY", "").strip()
DEBATE_GEMINI_API_KEY = os.environ.get("GEMINI_SKEPTIC_API_KEY", "").strip()
# Judge now uses Gemini — reads GEMINI_JUDGE_API_KEY, falls back to
# OPENROUTER_JUDGE_API_KEY so existing .env files keep working.
DEBATE_JUDGE_GEMINI_API_KEY = (
    os.environ.get("GEMINI_JUDGE_API_KEY", "").strip()
    or os.environ.get("OPENROUTER_JUDGE_API_KEY", "").strip()
)

# Model names for each debate provider — configurable via environment.
DEBATE_ADVOCATE_MODEL = os.environ.get("DEBATE_ADVOCATE_MODEL", "openai/gpt-oss-20b")
DEBATE_SKEPTIC_MODEL  = os.environ.get("DEBATE_SKEPTIC_MODEL",  "gemini-3.5-flash-lite")
DEBATE_JUDGE_MODEL    = os.environ.get("DEBATE_JUDGE_MODEL",    "gemini-3.5-flash-lite")

DEBATE_MAX_ROUNDS = _get_int("DEBATE_MAX_ROUNDS", 3)
DEBATE_CONVERGENCE_GAP = _get_int("DEBATE_CONVERGENCE_GAP", 15)

