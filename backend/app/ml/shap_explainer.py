r"""
ml/shap_explainer.py
======================
Source-code line-level attribution engine for the existing 4-class classifier
(Human, Machine, Hybrid, Adversarial).

TECHNICAL ATTRIBUTION METHOD:
-----------------------------
Measures counterfactual impact of individual source-code lines via line-level
occlusion/perturbation through the complete existing multi-view pipeline:

    Original Code
          ↓
    split_code() -> (main_code, sub_functions, comments_view)
          ↓
    ModernBERT masked-mean pooling -> (e_main, e_sub, e_comments)
          ↓
    AICD4ClassClassifier -> P_baseline(target_class)

For each non-empty source-code line i:
    1. Mask/neutralize line i to produce perturbed code S_{\i}.
    2. Pass S_{\i} through the exact same multi-view pipeline.
    3. Measure target class probability P_{\i}(target_class).
    4. Compute attribution score:
           contribution = P_baseline - P_{\i}

INTERPRETATION:
    - contribution > 0 (Positive / ↑):
          Removing this line causes target probability to drop.
          Therefore, this line pushed TOWARDS the model's prediction.
    - contribution < 0 (Negative / ↓):
          Removing this line causes target probability to rise.
          Therefore, this line pushed AGAINST the model's prediction.
    - contribution ≈ 0 (Neutral):
          Line has negligible influence on the model's decision.

IMPORTANT CONSTRAINTS:
    - Targets the 4-class classifier (NOT the LLM-family classifier).
    - Explains whatever class the model predicted (Human, Machine, Hybrid, Adversarial).
    - Does not alter model weights or create fabricated token-to-embedding mappings.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

from app.config import ALL_CLASSES, DEVICE
from app.ml.code_splitter import split_code
from app.ml.feature_extractor import ViewEmbeddings, masked_mean_pooling

logger = logging.getLogger(__name__)

# Attribution significance threshold: absolute delta must be >= 0.005 (0.5%)
SIGNIFICANCE_THRESHOLD = 0.005


@dataclass
class LineAttribution:
    line_number: int            # 1-indexed
    content: str                # Raw line string
    attribution: float          # Signed attribution score (+0.18, -0.04)
    direction: str              # "positive" | "negative" | "neutral"
    is_significant: bool        # True if |attribution| >= SIGNIFICANCE_THRESHOLD


@dataclass
class CodeExplanation:
    target_class: str
    baseline_confidence: float
    lines: List[LineAttribution]
    summary: Dict[str, Any]
    method: str = "counterfactual_line_occlusion"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_class": self.target_class,
            "baseline_confidence": round(self.baseline_confidence, 4),
            "method": self.method,
            "lines": [
                {
                    "line_number": item.line_number,
                    "content": item.content,
                    "attribution": round(item.attribution, 4),
                    "direction": item.direction,
                    "is_significant": item.is_significant,
                }
                for item in self.lines
            ],
            "summary": self.summary,
        }


def _mask_line(lines: List[str], line_idx: int, language: str) -> str:
    """Create a perturbed version of the code with line_idx masked.
    Replaces the line with a syntactically benign comment to preserve
    line numbering and indentation structure.
    """
    perturbed = list(lines)
    orig_line = perturbed[line_idx]
    
    # Calculate leading whitespace indentation
    indent_match = re.match(r"^(\s*)", orig_line)
    indent = indent_match.group(1) if indent_match else ""

    # Benign placeholder comment depending on language
    lang_lc = (language or "").lower()
    if lang_lc in ("python", "py"):
        # If the line ends with a colon, replacing with a bare comment could
        # break the following block; add 'pass' to preserve syntax where possible.
        stripped = orig_line.strip()
        if stripped.endswith(":"):
            perturbed[line_idx] = f"{indent}# [omitted block head]"
        else:
            perturbed[line_idx] = f"{indent}# [omitted]"
    else:
        perturbed[line_idx] = f"{indent}// [omitted]"

    return "\n".join(perturbed)


class CodeLineExplainer:
    """Computes source-code line-level attribution for the 4-class classifier."""

    def __init__(self, feature_extractor, classifier):
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.encoder = feature_extractor.encoder

    def explain(
        self,
        code: str,
        language: str = "python",
        target_class: Optional[str] = None,
        max_lines_to_eval: int = 150,
    ) -> Optional[CodeExplanation]:
        """Generate line-level attributions for the 4-class classifier.

        Args:
            code: Source code string to explain.
            language: Programming language name.
            target_class: Target class to explain. If None, uses the classifier's
                          top predicted class.
            max_lines_to_eval: Maximum non-empty lines to perturb (prevents timeout
                               on huge files).

        Returns:
            CodeExplanation instance with line-by-line attribution.
        """
        if not code or not code.strip():
            logger.warning("CodeLineExplainer received empty code.")
            return None

        raw_lines = code.split("\n")
        total_lines = len(raw_lines)

        # ── 1. Baseline prediction ──────────────────────────────────
        try:
            baseline_views = split_code(code, language=language)
            baseline_emb = self.feature_extractor.extract(baseline_views)
            baseline_pred = self.classifier.predict(baseline_emb)
        except Exception as exc:
            logger.exception("Failed to compute baseline prediction for explanation: %s", exc)
            return None

        if target_class is None:
            target_class = baseline_pred.predicted_class

        # Normalize target class key for probability lookup
        base_probs = baseline_pred.probabilities
        target_key = target_class
        if target_key not in base_probs:
            # Check case-insensitive match
            for k in base_probs:
                if k.lower() == target_class.lower():
                    target_key = k
                    break

        baseline_prob = float(base_probs.get(target_key, baseline_pred.confidence))

        # ── 2. Identify candidate lines to evaluate ──────────────────
        # Lines with only whitespace have 0 attribution by definition.
        eval_indices = []
        for i, line in enumerate(raw_lines):
            if line.strip():
                eval_indices.append(i)

        if len(eval_indices) > max_lines_to_eval:
            logger.info(
                "Code has %d non-empty lines; capping evaluation to first %d.",
                len(eval_indices), max_lines_to_eval,
            )
            eval_indices = eval_indices[:max_lines_to_eval]

        # ── 3. Perturbation & Batch Evaluation ───────────────────────
        perturbed_probs: Dict[int, float] = {}

        # Batch perturbed inputs through ModernBERT to maximize throughput
        batch_size = 16
        for start_idx in range(0, len(eval_indices), batch_size):
            chunk_indices = eval_indices[start_idx : start_idx + batch_size]
            
            chunk_views = []
            valid_indices = []
            for line_idx in chunk_indices:
                try:
                    p_code = _mask_line(raw_lines, line_idx, language)
                    views = split_code(p_code, language=language)
                    chunk_views.append(views)
                    valid_indices.append(line_idx)
                except Exception as exc:
                    logger.debug("Failed to create perturbed view for line %d: %s", line_idx + 1, exc)
                    perturbed_probs[line_idx] = baseline_prob

            if not chunk_views:
                continue

            # Batch encode all views in this chunk
            try:
                # 3 views per variant: main, sub, comments
                all_main = [v.main_code for v in chunk_views]
                all_sub = [v.sub_functions for v in chunk_views]
                all_comments = [v.comments_view for v in chunk_views]

                # Run encoder on batched texts
                lhs_main, mask_main = self.encoder.encode(all_main)
                lhs_sub, mask_sub = self.encoder.encode(all_sub)
                lhs_comm, mask_comm = self.encoder.encode(all_comments)

                e_main_batch = masked_mean_pooling(lhs_main, mask_main)
                e_sub_batch = masked_mean_pooling(lhs_sub, mask_sub)
                e_comm_batch = masked_mean_pooling(lhs_comm, mask_comm)

                # Forward pass through AICD4ClassClassifier
                with torch.no_grad():
                    e_m = e_main_batch.float().to(self.classifier.device)
                    e_s = e_sub_batch.float().to(self.classifier.device)
                    e_c = e_comm_batch.float().to(self.classifier.device)
                    logits = self.classifier.model(e_m, e_s, e_c)
                    probs_batch = torch.softmax(logits, dim=-1).cpu().tolist()

                # Map back probabilities for target class
                labels = self.classifier.labels
                target_idx = labels.index(target_key) if target_key in labels else 0

                for i, line_idx in enumerate(valid_indices):
                    perturbed_probs[line_idx] = float(probs_batch[i][target_idx])

            except Exception as exc:
                logger.warning("Batched perturbation failed (%s); falling back to sequential evaluation.", exc)
                for line_idx in valid_indices:
                    try:
                        p_code = _mask_line(raw_lines, line_idx, language)
                        v = split_code(p_code, language=language)
                        emb = self.feature_extractor.extract(v)
                        pred = self.classifier.predict(emb)
                        perturbed_probs[line_idx] = float(pred.probabilities.get(target_key, baseline_prob))
                    except Exception as seq_exc:
                        logger.debug("Sequential line %d failed: %s", line_idx + 1, seq_exc)
                        perturbed_probs[line_idx] = baseline_prob

        # ── 4. Build LineAttribution list ────────────────────────────
        line_attributions: List[LineAttribution] = []
        top_positives: List[Dict[str, Any]] = []
        top_negatives: List[Dict[str, Any]] = []

        for i, line_text in enumerate(raw_lines):
            line_no = i + 1

            if i in perturbed_probs:
                p_perturbed = perturbed_probs[i]
                # Contribution = original probability - perturbed probability
                # If removing the line drops target probability, attribution > 0 (positive).
                attribution = baseline_prob - p_perturbed
            else:
                attribution = 0.0

            if attribution >= SIGNIFICANCE_THRESHOLD:
                direction = "positive"
                is_sig = True
                top_positives.append({"line": line_no, "score": attribution, "content": line_text.strip()})
            elif attribution <= -SIGNIFICANCE_THRESHOLD:
                direction = "negative"
                is_sig = True
                top_negatives.append({"line": line_no, "score": attribution, "content": line_text.strip()})
            else:
                direction = "neutral"
                is_sig = False

            line_attributions.append(
                LineAttribution(
                    line_number=line_no,
                    content=line_text,
                    attribution=attribution,
                    direction=direction,
                    is_significant=is_sig,
                )
            )

        top_positives.sort(key=lambda x: x["score"], reverse=True)
        top_negatives.sort(key=lambda x: x["score"])

        summary = {
            "total_lines": total_lines,
            "evaluated_lines": len(eval_indices),
            "positive_contributing_lines": [item["line"] for item in top_positives],
            "negative_contributing_lines": [item["line"] for item in top_negatives],
            "max_positive_attribution": round(top_positives[0]["score"], 4) if top_positives else 0.0,
            "max_negative_attribution": round(top_negatives[0]["score"], 4) if top_negatives else 0.0,
            "top_positive_snippet": top_positives[0]["content"] if top_positives else None,
            "top_negative_snippet": top_negatives[0]["content"] if top_negatives else None,
        }

        return CodeExplanation(
            target_class=target_class,
            baseline_confidence=baseline_prob,
            lines=line_attributions,
            summary=summary,
        )
