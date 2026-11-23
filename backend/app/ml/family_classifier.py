"""
ml/family_classifier.py
=========================
Loads the T2-trained Stage 6 classifier head checkpoint
(config.FAMILY_CLASSIFIER_PATH, i.e. "stage6_classifier_head.pth")
and runs inference on a SINGLE 768-d embedding to predict which
LLM family generated the code.

ARCHITECTURE (discovered by inspect_checkpoints.py):

    The checkpoint is a raw OrderedDict state_dict with 11 tensors:

    feature_proj.0  → Linear(768, 256)    # shared projection
    feature_proj.1  → BatchNorm1d(256)    # shared batch norm
    binary_head     → Linear(256, 2)      # binary Human/AI (NOT used)
    family_head     → Linear(256, 12)     # 12-class LLM family

    We reconstruct this as an nn.Module, load the state_dict, and
    only use family_head for inference. binary_head is loaded (so
    strict=True passes) but never called.

ACTIVATION RULE:
    This classifier is ONLY called when the 4-class classifier
    (ml/classifier.py) predicts "AI" or "Machine". For Human, Hybrid,
    or Adversarial predictions, family_prediction is null.

LABEL MAPPING:
    The checkpoint stores NO class labels. FAMILY_CLASS_LABELS in
    config.py provides the assumed order (configurable via
    FAMILY_CLASS_LABELS env var). A loud warning is logged if the
    labels are not explicitly verified (FAMILY_LABELS_VERIFIED=False).
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn

from app.config import (
    DEVICE,
    FAMILY_CLASS_LABELS,
    FAMILY_CLASSIFIER_PATH,
    FAMILY_LABELS_VERIFIED,
    FAMILY_UNKNOWN_THRESHOLD,
)

logger = logging.getLogger(__name__)


class FamilyClassifierLoadError(RuntimeError):
    """Raised when stage6_classifier_head.pth cannot be loaded.
    Never caught to fall back to random weights."""


class Stage6ClassifierHead(nn.Module):
    """Architecture exactly matching the training code that produced
    stage6_classifier_head.pth. Discovered via inspect_checkpoints.py:

        feature_proj = Sequential(Linear(768, 256), BatchNorm1d(256))
        binary_head  = Linear(256, 2)
        family_head  = Linear(256, 12)

    binary_head is loaded for state_dict compatibility but never used.
    """

    def __init__(self, input_dim: int = 768, proj_dim: int = 256,
                 num_binary: int = 2, num_families: int = 12):
        super().__init__()
        self.feature_proj = nn.Sequential(
            nn.Linear(input_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
        )
        self.binary_head = nn.Linear(proj_dim, num_binary)
        self.family_head = nn.Linear(proj_dim, num_families)

    def forward_family(self, embedding: torch.Tensor) -> torch.Tensor:
        """Run through shared projection + family head only.
        Input: (batch, 768)  →  Output: (batch, 12) raw logits"""
        projected = self.feature_proj(embedding)
        return self.family_head(projected)


@dataclass
class FamilyPrediction:
    predicted_family: str
    confidence: float
    probabilities: Dict[str, float]
    is_unknown: bool  # True if confidence < FAMILY_UNKNOWN_THRESHOLD


class FamilyClassifier:
    """Loads stage6_classifier_head.pth and predicts LLM family."""

    def __init__(
        self,
        checkpoint_path: str = FAMILY_CLASSIFIER_PATH,
        device: torch.device = DEVICE,
        unknown_threshold: float = FAMILY_UNKNOWN_THRESHOLD,
    ):
        self.device = device
        self.unknown_threshold = unknown_threshold

        # --- Load the raw state_dict ---
        if not checkpoint_path:
            raise FamilyClassifierLoadError(
                "FAMILY_CLASSIFIER_PATH is not set. Place "
                "stage6_classifier_head.pth in models/ or set the env var."
            )

        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        if not isinstance(state_dict, dict):
            raise FamilyClassifierLoadError(
                f"Expected a state_dict (OrderedDict), got {type(state_dict)}. "
                f"This does not look like stage6_classifier_head.pth."
            )

        # --- Infer architecture from tensor shapes ---
        proj_weight = state_dict.get("feature_proj.0.weight")
        family_weight = state_dict.get("family_head.weight")
        binary_weight = state_dict.get("binary_head.weight")

        if proj_weight is None or family_weight is None:
            raise FamilyClassifierLoadError(
                f"Expected keys 'feature_proj.0.weight' and 'family_head.weight' "
                f"in checkpoint. Found keys: {list(state_dict.keys())}. "
                f"This does not look like stage6_classifier_head.pth."
            )

        input_dim = proj_weight.shape[1]   # 768
        proj_dim = proj_weight.shape[0]    # 256
        num_families = family_weight.shape[0]  # 12
        num_binary = binary_weight.shape[0] if binary_weight is not None else 2

        logger.info(
            "Family classifier architecture inferred: input=%d, proj=%d, "
            "families=%d, binary=%d",
            input_dim, proj_dim, num_families, num_binary,
        )

        # --- Construct model and load weights ---
        self.model = Stage6ClassifierHead(
            input_dim=input_dim,
            proj_dim=proj_dim,
            num_binary=num_binary,
            num_families=num_families,
        )

        try:
            self.model.load_state_dict(state_dict, strict=True)
        except RuntimeError as exc:
            raise FamilyClassifierLoadError(
                f"stage6_classifier_head.pth state_dict does not match "
                f"Stage6ClassifierHead architecture. Original error: {exc}"
            ) from exc

        self.model.to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        # --- Resolve labels ---
        self.labels = list(FAMILY_CLASS_LABELS)
        if len(self.labels) != num_families:
            raise FamilyClassifierLoadError(
                f"FAMILY_CLASS_LABELS has {len(self.labels)} entries but "
                f"family_head has {num_families} outputs. Fix "
                f"FAMILY_CLASS_LABELS in .env."
            )

        if not FAMILY_LABELS_VERIFIED:
            logger.warning(
                "=" * 78 + "\n"
                "FAMILY LABEL WARNING: stage6_classifier_head.pth stores NO "
                "class labels. Using the ASSUMED order %s from "
                "config.FAMILY_CLASS_LABELS. Verify this against your T2 "
                "training notebook's label encoder and set "
                "FAMILY_CLASS_LABELS in .env once confirmed.\n" + "=" * 78,
                self.labels,
            )

        logger.info(
            "Family classifier ready | %d families | labels=%s | "
            "unknown_threshold=%.2f",
            num_families, self.labels, self.unknown_threshold,
        )

    @torch.no_grad()
    def predict(self, embedding: torch.Tensor) -> FamilyPrediction:
        """Predict LLM family from a single 768-d embedding.

        Args:
            embedding: A 1-d tensor of shape (768,) — typically e_main
                       from the feature extractor (the main_code view
                       pooled embedding).

        Returns:
            FamilyPrediction with the predicted family, confidence,
            full probability distribution, and is_unknown flag.
        """
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)  # (1, 768)

        embedding = embedding.float().to(self.device)
        logits = self.model.forward_family(embedding)  # (1, 12)
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().tolist()

        probabilities = {
            label: float(p) for label, p in zip(self.labels, probs)
        }

        predicted_family = max(probabilities, key=probabilities.get)
        confidence = probabilities[predicted_family]
        is_unknown = confidence < self.unknown_threshold

        if is_unknown:
            logger.info(
                "Family prediction below threshold (%.2f < %.2f): "
                "reporting as Unknown. Raw prediction was '%s'.",
                confidence, self.unknown_threshold, predicted_family,
            )

        return FamilyPrediction(
            predicted_family="Unknown" if is_unknown else predicted_family,
            confidence=confidence,
            probabilities=probabilities,
            is_unknown=is_unknown,
        )
