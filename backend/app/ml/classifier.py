"""
ml/classifier.py
==================
Loads the trained T3 four-class classifier checkpoint
(config.CLASSIFIER_CHECKPOINT_PATH, i.e. "best_4class_classifier.pt")
and runs inference: the three 768-d pooled view embeddings go through
their trainable 768->256 projections, get concatenated into a 768-d
fused vector, and pass through the trained MLP head to produce
P_ml over {Human, AI, Hybrid, Adversarial}.

Per the project requirements this module:
  - does NOT retrain anything
  - does NOT construct the classifier with random weights as a
    fallback -- a checkpoint load failure raises loudly instead
  - does NOT assume the numeric class index -> semantic label
    mapping; it inspects the checkpoint's own "classes" field first
    (see _resolve_label_mapping below)
"""

import logging
from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn as nn

from app.config import CLASS_LABELS, CLASSIFIER_CHECKPOINT_PATH, DEVICE
from app.ml.feature_extractor import ViewEmbeddings

logger = logging.getLogger(__name__)

_DEFAULT_ARCHITECTURE = {
    "input_dim": 768,
    "projection_dim": 256,
    "hidden_dim": 256,
    "num_classes": 4,
    "dropout": 0.30,
}


class ClassifierLoadError(RuntimeError):
    """Raised when best_4class_classifier.pt cannot be loaded as-is.
    Never caught to fall back to random weights."""


class AICD4ClassClassifier(nn.Module):
    """Architecture must exactly match the one used at T3 training
    time (three 768->projection_dim projections, concatenated, then
    an MLP head) -- this is the ONLY classifier definition in the
    app; nothing here retrains or redefines it differently."""

    def __init__(self, input_dim=768, projection_dim=256, hidden_dim=256, num_classes=4, dropout=0.30):
        super().__init__()
        self.main_projection = nn.Linear(input_dim, projection_dim)
        self.sub_projection = nn.Linear(input_dim, projection_dim)
        self.comments_projection = nn.Linear(input_dim, projection_dim)
        fused_dim = projection_dim * 3
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, e_main, e_sub, e_comments):
        p_main = self.main_projection(e_main)
        p_sub = self.sub_projection(e_sub)
        p_comments = self.comments_projection(e_comments)
        fused = torch.cat([p_main, p_sub, p_comments], dim=-1)
        return self.classifier(fused)


@dataclass
class MLPrediction:
    predicted_class: str
    confidence: float
    probabilities: Dict[str, float]


def _resolve_label_mapping(checkpoint_classes: List[str]) -> List[str]:
    """
    Decide the index -> semantic label mapping.

    Rule (never silently guess):
      - If the checkpoint's own "classes" field already contains real
        semantic names (anything other than the generic
        "Class_0".."Class_3" placeholders the training script wrote),
        trust and use it.
      - Otherwise, fall back to config.CLASS_LABELS, but log a loud,
        impossible-to-miss warning that the mapping is ASSUMED and
        must be verified against the T3 training notebook/label
        encoder, and that CLASS_LABEL_ORDER should be set in .env
        once confirmed.
    """
    generic_pattern = all(
        isinstance(c, str) and c.strip().lower().startswith("class_")
        for c in checkpoint_classes
    )

    if checkpoint_classes and not generic_pattern:
        logger.info("Using semantic class labels found in checkpoint: %s", checkpoint_classes)
        return list(checkpoint_classes)

    logger.warning(
        "=" * 78 + "\n"
        "LABEL MAPPING WARNING: best_4class_classifier.pt only stores generic "
        "placeholder class names (%s) -- it does NOT tell us which output index "
        "is Human/AI/Hybrid/Adversarial. Falling back to the ASSUMED order %s "
        "from config.CLASS_LABELS. Verify this against your T3 training "
        "notebook's label encoder and, once confirmed, set "
        "CLASS_LABEL_ORDER=Human,AI,Hybrid,Adversarial (in the correct order) "
        "in your .env so this stops being an assumption.\n" + "=" * 78,
        checkpoint_classes, CLASS_LABELS,
    )
    return list(CLASS_LABELS)


class Classifier:
    def __init__(self, checkpoint_path: str = CLASSIFIER_CHECKPOINT_PATH, device: torch.device = DEVICE):
        self.device = device

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        if "model_state_dict" not in checkpoint:
            raise ClassifierLoadError(
                f"Checkpoint at {checkpoint_path} has no 'model_state_dict' key "
                f"(found keys: {list(checkpoint.keys())}). This does not look "
                f"like best_4class_classifier.pt as produced by the T3 training "
                f"script -- refusing to guess an architecture."
            )

        architecture = checkpoint.get("architecture")
        if architecture is None:
            logger.warning(
                "Checkpoint has no 'architecture' metadata; falling back to "
                "the default architecture dict %s. Verify this matches how "
                "best_4class_classifier.pt was actually trained.",
                _DEFAULT_ARCHITECTURE,
            )
            architecture = _DEFAULT_ARCHITECTURE

        self.model = AICD4ClassClassifier(
            input_dim=architecture.get("input_dim", 768),
            projection_dim=architecture.get("projection_dim", 256),
            hidden_dim=architecture.get("hidden_dim", 256),
            num_classes=architecture.get("num_classes", 4),
            dropout=architecture.get("dropout", 0.30),
        )

        try:
            self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        except RuntimeError as exc:
            raise ClassifierLoadError(
                f"best_4class_classifier.pt's state_dict does not match the "
                f"AICD4ClassClassifier architecture defined in ml/classifier.py "
                f"(architecture metadata used: {architecture}). Original error: "
                f"{exc}"
            ) from exc

        self.model.to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self.labels = _resolve_label_mapping(checkpoint.get("classes", []))
        if len(self.labels) != architecture.get("num_classes", 4):
            raise ClassifierLoadError(
                f"Resolved {len(self.labels)} class labels but the classifier "
                f"head has {architecture.get('num_classes', 4)} output logits. "
                f"Fix CLASS_LABEL_ORDER in .env."
            )

    @torch.no_grad()
    def predict(self, embeddings: ViewEmbeddings) -> MLPrediction:
        e_main = embeddings.e_main.float().to(self.device).unsqueeze(0)
        e_sub = embeddings.e_sub.float().to(self.device).unsqueeze(0)
        e_comments = embeddings.e_comments.float().to(self.device).unsqueeze(0)

        logits = self.model(e_main, e_sub, e_comments)
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().tolist()

        probabilities = {label: float(p) for label, p in zip(self.labels, probs)}
        predicted_class = max(probabilities, key=probabilities.get)
        confidence = probabilities[predicted_class]

        return MLPrediction(
            predicted_class=predicted_class,
            confidence=confidence,
            probabilities=probabilities,
        )
