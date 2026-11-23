"""
ml/modernbert_encoder.py
==========================
Loads the T2-TRAINED ModernBERT checkpoint you supply
(config.MODERNBERT_MODEL_PATH, the file you called
"model_complete.pth") as the single, shared, FROZEN encoder used for
all three views.

WHY THIS IS NOT A PLAIN `torch.load(path)` + `.eval()`:

That checkpoint was produced by `torch.save(model, path)` -- the
whole pickled nn.Module, not `state_dict()`. Unpickling it re-binds
its stored instance attributes to whatever ModernBertConfig /
ModernBertAttention classes are installed in THIS environment's
`transformers` right now. If that version differs even slightly from
the one used at training time, calling `.forward()` on the unpickled
object raises AttributeErrors (this project has already hit
`ModernBertConfig.reference_compile` and
`ModernBertAttention.num_heads` this way). Patching individual
missing attributes is whack-a-mole, not a fix.

THE FIX IMPLEMENTED HERE:
  1. Load whatever is at MODERNBERT_MODEL_PATH.
  2. Extract ONLY its tensors via `.state_dict()` (never call
     `.forward()` on a raw unpickled object).
  3. Reconstruct a fresh ModernBERT architecture via
     `AutoModel.from_pretrained(MODERNBERT_BASE_ARCHITECTURE)` under
     the CURRENTLY installed transformers -- this defines a
     `forward()` that matches the installed library exactly.
  4. Load the extracted tensors into that fresh architecture with
     `strict=False`, auto-retrying with common key-prefix stripping
     ("model.", "modernbert.", ...) if the raw keys don't line up,
     and report the match quality instead of assuming it's correct.
  5. Freeze every parameter. Run one real sanity forward pass before
     declaring the encoder ready.

This mirrors the diagnosed fix from the T3 training-notebook
incident, generalized into a reusable component.
"""

from typing import List, Tuple

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from app.config import DEVICE, MAX_LENGTH, MODERNBERT_BASE_ARCHITECTURE, MODERNBERT_MODEL_PATH


class ModernBertLoadError(RuntimeError):
    """Raised when the T2 checkpoint cannot be reconciled with the
    currently-installed transformers architecture. Deliberately NOT
    caught silently anywhere -- a bad encoder load must surface, not
    be papered over with random weights."""


def _extract_state_dict(checkpoint_path: str) -> dict:
    loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if isinstance(loaded, dict) and all(isinstance(v, torch.Tensor) for v in loaded.values()):
        return loaded

    if isinstance(loaded, dict) and "state_dict" in loaded:
        return loaded["state_dict"]

    if isinstance(loaded, nn.Module):
        # Reading .state_dict() only touches stored tensors -- it
        # does not invoke forward(), so it is NOT subject to the
        # reference_compile / num_heads AttributeErrors.
        return loaded.state_dict()

    raise ModernBertLoadError(
        f"Unrecognized checkpoint format at {checkpoint_path}: type={type(loaded)}. "
        f"Expected a raw state_dict, a dict with a 'state_dict' key, or a pickled "
        f"nn.Module. Refusing to guess further."
    )


def _load_matching_state_dict(model: nn.Module, state_dict: dict) -> Tuple[int, int]:
    target_keys = set(model.state_dict().keys())

    def try_load(sd):
        missing, unexpected = model.load_state_dict(sd, strict=False)
        overlap = len(target_keys & set(sd.keys()))
        return overlap, missing, unexpected

    best_overlap, missing, unexpected = try_load(state_dict)
    best_sd = state_dict

    if best_overlap < 0.9 * len(target_keys):
        for prefix in ("model.", "modernbert.", "encoder.", "bert."):
            if any(k.startswith(prefix) for k in state_dict.keys()):
                stripped = {
                    (k[len(prefix):] if k.startswith(prefix) else k): v
                    for k, v in state_dict.items()
                }
                overlap, m, u = try_load(stripped)
                if overlap > best_overlap:
                    best_overlap, missing, unexpected = overlap, m, u
                    best_sd = stripped

    # Final authoritative load with whichever variant matched best.
    model.load_state_dict(best_sd, strict=False)

    return best_overlap, len(target_keys)


class ModernBertEncoder:
    """
    Loads and wraps the T2-trained, frozen ModernBERT model +
    tokenizer. Used identically for all three views (main_code,
    sub_functions, comments_view) -- there is only ever ONE encoder
    instance, called three times.
    """

    def __init__(
        self,
        checkpoint_path: str = MODERNBERT_MODEL_PATH,
        base_architecture: str = MODERNBERT_BASE_ARCHITECTURE,
        device: torch.device = DEVICE,
        max_length: int = MAX_LENGTH,
    ):
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(base_architecture)

        state_dict = _extract_state_dict(checkpoint_path)

        self.model = AutoModel.from_pretrained(base_architecture)
        matched, total = _load_matching_state_dict(self.model, state_dict)

        if matched < 0.9 * total:
            raise ModernBertLoadError(
                f"T2 checkpoint at {checkpoint_path} only matched {matched}/{total} "
                f"expected parameter tensors for {base_architecture}. This likely "
                f"means the checkpoint is not a bare ModernBertModel encoder, or "
                f"was trained from a different base architecture. Refusing to "
                f"proceed with a mismatched/partially-random encoder -- inspect "
                f"the checkpoint with backend/inspect_checkpoints.py before retrying."
            )

        self.match_ratio = matched / total

        self.model.to(self.device)
        self.model.eval()

        for param in self.model.parameters():
            param.requires_grad = False

        self.hidden_size = self.model.config.hidden_size

        self._sanity_check()

    def _sanity_check(self) -> None:
        with torch.no_grad():
            encoded = self.tokenizer(
                "def f(x):\n    return x", return_tensors="pt"
            ).to(self.device)
            outputs = self.model(**encoded)
            if outputs.last_hidden_state.shape[-1] != self.hidden_size:
                raise ModernBertLoadError(
                    "Sanity forward pass produced an unexpected hidden size; "
                    "the loaded T2 checkpoint may not be compatible."
                )

    def is_frozen(self) -> bool:
        return all(not p.requires_grad for p in self.model.parameters())

    @torch.no_grad()
    def encode(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        if not texts:
            raise ValueError("ModernBertEncoder.encode() received an empty list of texts.")

        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.model(**encoded)
        return outputs.last_hidden_state, encoded["attention_mask"]
