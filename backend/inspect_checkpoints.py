"""
inspect_checkpoints.py
========================
STEP 1/2 of the required implementation order: run this BEFORE
trusting the app's model loading. It inspects the two checkpoint
files you place in models/ and prints exactly what's in them, so
nothing downstream is a silent assumption.

Usage (from backend/):
    python inspect_checkpoints.py

It never loads a model into a live nn.Module for a forward pass --
it only reads raw torch.load() output and reports on it, so it is
safe to run even against a checkpoint that fails at inference time
due to a library-version mismatch.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.config import CLASSIFIER_CHECKPOINT_PATH, MODERNBERT_MODEL_PATH  # noqa: E402

FAMILY_CHECKPOINT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "stage6_classifier_head.pth",
)


def inspect_modernbert_checkpoint(path: str) -> None:
    print("=" * 78)
    print(f"INSPECTING MODERNBERT CHECKPOINT: {path}")
    print("=" * 78)

    if not os.path.exists(path):
        print(f"NOT FOUND at {path}. Place model_complete.pth there (or set "
              f"MODERNBERT_MODEL_PATH in .env) before running the app.")
        return

    loaded = torch.load(path, map_location="cpu", weights_only=False)
    print("Top-level type:", type(loaded))

    if isinstance(loaded, torch.nn.Module):
        state_dict = loaded.state_dict()
        print("It's a pickled nn.Module. Config (if present):")
        config = getattr(loaded, "config", None)
        if config is not None:
            print(" ", {k: v for k, v in vars(config).items() if not k.startswith("_")})
        print(f"Number of parameter/buffer tensors: {len(state_dict)}")
        print("Sample keys:", list(state_dict.keys())[:10])
    elif isinstance(loaded, dict):
        print("It's a dict. Top-level keys:", list(loaded.keys())[:20])
        if all(isinstance(v, torch.Tensor) for v in loaded.values()):
            print("Looks like a raw state_dict.")
            print(f"Number of tensors: {len(loaded)}")
            print("Sample keys:", list(loaded.keys())[:10])
    else:
        print("Unrecognized top-level type -- inspect manually.")


def inspect_classifier_checkpoint(path: str) -> None:
    print("\n" + "=" * 78)
    print(f"INSPECTING CLASSIFIER CHECKPOINT: {path}")
    print("=" * 78)

    if not os.path.exists(path):
        print(f"NOT FOUND at {path}. Place best_4class_classifier.pt there (or "
              f"set CLASSIFIER_CHECKPOINT_PATH in .env) before running the app.")
        return

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    if not isinstance(checkpoint, dict):
        print("Unexpected: top-level object is not a dict:", type(checkpoint))
        return

    print("Top-level keys:", list(checkpoint.keys()))

    if "architecture" in checkpoint:
        print("architecture metadata:", checkpoint["architecture"])
    else:
        print("NO 'architecture' key found -- classifier.py will fall back to "
              "its built-in default architecture. Verify this matches training.")

    if "classes" in checkpoint:
        print("classes metadata:", checkpoint["classes"])
        generic = all(str(c).lower().startswith("class_") for c in checkpoint["classes"])
        if generic:
            print(
                "!! These are GENERIC placeholder names, not semantic labels. "
                "The app will fall back to config.CLASS_LABELS and log a "
                "warning at load time. Set CLASS_LABEL_ORDER in .env once you "
                "confirm the true Human/AI/Hybrid/Adversarial index order from "
                "your T3 training notebook / dataset label encoder."
            )
    else:
        print("NO 'classes' key found -- same fallback behavior as above applies.")

    if "model_state_dict" in checkpoint:
        keys = list(checkpoint["model_state_dict"].keys())
        print(f"model_state_dict has {len(keys)} tensors. Sample keys:", keys[:10])
    else:
        print("!! NO 'model_state_dict' key -- classifier.py will refuse to load this file.")

    for meta_key in ("epoch", "val_macro_f1", "best_val_macro_f1", "val_accuracy"):
        if meta_key in checkpoint:
            print(f"{meta_key}:", checkpoint[meta_key])


def inspect_family_checkpoint(path: str) -> None:
    """Inspect stage6_classifier_head.pth — the LLM-family classifier head
    trained on AICD-T2.  Prints everything needed to reconstruct the
    architecture in family_classifier.py:
      - checkpoint format (dict vs Module vs raw state_dict)
      - top-level metadata keys (architecture, classes, epoch, etc.)
      - every tensor key + shape (to deduce input_dim, hidden_dim, num_classes)
      - class/label names if present
    """
    print("\n" + "=" * 78)
    print(f"INSPECTING FAMILY CLASSIFIER CHECKPOINT: {path}")
    print("=" * 78)

    if not os.path.exists(path):
        print(f"NOT FOUND at {path}. Place stage6_classifier_head.pth in the "
              f"models/ folder or set FAMILY_CLASSIFIER_PATH in .env.")
        return

    loaded = torch.load(path, map_location="cpu", weights_only=False)
    print("Top-level type:", type(loaded))

    # ── Case 1: pickled nn.Module ────────────────────────────────────
    if isinstance(loaded, torch.nn.Module):
        print("Format: PICKLED nn.Module")
        state_dict = loaded.state_dict()
        print(f"Number of parameter/buffer tensors: {len(state_dict)}")
        print("\nAll state_dict keys + shapes:")
        for k, v in state_dict.items():
            print(f"  {k:50s}  {tuple(v.shape)}")

        config = getattr(loaded, "config", None)
        if config is not None:
            print("\nModel config attributes:")
            print(" ", {k: v for k, v in vars(config).items() if not k.startswith("_")})

        # Try to read class names from common attributes
        for attr in ("classes", "class_names", "labels", "label_names",
                     "family_labels", "num_classes"):
            val = getattr(loaded, attr, None)
            if val is not None:
                print(f"\nAttribute '{attr}':", val)

        # Print the module's repr for structure
        print("\nModule repr (truncated to 2000 chars):")
        print(repr(loaded)[:2000])
        return

    # ── Case 2: dict checkpoint ──────────────────────────────────────
    if isinstance(loaded, dict):
        print("Format: DICT")
        print("Top-level keys:", list(loaded.keys()))

        # Print every non-tensor metadata value
        for k, v in loaded.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: Tensor {tuple(v.shape)} ({v.dtype})")
            elif isinstance(v, dict) and all(isinstance(vv, torch.Tensor)
                                              for vv in v.values()):
                print(f"  {k}: state_dict with {len(v)} tensors")
                for kk, vv in v.items():
                    print(f"    {kk:50s}  {tuple(vv.shape)} ({vv.dtype})")
            else:
                print(f"  {k}: {v}")

        # Check if this IS a raw state_dict (all values are tensors)
        if all(isinstance(v, torch.Tensor) for v in loaded.values()):
            print("\nLooks like a RAW state_dict (no wrapper).")
            print(f"Number of tensors: {len(loaded)}")
            print("\nAll keys + shapes:")
            for k, v in loaded.items():
                print(f"  {k:50s}  {tuple(v.shape)} ({v.dtype})")

            # Attempt to infer architecture from weight shapes
            print("\n--- Architecture inference from tensor shapes ---")
            for k, v in loaded.items():
                if "weight" in k and v.dim() == 2:
                    out_dim, in_dim = v.shape
                    print(f"  Linear layer '{k}': in={in_dim} -> out={out_dim}")
                elif "bias" in k and v.dim() == 1:
                    print(f"  Bias '{k}': dim={v.shape[0]}")

        # Look for metadata keys common in training checkpoints
        for meta_key in ("architecture", "classes", "class_names", "labels",
                         "label_names", "family_labels", "num_classes",
                         "label_encoder", "label_map", "class_to_idx",
                         "idx_to_class", "epoch", "val_macro_f1",
                         "best_val_macro_f1", "val_accuracy", "train_accuracy",
                         "model_state_dict", "optimizer_state_dict"):
            if meta_key in loaded:
                val = loaded[meta_key]
                if not isinstance(val, (torch.Tensor, dict)):
                    print(f"\nMetadata '{meta_key}': {val}")
        return

    # ── Case 3: something else ───────────────────────────────────────
    print(f"Unrecognized type: {type(loaded)}")
    print("repr (truncated):", repr(loaded)[:1000])


if __name__ == "__main__":
    inspect_modernbert_checkpoint(MODERNBERT_MODEL_PATH)
    inspect_classifier_checkpoint(CLASSIFIER_CHECKPOINT_PATH)
    inspect_family_checkpoint(FAMILY_CHECKPOINT_PATH)
