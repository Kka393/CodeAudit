"""Download model weights from Hugging Face Hub at container startup."""
import os
from huggingface_hub import hf_hub_download

MODEL_REPO = os.environ.get("MODEL_REPO", "YOUR_USERNAME/aicd-models")
MODEL_DIR = "/models"
HF_TOKEN = os.environ.get("HF_TOKEN", "")

FILES = [
    "model_complete.pth",
    "best_4class_classifier.pt",
    "stage6_classifier_head.pth",
]

os.makedirs(MODEL_DIR, exist_ok=True)

for name in FILES:
    try:
        path = hf_hub_download(
            repo_id=MODEL_REPO,
            filename=name,
            local_dir=MODEL_DIR,
            token=HF_TOKEN or None,
        )
        print(f"[models] {name} -> {path}")
    except Exception as exc:
        print(f"[models] WARNING: could not download {name}: {exc}")