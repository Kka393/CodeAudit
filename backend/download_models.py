import os
from huggingface_hub import hf_hub_download

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
MODEL_DIR = os.path.join(ROOT, "models")
MODEL_REPO = os.environ.get("MODEL_REPO", "kka2006/aicd-models")

os.makedirs(MODEL_DIR, exist_ok=True)
for name in ["model_complete.pth", "best_4class_classifier.pt", "stage6_classifier_head.pth"]:
    try:
        hf_hub_download(repo_id=MODEL_REPO, filename=name, local_dir=MODEL_DIR,
                        token=os.environ.get("HF_TOKEN") or None)
        print(f"[models] {name} ready")
    except Exception as e:
        print(f"[models] WARNING {name}: {e}")