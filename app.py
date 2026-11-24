"""HF Spaces entrypoint — serves the existing FastAPI backend on port 7860."""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "backend"))

# Download model weights before the app loads them
exec(compile(open(os.path.join(ROOT, "backend", "download_models.py")).read(),
             "download_models.py", "exec"))

import uvicorn
from app.main import app  # backend/app/main.py -> FastAPI app

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))