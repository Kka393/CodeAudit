"""
api/predict.py
================
Thin FastAPI route. All actual orchestration lives in
app/main.py:run_full_pipeline() -- this file only validates the
HTTP request/response shape and calls it.
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.config import SUPPORTED_LANGUAGES

logger = logging.getLogger(__name__)

router = APIRouter()


class PredictRequest(BaseModel):
    code: str = Field(..., description="Raw source code to analyze.")
    language: str = Field("python", description=f"One of: {SUPPORTED_LANGUAGES}")
    confidence_threshold: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Override for the confidence gate threshold (0.0-1.0). "
                    "If omitted, uses CONFIDENCE_THRESHOLD from server config.",
    )


@router.post("/predict")
def predict(request: PredictRequest):
    from app.main import PipelineError, run_full_pipeline  # local import avoids circular import at module load time

    try:
        return run_full_pipeline(
            request.code,
            request.language,
            confidence_threshold=request.confidence_threshold,
        )
    except PipelineError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unhandled error in /predict")
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}") from exc


@router.get("/health")
def health():
    return {"status": "ok"}
