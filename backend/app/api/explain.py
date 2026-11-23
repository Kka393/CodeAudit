"""
api/explain.py
================
Source-code line-level attribution endpoint for the 4-class classifier.

POST /explain
Request:  { "code": "...", "language": "python", "target_class": Optional[str] }
Response: {
    "error": null,
    "target_class": "Machine",
    "baseline_confidence": 0.84,
    "method": "counterfactual_line_occlusion",
    "lines": [
        { "line_number": 1, "content": "...", "attribution": 0.18, "direction": "positive", "is_significant": true },
        ...
    ],
    "summary": { ... }
}

Explains the primary 4-class classifier ({Human, Machine, Hybrid, Adversarial}).
Does NOT explain or attach to the downstream LLM-family classifier.
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.config import SUPPORTED_LANGUAGES

logger = logging.getLogger(__name__)

router = APIRouter()

# Lazy singleton for the code line explainer
_line_explainer = None


def _get_line_explainer():
    """Initialize the line-level explainer on first use."""
    global _line_explainer
    if _line_explainer is None:
        from app.main import get_pipeline
        from app.ml.shap_explainer import CodeLineExplainer

        pipeline = get_pipeline()
        _line_explainer = CodeLineExplainer(pipeline.feature_extractor, pipeline.classifier)
    return _line_explainer


class ExplainRequest(BaseModel):
    code: str = Field(..., description="Raw source code to explain.")
    language: str = Field("python", description=f"One of: {SUPPORTED_LANGUAGES}")
    target_class: Optional[str] = Field(
        None,
        description="Optional 4-class target class to explain (Human, Machine, Hybrid, Adversarial). "
                    "If omitted, explains the model's top predicted class.",
    )


@router.post("/explain")
def explain(request: ExplainRequest):
    """Generate source-code line-level attribution for the 4-class prediction.

    Evaluates counterfactual impact of each line on the 4-class classifier.
    Positive attribution indicates the line contributed towards the predicted class;
    Negative attribution indicates the line opposed it.
    """
    if not request.code or not request.code.strip():
        raise HTTPException(status_code=400, detail="`code` must be a non-empty string.")

    try:
        explainer = _get_line_explainer()
        explanation = explainer.explain(
            code=request.code,
            language=request.language,
            target_class=request.target_class,
        )

        if explanation is None:
            return {
                "error": "Line-level attribution could not be computed for this code snippet.",
                "target_class": None,
                "baseline_confidence": None,
                "lines": [],
                "summary": {},
            }

        result = explanation.to_dict()
        result["error"] = None
        return result

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unhandled error in /explain")
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}") from exc
