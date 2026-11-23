"""
api/debate.py
===============
SSE (Server-Sent Events) endpoint for the multi-round LLM debate.

GET /debate/stream?code=<urlencoded>&ai_prob=0.63&max_rounds=3

Streams DebateEvent objects as SSE `data:` lines. The frontend
connects via EventSource or fetch with ReadableStream and renders
each event (advocate/skeptic/judge) as it arrives, giving a live
updating feed of the debate in progress.

The debate is ONLY triggered manually by the user after a low-confidence
/predict result. It never runs automatically.
"""

import json
import logging
from typing import Optional

from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter()

# Singleton debate engine (initialized lazily)
_debate_engine = None


def _get_debate_engine():
    """Lazy singleton for the DebateEngine. Constructed on first use
    rather than at import time, so missing API keys only error when
    debate is actually requested."""
    global _debate_engine
    if _debate_engine is None:
        from app.agents.debate_engine import DebateEngine
        _debate_engine = DebateEngine()
    return _debate_engine


class DebateStreamRequest(BaseModel):
    code: str = Field(..., description="Source code to debate")
    ai_prob: float = Field(..., ge=0.0, le=1.0, description="AI probability from /predict")
    max_rounds: Optional[int] = Field(None, ge=1, le=5, description="Max debate rounds")


def _stream_debate(code: str, ai_prob: float, max_rounds: Optional[int]):
    from app.config import DEBATE_MAX_ROUNDS

    rounds = max_rounds if max_rounds is not None else DEBATE_MAX_ROUNDS

    def event_generator():
        # Flush headers and proxy buffers immediately with a comment
        yield f": {' ' * 1024}\n\n"
        try:
            engine = _get_debate_engine()
            # SWAP: replace run_multi_stream with run_single for single-round
            for event in engine.run_multi_stream(code, ai_prob, max_rounds=rounds):
                yield f"data: {json.dumps(event.to_dict())}\n\n"
        except Exception as exc:
            logger.exception("Debate stream error: %s", exc)
            error_event = {
                "event_type": "error",
                "content": str(exc),
            }
            yield f"data: {json.dumps(error_event)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@router.post("/debate/stream")
def debate_stream_post(request: DebateStreamRequest):
    """POST /debate/stream — streams SSE debate events from JSON body."""
    return _stream_debate(request.code, request.ai_prob, request.max_rounds)


@router.get("/debate/stream")
def debate_stream_get(
    code: str = Query(..., description="Source code to debate"),
    ai_prob: float = Query(..., ge=0.0, le=1.0, description="AI probability from /predict"),
    max_rounds: Optional[int] = Query(None, ge=1, le=5, description="Max debate rounds"),
):
    """GET /debate/stream — streams SSE debate events from query params."""
    return _stream_debate(code, ai_prob, max_rounds)
