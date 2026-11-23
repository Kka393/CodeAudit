"""
agents/debate_stub.py
=======================
Placeholder for the future debate mechanism. In THIS phase it must
never actually debate anything -- it only logs that debate is not
yet implemented and returns the aggregator result unchanged, with
debate_applied=False. The evidence packet is preserved so a later
phase can consume it without recomputation.
"""

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


def run_debate_loop(evidence_packet: Dict[str, Any]) -> Dict[str, Any]:
    logger.info(
        "Confidence gate flagged uncertainty (needs_debate=True). "
        "LLM debate is available on request via the UI or /debate/stream."
    )

    result = dict(evidence_packet)
    result["debate_applied"] = False
    return result
