from __future__ import annotations
from dataclasses import dataclass
from typing import List
from .state import PerceptionState

@dataclass
class RiskResult:
    score: float
    reasons: List[str]

def score_risk(state: PerceptionState) -> RiskResult:
    score = 0.05
    reasons: List[str] = []

    # Unknown people in frame
    unknown = [p for p in state.people if not p.get("recognized")]
    if unknown:
        score += 0.20
        reasons.append("unknown person")

    # Pose availability
    if state.pose.get("detected"):
        # Light nudge if pose exists but keypoints are low quality
        if not state.pose.get("keypoints"):
            score += 0.05
            reasons.append("pose low confidence")

    # If a recognized person exists but we are not authorized yet
    recognized_names = [p.get("name") for p in state.people if p.get("recognized")]
    if recognized_names and not state.authorized_user:
        score += 0.10
        reasons.append("recognized but locked")

    # Clamp
    score = max(0.0, min(1.0, score))
    return RiskResult(score=score, reasons=reasons)
