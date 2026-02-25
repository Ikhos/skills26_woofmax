"""
Stub PoseAnalyzer when MediaPipe is disabled (e.g. BRUNO_DISABLE_POSE=1 on Raspberry Pi).
Pose detection disabled; analyze_bgr_frame always returns no pose.
"""
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

POSE_EDGES: List[Tuple[int, int]] = [
    (11, 12), (11, 23), (12, 24), (23, 24),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (23, 25), (25, 27),
    (24, 26), (26, 28),
]


@dataclass
class PoseResult:
    detected: bool
    fall_score: float
    keypoints: Optional[List[Dict[str, float]]]
    notes: Dict[str, Any]


class PoseAnalyzer:
    """No-op pose when MediaPipe is unavailable or disabled on Pi."""

    def __init__(self, model_path: str = "bruno/models/pose_landmarker_lite.task"):
        pass

    def close(self):
        pass

    def analyze_bgr_frame(self, frame_bgr, timestamp_ms: int) -> PoseResult:
        return PoseResult(False, 0.0, None, {"reason": "pose_disabled"})


def draw_pose_skeleton_in_bbox(frame_bgr, keypoints, bbox, min_vis: float = 0.55):
    """No-op when pose is disabled."""
    pass
