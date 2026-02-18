import numpy as np
import cv2
import mediapipe as mp
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

@dataclass
class PoseResult:
    detected: bool
    fall_score: float
    keypoints: Optional[List[Dict[str, float]]]
    notes: Dict[str, Any]

POSE_EDGES: List[Tuple[int, int]] = [
    (11, 12), (11, 23), (12, 24), (23, 24),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (23, 25), (25, 27),
    (24, 26), (26, 28),
]

class PoseAnalyzer:
    def __init__(self, model_path: str = "bruno/models/pose_landmarker_lite.task"):
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)

    def close(self):
        try:
            self.landmarker.close()
        except Exception:
            pass

    def analyze_bgr_frame(self, frame_bgr, timestamp_ms: int) -> PoseResult:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = self.landmarker.detect_for_video(mp_img, timestamp_ms)

        if not res.pose_landmarks:
            return PoseResult(False, 0.0, None, {"reason": "no_pose"})

        lm = res.pose_landmarks[0]
        pts = [{"x": float(p.x), "y": float(p.y), "z": float(p.z), "v": float(p.visibility)} for p in lm]

        ls, rs, lh, rh = pts[11], pts[12], pts[23], pts[24]
        dx_s = (rs["x"] - ls["x"])
        dy_s = (rs["y"] - ls["y"])
        slope = abs(dy_s) / (abs(dx_s) + 1e-6)

        torso_h = abs(((lh["y"] + rh["y"]) / 2.0) - ((ls["y"] + rs["y"]) / 2.0))

        horizontal_score = 1.0 - np.clip(slope / 1.5, 0.0, 1.0)
        flat_score = 1.0 - np.clip(torso_h / 0.25, 0.0, 1.0)

        fall_score = float(np.clip(0.55 * horizontal_score + 0.45 * flat_score, 0.0, 1.0))

        notes = {"slope": float(slope), "torso_h": float(torso_h)}
        return PoseResult(True, fall_score, pts, notes)

def _pt_abs(keypoints, i, w, h, min_vis):
    p = keypoints[i]
    if p["v"] < min_vis:
        return None
    return (int(p["x"] * w), int(p["y"] * h))

def draw_pose_skeleton_in_bbox(frame_bgr, keypoints, bbox, min_vis: float = 0.55):
    """
    Draw skeleton ONLY inside a bbox (x1,y1,x2,y2). Prevents skeletons on random stuff.
    """
    if keypoints is None or bbox is None:
        return

    x1, y1, x2, y2 = bbox
    h, w = frame_bgr.shape[:2]

    # draw to overlay then copy only bbox region
    overlay = frame_bgr.copy()

    def in_box(pt):
        if pt is None:
            return False
        x, y = pt
        return (x1 <= x <= x2) and (y1 <= y <= y2)

    # edges
    for a, b in POSE_EDGES:
        pa = _pt_abs(keypoints, a, w, h, min_vis)
        pb = _pt_abs(keypoints, b, w, h, min_vis)
        if not (in_box(pa) and in_box(pb)):
            continue
        cv2.line(overlay, pa, pb, (255, 255, 255), 2)

    # joints
    for i in range(min(33, len(keypoints))):
        p = _pt_abs(keypoints, i, w, h, min_vis)
        if not in_box(p):
            continue
        cv2.circle(overlay, p, 3, (255, 255, 255), -1)

    # copy only ROI back
    frame_bgr[y1:y2, x1:x2] = overlay[y1:y2, x1:x2]
