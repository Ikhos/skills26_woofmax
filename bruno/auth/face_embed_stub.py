"""
Stub FaceEmbedID when InsightFace is disabled (e.g. BRUNO_DISABLE_FACEID=1 on Raspberry Pi).
No face detection or enrollment; match_faces always returns [].
"""
from pathlib import Path
from typing import List, Dict


class FaceEmbedID:
    """No-op face recognition when InsightFace is unavailable or disabled on Pi."""

    def __init__(self, users_root: str):
        self.users_root = users_root

    def ensure_user(self, user_id: str):
        (Path(self.users_root) / user_id / "face").mkdir(parents=True, exist_ok=True)

    def enroll(self, user_id: str, frame_bgr, n_samples: int = 10) -> bool:
        return False

    def detect_faces(self, frame_bgr) -> List[Dict]:
        return []

    def match_faces(self, frame_bgr, threshold: float = 0.35) -> List[Dict]:
        return []
