import os
import json
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import cv2
from insightface.app import FaceAnalysis

def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(1.0 - np.dot(a, b))

class FaceEmbedID:
    """
    Multi-face recognition using embeddings (InsightFace).
    Stores per-user embeddings locally.
    """
    def __init__(self, users_root: str):
        self.users_root = users_root
        self.app = FaceAnalysis(name="buffalo_l")
        # ctx_id=0 uses GPU if available, otherwise CPU
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    def _user_dir(self, user_id: str) -> Path:
        return Path(self.users_root) / user_id

    def _emb_path(self, user_id: str) -> Path:
        return self._user_dir(user_id) / "face" / "embeddings.json"

    def ensure_user(self, user_id: str):
        (self._user_dir(user_id) / "face").mkdir(parents=True, exist_ok=True)

    def enroll(self, user_id: str, frame_bgr, n_samples: int = 10) -> bool:
        """
        Capture embeddings from current frame. Best if the person is close and well lit.
        """
        self.ensure_user(user_id)
        faces = self.detect_faces(frame_bgr)
        if not faces:
            return False

        # pick largest face in frame
        face = max(faces, key=lambda f: (f["bbox"][2]-f["bbox"][0])*(f["bbox"][3]-f["bbox"][1]))
        emb = np.array(face["embedding"], dtype=np.float32)

        path = self._emb_path(user_id)
        existing = []
        if path.exists():
            try:
                existing = json.loads(path.read_text())
            except Exception:
                existing = []

        existing.append(emb.tolist())
        path.write_text(json.dumps(existing, indent=2))
        return True

    def detect_faces(self, frame_bgr) -> List[Dict]:
        """
        Returns list of faces with bbox + embedding.
        bbox = [x1,y1,x2,y2]
        """
        faces = self.app.get(frame_bgr)
        out = []
        for f in faces:
            x1, y1, x2, y2 = f.bbox.astype(int).tolist()
            out.append({
                "bbox": [x1, y1, x2, y2],
                "embedding": f.embedding.astype(np.float32).tolist(),
                "det_score": float(getattr(f, "det_score", 0.0))
            })
        return out

    def match_faces(self, frame_bgr, threshold: float = 0.35) -> List[Dict]:
        """
        For each detected face, returns best match user_id if distance <= threshold.
        """
        faces = self.detect_faces(frame_bgr)
        if not faces:
            return []

        # Load all enrolled embeddings
        gallery = []
        for user_dir in Path(self.users_root).glob("*"):
            if not user_dir.is_dir():
                continue
            user_id = user_dir.name
            path = self._emb_path(user_id)
            if not path.exists():
                continue
            try:
                embs = json.loads(path.read_text())
            except Exception:
                continue
            for e in embs:
                gallery.append((user_id, np.array(e, dtype=np.float32)))

        results = []
        for face in faces:
            emb = np.array(face["embedding"], dtype=np.float32)
            best = None
            best_dist = 999.0
            for uid, gemb in gallery:
                d = _cosine_distance(emb, gemb)
                if d < best_dist:
                    best_dist = d
                    best = uid

            if best is not None and best_dist <= threshold:
                results.append({
                    "bbox": face["bbox"],
                    "user_id": best,
                    "distance": best_dist,
                    "confidence": float(max(0.0, 1.0 - best_dist))  # simple proxy
                })
            else:
                results.append({
                    "bbox": face["bbox"],
                    "user_id": None,
                    "distance": best_dist
                })
        return results
