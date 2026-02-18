import os
import json
from dataclasses import dataclass
from typing import Optional, Dict, List

import numpy as np
from insightface.app import FaceAnalysis


@dataclass
class MatchResult:
    status: str  # matched | unknown | no_face | not_enrolled
    user_id: Optional[str]
    confidence: Optional[float]
    details: Dict


class FaceID:
    """
    Pretrained face recognition using InsightFace embeddings.
    Multi-user: stores per-user embeddings in data/users/<user_id>/identity/
    """

    def __init__(self, users_root: str = "data/users", threshold: float = 0.38):
        self.users_root = users_root
        self.threshold = threshold
        os.makedirs(self.users_root, exist_ok=True)

        self.app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        self.app.prepare(ctx_id=0, det_size=(640, 640))

        # gallery: user_id -> embeddings (N, 512)
        self.gallery: Dict[str, np.ndarray] = {}
        self._load_gallery()

    def _user_identity_dir(self, user_id: str) -> str:
        return os.path.join(self.users_root, user_id, "identity")

    def _emb_path(self, user_id: str) -> str:
        return os.path.join(self._user_identity_dir(user_id), "embeddings.npy")

    def _meta_path(self, user_id: str) -> str:
        return os.path.join(self._user_identity_dir(user_id), "meta.json")

    def ensure_user_dirs(self, user_id: str):
        os.makedirs(self._user_identity_dir(user_id), exist_ok=True)
        os.makedirs(os.path.join(self.users_root, user_id, "scans"), exist_ok=True)

        memory_path = os.path.join(self.users_root, user_id, "memory.json")
        if not os.path.exists(memory_path):
            with open(memory_path, "w") as f:
                json.dump({"user_id": user_id, "created": True, "notes": ""}, f, indent=2)

    def _load_gallery(self):
        self.gallery = {}
        if not os.path.isdir(self.users_root):
            return

        for user_id in os.listdir(self.users_root):
            path = self._emb_path(user_id)
            if os.path.isfile(path):
                try:
                    emb = np.load(path)
                    if emb.ndim == 1:
                        emb = emb.reshape(1, -1)
                    self.gallery[user_id] = emb
                except Exception:
                    continue

    def _get_best_face(self, bgr_img):
        faces = self.app.get(bgr_img)
        if not faces:
            return None
        faces = sorted(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
            reverse=True,
        )
        return faces[0]

    def enroll(self, user_id: str, frames: List[np.ndarray]) -> Dict:
        self.ensure_user_dirs(user_id)

        embs = []
        for frame in frames:
            face = self._get_best_face(frame)
            if face is None or face.embedding is None:
                continue
            embs.append(face.embedding)

        if len(embs) < 8:
            return {"ok": False, "error": "Not enough face samples. Improve lighting and keep face centered."}

        embs = np.stack(embs, axis=0)
        np.save(self._emb_path(user_id), embs)

        meta = {"user_id": user_id, "num_samples": int(len(embs))}
        with open(self._meta_path(user_id), "w") as f:
            json.dump(meta, f, indent=2)

        self._load_gallery()
        return {"ok": True, "user_id": user_id, "num_samples": int(len(embs))}

    def match(self, bgr_img) -> MatchResult:
        if not self.gallery:
            return MatchResult(status="not_enrolled", user_id=None, confidence=None, details={})

        face = self._get_best_face(bgr_img)
        if face is None or face.embedding is None:
            return MatchResult(status="no_face", user_id=None, confidence=None, details={})

        q = face.embedding.astype(np.float32)
        q = q / (np.linalg.norm(q) + 1e-9)

        best_user = None
        best_dist = 1e9

        for user_id, emb_mat in self.gallery.items():
            g = emb_mat.astype(np.float32)
            g = g / (np.linalg.norm(g, axis=1, keepdims=True) + 1e-9)

            sims = np.dot(g, q)
            dist = float(1.0 - np.max(sims))  # cosine distance

            if dist < best_dist:
                best_dist = dist
                best_user = user_id

        conf = float(max(0.0, 1.0 - (best_dist / max(self.threshold, 1e-6))))

        if best_dist <= self.threshold:
            self.ensure_user_dirs(best_user)
            return MatchResult(
                status="matched",
                user_id=best_user,
                confidence=conf,
                details={"distance": best_dist, "threshold": self.threshold},
            )

        return MatchResult(
            status="unknown",
            user_id=None,
            confidence=conf,
            details={"distance": best_dist, "threshold": self.threshold, "best_candidate": best_user},
        )
