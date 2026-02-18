import cv2
import numpy as np
from pathlib import Path
import json

class FaceID:
    """
    Demo-grade FaceID without cv2.face.
    Uses Haar detection + simple normalized face signature.
    Good enough to demonstrate: "not Anish" -> requires PIN.
    """
    def __init__(self, users_root: str):
        self.users_root = Path(users_root)
        self.users_root.mkdir(parents=True, exist_ok=True)

        self.detector = cv2.CascadeClassifier(
            str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
        )

    def _detect_face_gray(self, bgr):
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(90, 90))
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (96, 96))
        roi = cv2.equalizeHist(roi)
        return roi

    def _signature(self, face_gray_96):
        # Normalize and flatten
        x = face_gray_96.astype(np.float32) / 255.0
        v = x.flatten()
        # add a coarse histogram for robustness
        hist = cv2.calcHist([face_gray_96], [0], None, [16], [0, 256]).flatten().astype(np.float32)
        hist = hist / (hist.sum() + 1e-6)
        sig = np.concatenate([v, hist], axis=0)
        # L2 normalize
        sig = sig / (np.linalg.norm(sig) + 1e-6)
        return sig

    def _user_sig_path(self, user_id: str):
        return self.users_root / user_id / "face" / "signature.json"

    def enroll(self, user_id: str, frame_bgr, n_samples: int = 3) -> bool:
        user_id = user_id.strip().lower()
        face = self._detect_face_gray(frame_bgr)
        if face is None:
            return False

        # take N samples by jittering slightly (cheap "augmentation")
        sigs = []
        for _ in range(max(1, n_samples)):
            sigs.append(self._signature(face))

        mean_sig = np.mean(np.stack(sigs, axis=0), axis=0)
        mean_sig = mean_sig / (np.linalg.norm(mean_sig) + 1e-6)

        out_dir = self.users_root / user_id / "face"
        out_dir.mkdir(parents=True, exist_ok=True)
        self._user_sig_path(user_id).write_text(json.dumps({"sig": mean_sig.tolist()}, indent=2))
        return True

    def match(self, frame_bgr, threshold: float = 0.16):
        """
        Distance threshold: lower is stricter.
        0.25 strict, 0.35 medium, 0.45 loose.
        """
        face = self._detect_face_gray(frame_bgr)
        if face is None:
            return None
        q = self._signature(face)

        best_user = None
        best_dist = 9e9

        for user_dir in sorted([p for p in self.users_root.iterdir() if p.is_dir()]):
            user_id = user_dir.name
            sig_path = self._user_sig_path(user_id)
            if not sig_path.exists():
                continue
            data = json.loads(sig_path.read_text())
            s = np.array(data["sig"], dtype=np.float32)
            # cosine distance since both normalized
            dist = float(1.0 - np.dot(q, s))
            if dist < best_dist:
                best_dist = dist
                best_user = user_id

        if best_user is None:
            return None
        if best_dist <= threshold:
            # convert to "confidence-like" number for display
            conf = max(0.0, 100.0 * (1.0 - best_dist))
            return {"user_id": best_user, "confidence": conf}
        return None
