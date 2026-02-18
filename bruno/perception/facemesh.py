import time
from dataclasses import dataclass
from typing import Optional, List, Tuple

import cv2
import mediapipe as mp


@dataclass
class FaceMeshResult:
    faces: List[List[Tuple[int, int]]]  # faces -> list of (x,y) points in pixels


class FaceMeshAnalyzer:
    """
    MediaPipe Tasks Face Landmarker wrapper.
    Works on macOS where mediapipe.solutions may not exist.
    """

    def __init__(self, model_path: str = "bruno/models/face_landmarker.task", num_faces: int = 1):
        BaseOptions = mp.tasks.BaseOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions

        self._latest = None

        def _callback(result, output_image, timestamp_ms: int):
            self._latest = result

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.LIVE_STREAM,
            num_faces=num_faces,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            result_callback=_callback,
        )

        self._landmarker = FaceLandmarker.create_from_options(options)

    def close(self):
        try:
            self._landmarker.close()
        except Exception:
            pass

    def analyze_bgr_frame(self, frame_bgr) -> Optional[FaceMeshResult]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms = time.time_ns() // 1_000_000
        self._landmarker.detect_async(mp_image, ts_ms)

        if not self._latest or not getattr(self._latest, "face_landmarks", None):
            return None

        h, w = frame_bgr.shape[:2]
        faces_px: List[List[Tuple[int, int]]] = []
        for face in self._latest.face_landmarks:
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in face]
            faces_px.append(pts)

        return FaceMeshResult(faces=faces_px)

    def draw(self, frame_bgr, result: Optional[FaceMeshResult]):
        if not result:
            return frame_bgr

        # draw every Nth point to keep FPS decent
        step = 6
        for pts in result.faces:
            for i in range(0, len(pts), step):
                x, y = pts[i]
                cv2.circle(frame_bgr, (x, y), 1, (0, 255, 0), -1)
        return frame_bgr
