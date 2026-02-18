from typing import Dict, Any, List
import numpy as np
from ultralytics import YOLO

class YoloDetector:
    def __init__(self, weights: str = "yolov8n.pt", conf: float = 0.35):
        self.model = YOLO(weights)
        self.conf = conf

    def predict(self, bgr_frame) -> Dict[str, Any]:
        # ultralytics expects RGB
        rgb = bgr_frame[:, :, ::-1]
        results = self.model.predict(rgb, conf=self.conf, verbose=False)

        out: List[Dict[str, Any]] = []
        r0 = results[0]
        names = r0.names

        if r0.boxes is None:
            return {"ok": True, "detections": []}

        for b in r0.boxes:
            cls = int(b.cls[0].item())
            conf = float(b.conf[0].item())
            xyxy = [float(x) for x in b.xyxy[0].tolist()]
            out.append({
                "label": names.get(cls, str(cls)),
                "confidence": conf,
                "bbox_xyxy": xyxy
            })

        # Sort highest confidence first
        out.sort(key=lambda d: d["confidence"], reverse=True)
        return {"ok": True, "detections": out}
