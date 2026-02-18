from typing import Dict, Any, List
from ultralytics import YOLO

class YoloTracker:
    def __init__(self, weights: str = "yolov8n.pt", conf: float = 0.35):
        self.model = YOLO(weights)
        self.conf = conf

    def track(self, bgr_frame) -> Dict[str, Any]:
        rgb = bgr_frame[:, :, ::-1]
        results = self.model.track(rgb, conf=self.conf, persist=True, verbose=False)
        r0 = results[0]
        names = r0.names

        dets: List[Dict[str, Any]] = []
        if r0.boxes is None:
            return {"ok": True, "detections": dets}

        for b in r0.boxes:
            cls = int(b.cls[0].item())
            conf = float(b.conf[0].item())
            xyxy = [float(x) for x in b.xyxy[0].tolist()]

            track_id = None
            if hasattr(b, "id") and b.id is not None:
                try:
                    track_id = int(b.id[0].item())
                except Exception:
                    track_id = None

            dets.append({
                "label": names.get(cls, str(cls)),
                "confidence": conf,
                "bbox_xyxy": xyxy,
                "track_id": track_id
            })

        dets.sort(key=lambda d: d["confidence"], reverse=True)
        return {"ok": True, "detections": dets}
