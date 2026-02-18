from typing import Dict, Any, List, Tuple
from ultralytics import YOLO

def _iou(a: List[float], b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter + 1e-9
    return inter / union

def _nms(dets: List[Dict[str, Any]], iou_thr: float = 0.55) -> List[Dict[str, Any]]:
    # NMS within same label only
    dets = sorted(dets, key=lambda d: d["confidence"], reverse=True)
    kept = []
    for d in dets:
        ok = True
        for k in kept:
            if d["label"] == k["label"] and _iou(d["bbox_xyxy"], k["bbox_xyxy"]) > iou_thr:
                ok = False
                break
        if ok:
            kept.append(d)
    return kept

class YoloStack:
    """
    Runs multiple YOLO models and merges detections.
    This is how you scale to "detect lots of stuff" by stacking specialized weights.
    """

    def __init__(self, weights: List[str], conf: float = 0.35, iou_merge: float = 0.55):
        self.models: List[Tuple[str, YOLO]] = [(w, YOLO(w)) for w in weights]
        self.conf = conf
        self.iou_merge = iou_merge

    def predict(self, bgr_frame) -> Dict[str, Any]:
        rgb = bgr_frame[:, :, ::-1]
        merged: List[Dict[str, Any]] = []

        for wname, model in self.models:
            results = model.predict(rgb, conf=self.conf, verbose=False)
            r0 = results[0]
            names = r0.names

            if r0.boxes is None:
                continue

            for b in r0.boxes:
                cls = int(b.cls[0].item())
                conf = float(b.conf[0].item())
                xyxy = [float(x) for x in b.xyxy[0].tolist()]
                merged.append({
                    "label": names.get(cls, str(cls)),
                    "confidence": conf,
                    "bbox_xyxy": xyxy,
                    "source_model": wname
                })

        merged = _nms(merged, iou_thr=self.iou_merge)
        merged.sort(key=lambda d: d["confidence"], reverse=True)

        return {"ok": True, "detections": merged, "models": [w for w, _ in self.models]}
