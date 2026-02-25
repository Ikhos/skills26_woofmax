"""
Stub YOLOTracker when Ultralytics/PyTorch is disabled (e.g. BRUNO_DISABLE_YOLO=1 on Raspberry Pi).
Returns no detections.
"""


class YOLOTracker:
    """No-op object detection when YOLO is unavailable or disabled on Pi."""

    def __init__(self, model_name="yolov8n.pt"):
        pass

    def track(self, frame):
        return {"detections": []}
