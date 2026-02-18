from ultralytics import YOLO
import numpy as np


class YOLOTracker:
    def __init__(self, model_name="yolov8n.pt"):
        print("BRUNO: Loading YOLO model...")
        self.model = YOLO(model_name)
        print("BRUNO: YOLO ready.")

    def track(self, frame):
        results = self.model.track(frame, persist=True, verbose=False)

        detections = []

        for r in results:
            if r.boxes is None:
                continue

            boxes = r.boxes.xyxy.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy()
            ids = r.boxes.id

            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i].astype(int)
                label = self.model.names[int(classes[i])]
                track_id = int(ids[i]) if ids is not None else None

                detections.append({
                    "box": (x1, y1, x2, y2),
                    "label": label,
                    "track_id": track_id
                })

        return {"detections": detections}
