import cv2

def draw_detections(frame, detections, max_show: int = 5):
    h, w = frame.shape[:2]
    for det in detections[:max_show]:
        x1, y1, x2, y2 = det["bbox_xyxy"]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
        label = f'{det["label"]} {det["confidence"]:.2f}'
        cv2.putText(frame, label, (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return frame
