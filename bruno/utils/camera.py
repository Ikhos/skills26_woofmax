import cv2

def open_camera(index: int = 0, width: int = 640, height: int = 480):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {index}. Try 0, 1, 2...")

    # Try to set resolution (not guaranteed depending on camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap

def read_frame(cap):
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    return frame

def close_camera(cap):
    try:
        cap.release()
    except Exception:
        pass
    cv2.destroyAllWindows()
