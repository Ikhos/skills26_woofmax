import base64
import requests
import cv2

OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"

def _frame_to_jpeg_b64(frame) -> str:
    """
    Convert an OpenCV BGR frame to base64 JPEG.
    """
    ok, buf = cv2.imencode(".jpg", frame)
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return base64.b64encode(buf.tobytes()).decode("utf-8")

def ollama_chat(model: str, prompt: str, image_b64: str | None = None) -> str:
    """
    Call Ollama /api/generate. If image_b64 is provided, attaches it for vision models like LLaVA.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }
    if image_b64:
        payload["images"] = [image_b64]

    r = requests.post(OLLAMA_GENERATE_URL, json=payload, timeout=120)
    r.raise_for_status()
    return (r.json().get("response") or "").strip()
