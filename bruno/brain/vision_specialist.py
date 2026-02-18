import json
from bruno.brain.ollama_client import ollama_chat, _frame_to_jpeg_b64

VISION_MODEL = "llava:7b"

SYSTEM = """
You are a visual perception AI.
Describe the scene briefly.
Return JSON:

{
  "scene": "short description",
  "notable": ["..."]
}
"""

def analyze_scene(frame):
    img_b64 = _frame_to_jpeg_b64(frame)
    txt = ollama_chat(VISION_MODEL, SYSTEM, image_b64=img_b64)

    try:
        return json.loads(txt)
    except Exception:
        return {"scene": txt[:200], "notable": []}
