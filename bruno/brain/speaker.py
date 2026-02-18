import json
from bruno.brain.ollama_client import ollama_chat

SPEAKER_MODEL = "llama3.1:8b"

SYSTEM = """
You are BRUNO, a helpful robot dog like Baymax.
Speak calmly and clearly. Short sentences.
No medical diagnosis. Encourage safety and professional help if severe.
Return ONLY JSON:

{
  "say": "speech output",
  "ask_user": "",
  "confidence": 0.0
}
"""

def speak_response(event: dict, vision_data=None):
    prompt = f"""{SYSTEM}

SENSOR_EVENT_JSON:
{json.dumps(event, indent=2)}

VISION_DATA:
{json.dumps(vision_data or {}, indent=2)}
"""
    txt = ollama_chat(SPEAKER_MODEL, prompt)

    try:
        return json.loads(txt)
    except Exception:
        return {"say": txt[:240], "ask_user": "", "confidence": 0.4}
