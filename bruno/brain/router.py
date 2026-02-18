import json
from bruno.brain.ollama_client import ollama_chat

ROUTER_MODEL = "qwen2.5:7b"

ROUTER_SYSTEM = """
You are a routing AI for a robot dog assistant.
You must ALWAYS respond in valid JSON only.

Decision rules:
- needs_speaker: almost always true (unless explicitly silent mode).
- needs_vision: true if the event lacks enough visual context.
- needs_health_reasoning: true if the event mentions symptoms, injury, or scan.

Return ONLY this JSON schema:

{
  "needs_vision": true/false,
  "needs_speaker": true/false,
  "needs_health_reasoning": true/false,
  "confidence": 0.0
}
"""

def route(event: dict):
    prompt = f"""{ROUTER_SYSTEM}

SENSOR_EVENT_JSON:
{json.dumps(event, indent=2)}
"""
    txt = ollama_chat(ROUTER_MODEL, prompt)

    fallback = {
        "needs_vision": False,
        "needs_speaker": True,
        "needs_health_reasoning": False,
        "confidence": 0.25
    }

    try:
        data = json.loads(txt)
        for k in ["needs_vision", "needs_speaker", "needs_health_reasoning", "confidence"]:
            if k not in data:
                return fallback
        return data
    except Exception:
        return fallback
