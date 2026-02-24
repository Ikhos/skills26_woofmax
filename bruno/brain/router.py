import json
from bruno.brain.ollama_client import ollama_chat

ROUTER_MODEL = "qwen2.5:7b"

ROUTER_SYSTEM = """
You are a routing AI for a robot dog assistant.
You must ALWAYS respond in valid JSON only.

Decision rules:
- needs_speaker: almost always true (unless explicitly silent mode).
- needs_vision: true if the request requires image analysis (e.g. skin scan).
- needs_health_reasoning: true if the event mentions symptoms, injury, or health scan.
- needs_vitals: true if the user asks about heart rate, pulse, or vitals.

Explicit intent triggers:
- If transcript includes "check my skin" → needs_vision = true
- If transcript includes "heart rate", "pulse", or "check my vitals" → needs_vitals = true

Return ONLY this JSON schema:

{
  "needs_vision": true/false,
  "needs_speaker": true/false,
  "needs_health_reasoning": true/false,
  "needs_vitals": true/false,
  "confidence": 0.0
}
"""

def route(event: dict):
    transcript = event.get("transcript", "").lower()

    # 🔥 Hard-coded fast path (prevents LLM hallucination)
    if "check my skin" in transcript:
        return {
            "needs_vision": True,
            "needs_speaker": True,
            "needs_health_reasoning": True,
            "needs_vitals": False,
            "confidence": 0.95
        }

    if (
        "heart rate" in transcript
        or "pulse" in transcript
        or "check my vitals" in transcript
    ):
        return {
            "needs_vision": False,
            "needs_speaker": True,
            "needs_health_reasoning": True,
            "needs_vitals": True,
            "confidence": 0.95
        }

    # 🧠 Otherwise use LLM routing
    prompt = f"""{ROUTER_SYSTEM}

SENSOR_EVENT_JSON:
{json.dumps(event, indent=2)}
"""
    txt = ollama_chat(ROUTER_MODEL, prompt)

    fallback = {
        "needs_vision": False,
        "needs_speaker": True,
        "needs_health_reasoning": False,
        "needs_vitals": False,
        "confidence": 0.25
    }

    try:
        data = json.loads(txt)

        required_keys = [
            "needs_vision",
            "needs_speaker",
            "needs_health_reasoning",
            "needs_vitals",
            "confidence"
        ]

        for k in required_keys:
            if k not in data:
                return fallback

        return data

    except Exception:
        return fallback