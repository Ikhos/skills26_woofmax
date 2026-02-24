import json
import re
from bruno.brain.ollama_client import ollama_chat

SPEAKER_MODEL = "llama3.1:8b"

SYSTEM = """
You are BRUNO, a calm and helpful robot assistant.
Speak clearly and briefly.
No medical diagnosis. Encourage professional help if severe.
Return ONLY JSON:

{
  "say": "speech output",
  "ask_user": "",
  "confidence": 0.0
}
"""


def clean_llm_json(text: str):
    """
    Removes ```json blocks if model wraps output.
    """
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"```json", "", text)
        text = text.replace("```", "").strip()

    return text


def speak_response(event: dict, vision_data=None, vitals_data=None, routing=None):
    """
    Handles vitals first.
    Then acne result.
    Falls back to LLM if needed.
    """

    # ----------------------------------
    # ❤️ DIRECT HEART RATE RESPONSE
    # ----------------------------------
    if vitals_data and vitals_data.get("heart_rate"):
        hr = vitals_data["heart_rate"]
        confidence = vitals_data.get("confidence", 0.8)

        return {
            "say": f"Your estimated heart rate is {hr} beats per minute.",
            "ask_user": "",
            "confidence": confidence
        }

    # ----------------------------------
    # 🧴 DIRECT ACNE CLASSIFIER RESPONSE
    # ----------------------------------
    if vision_data and vision_data.get("acne_stage"):
        stage = vision_data["acne_stage"]
        confidence = vision_data.get("confidence", 0.8)

        return {
            "say": f"I have analyzed your skin. Your current acne stage is {stage}.",
            "ask_user": "",
            "confidence": confidence
        }

    # ----------------------------------
    # 🔁 FALLBACK TO LLM SPEAKER
    # ----------------------------------
    prompt = f"""{SYSTEM}

SENSOR_EVENT_JSON:
{json.dumps(event, indent=2)}

VISION_DATA:
{json.dumps(vision_data or {}, indent=2)}

VITALS_DATA:
{json.dumps(vitals_data or {}, indent=2)}
"""

    try:
        txt = ollama_chat(SPEAKER_MODEL, prompt)
        txt = clean_llm_json(txt)
        return json.loads(txt)

    except Exception as e:
        print("Speaker error:", e)
        return {
            "say": "I am having trouble generating a response.",
            "ask_user": "",
            "confidence": 0.3
        }