# bruno/brain/speaker.py

import json
from bruno.voice.tts import speak


class BrunoBrain:
    def __init__(self):
        pass

    def build_context(self, scene_summary, user_id=None, health_data=None):
        context = {
            "scene": scene_summary,
            "user_id": user_id,
            "health": health_data
        }
        return context

    def generate_response(self, context):
        scene = context.get("scene", {})
        user = context.get("user_id")

        if not scene or not scene.get("counts"):
            return "I do not detect significant objects at this time."

        objects = list(scene["counts"].keys())

        base = f"I see {', '.join(objects)}."

        if user:
            return base + f" Hello {user}. Would you like a health check or environment safety scan?"
        else:
            return base + " Would you like me to identify you for personalized assistance?"


# ---------------------------------------------------
# 🔥 ADD THIS FUNCTION (DO NOT REMOVE YOUR CLASS)
# ---------------------------------------------------

def speak_response(event, vision_data=None, routing=None):
    """
    Adapter layer so orchestrator can call speaker cleanly.
    """

    brain = BrunoBrain()

    transcript = event.get("transcript", "")

    # Vision data expected format:
    # { "counts": {...} }
    scene_summary = vision_data if vision_data else {}

    context = brain.build_context(
        scene_summary=scene_summary,
        user_id=None,
        health_data=None
    )

    response = brain.generate_response(context)

    speak(response)

    return {"say": response}