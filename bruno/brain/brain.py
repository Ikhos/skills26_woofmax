import json

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
