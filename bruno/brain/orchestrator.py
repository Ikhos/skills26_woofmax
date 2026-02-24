from bruno.brain.router import route
from bruno.brain.speaker import speak_response
from bruno.brain.vision_specialist import analyze_scene
from bruno.health.health_specialist import analyze_vitals


def think_sync(event: dict, frame=None, cap=None, face_box=None, debug=True):
    routing = route(event)

    if debug:
        print("----- ROUTER OUTPUT -----")
        print(routing)

    vision_data = None
    vitals_data = None

    # --------------------------------------------
    # 🔒 ENFORCE LOGICAL CONSISTENCY
    # If any health or vision analysis is needed,
    # we MUST speak back.
    # --------------------------------------------
    if (
        routing.get("needs_vision")
        or routing.get("needs_health_reasoning")
        or routing.get("needs_vitals")
    ):
        routing["needs_speaker"] = True
    # --------------------------------------------

    # 🧴 Acne / vision scan
    if routing.get("needs_vision") and frame is not None:
        vision_data = analyze_scene(frame, routing=routing)

        if debug:
            print("----- VISION OUTPUT -----")
            print(vision_data)

    # ❤️ Heart rate / vitals
    if (
        routing.get("needs_vitals")
        and cap is not None
        and face_box is not None
    ):
        vitals_data = analyze_vitals(cap, face_box)

        if debug:
            print("----- VITALS OUTPUT -----")
            print(vitals_data)

    needs_speaker = routing.get("needs_speaker", True)

    if needs_speaker:
        out = speak_response(
            event,
            vision_data=vision_data,
            vitals_data=vitals_data,
            routing=routing
        )

        if debug:
            print("----- SPEAKER OUTPUT -----")
            print(out)

        return out

    return {"say": "", "ask_user": "", "confidence": 0.0}