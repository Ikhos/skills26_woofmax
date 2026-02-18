from bruno.brain.router import route
from bruno.brain.speaker import speak_response
from bruno.brain.vision_specialist import analyze_scene

def think_sync(event: dict, frame=None, debug=True):
    routing = route(event)

    if debug:
        print("----- ROUTER OUTPUT -----")
        print(routing)

    vision_data = None
    if routing.get("needs_vision") and frame is not None:
        vision_data = analyze_scene(frame)
        if debug:
            print("----- VISION OUTPUT -----")
            print(vision_data)

    needs_speaker = routing.get("needs_speaker", True)

    if needs_speaker:
        out = speak_response(event, vision_data)
        if debug:
            print("----- SPEAKER OUTPUT -----")
            print(out)
        return out

    return {"say": "", "ask_user": "", "confidence": 0.0}
