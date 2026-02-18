import time
from fastapi import FastAPI
from bruno.bus.messages import PerceptionEvent, BrainCommand

app = FastAPI()

def _scene_phrase(evt: PerceptionEvent) -> str:
    counts = (evt.scene_summary or {}).get("counts", {})
    objs = list(counts.keys())[:6]
    if not objs:
        return "I do not see any clear objects right now."
    return "I see " + ", ".join(objs) + "."

def brain_reply(evt: PerceptionEvent) -> BrainCommand:
    notes = evt.notes or {}
    user_text = (notes.get("user_text") or "").strip()
    t = user_text.lower()

    health = notes.get("health")
    scene = _scene_phrase(evt)

    # 1) If health data provided (result of scan)
    if health:
        sym = health.get("symmetry_score")
        face_ok = bool(health.get("face_detected", False))

        if not face_ok:
            return BrainCommand(
                say="I could not clearly detect your face. Please face me in better lighting and try again.",
                actions=["offer_rescan"],
                safety={"stop_motion": False},
            )

        if sym is None:
            return BrainCommand(
                say="Scan complete. I captured basic face signals. Would you like another scan or a progress check?",
                actions=["offer_rescan", "offer_progress_check"],
                safety={"stop_motion": False},
            )

        if sym > 0.92:
            say = f"Scan complete. Your symmetry score is {sym:.2f}. I did not detect obvious asymmetry signals. Would you like a progress check or another scan?"
        else:
            say = f"Scan complete. Your symmetry score is {sym:.2f}. This could be worth rechecking with better lighting. Would you like another scan?"
        return BrainCommand(say=say, actions=["offer_progress_check", "offer_rescan"], safety={"stop_motion": False})

    # 2) If user said something (voice/typed reply)
    if user_text:
        # greetings
        if any(k in t for k in ["hi", "hello", "hey", "yo", "wassup"]):
            return BrainCommand(
                say="Hello. How can I help you today? You can say health scan, safety scan, or what do you see.",
                actions=["offer_health_scan", "offer_safety_scan", "offer_scene"],
                safety={"stop_motion": False},
            )

        # ask what it sees
        if any(k in t for k in ["what do you see", "what do u see", "what do you detect", "what's around", "scan room", "look around"]):
            return BrainCommand(
                say=scene + " Would you like a health scan or a safety scan?",
                actions=["offer_health_scan", "offer_safety_scan"],
                safety={"stop_motion": False},
            )

        # request health scan
        if any(k in t for k in ["health", "progress", "scan me", "check me", "checkup", "wound", "symmetry"]):
            return BrainCommand(
                say="Understood. Please face me in good lighting. I am ready to scan now.",
                actions=["do_health_scan"],
                safety={"stop_motion": False},
            )

        # request safety scan
        if any(k in t for k in ["safety", "hazard", "danger", "obstacle", "clear path"]):
            return BrainCommand(
                say="Understood. I will focus on environment safety. Please show me the area and ask again.",
                actions=["do_safety_scan"],
                safety={"stop_motion": False},
            )

        # request identity
        if any(k in t for k in ["identify", "who am i", "my name", "unlock", "log in", "login"]):
            return BrainCommand(
                say="Understood. I can identify you for personalized history. Please complete identity verification.",
                actions=["do_identify"],
                safety={"stop_motion": False},
            )

        # default reply to unknown text
        return BrainCommand(
            say="I understand. Would you like a health scan, a safety scan, or a description of what I see?",
            actions=["offer_health_scan", "offer_safety_scan", "offer_scene"],
            safety={"stop_motion": False},
        )

    # 3) No user text: default proactive line
    return BrainCommand(
        say=scene + " Would you like a health scan or an environment safety scan? If you want personalized history, I can identify you.",
        actions=["offer_health_scan", "offer_safety_scan", "offer_identify"],
        safety={"stop_motion": False},
    )

@app.post("/event", response_model=BrainCommand)
def handle_event(evt: PerceptionEvent):
    return brain_reply(evt)

@app.get("/health")
def healthcheck():
    return {"ok": True, "ts": time.strftime("%Y-%m-%dT%H-%M-%S")}
