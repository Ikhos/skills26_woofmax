from bruno.brain.orchestrator import think_sync

event = {
    "faces": [{"user_id": "anish", "confidence": 0.82}],
    "detections": ["person", "chair"],
    "pose": {"detected": True},
    "autopilot": True,
    "user_text": "BRUNO, what do you see and what should I do?"
}

out = think_sync(event, debug=True)

print("----- FINAL OUTPUT -----")
print(out)
