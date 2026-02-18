from typing import Dict, Any, List

# Simple category mapping (expand any time)
FURNITURE = {"chair", "couch", "bed", "dining table", "tv", "laptop", "keyboard", "mouse"}
STRUCTURES = {"door", "potted plant", "sink", "toilet"}
OBJECTS = {"bottle", "cup", "book", "backpack", "handbag", "cell phone"}
POTENTIAL_HAZARDS = {"knife", "scissors", "bottle"}  # placeholder; expand later

def summarize_scene(detections: List[Dict[str, Any]]) -> Dict[str, Any]:
    labels = [d["label"] for d in detections]
    counts = {}
    for l in labels:
        counts[l] = counts.get(l, 0) + 1

    buckets = {"furniture": [], "structures": [], "objects": [], "hazards": [], "other": []}

    for d in detections:
        lab = d["label"]
        if lab in FURNITURE:
            buckets["furniture"].append(d)
        elif lab in STRUCTURES:
            buckets["structures"].append(d)
        elif lab in OBJECTS:
            buckets["objects"].append(d)
        elif lab in POTENTIAL_HAZARDS:
            buckets["hazards"].append(d)
        else:
            buckets["other"].append(d)

    # Simple hazard messages (non-medical, general safety)
    hazard_msgs = []
    if buckets["hazards"]:
        top = sorted(buckets["hazards"], key=lambda x: x["confidence"], reverse=True)[:3]
        hazard_msgs.append("Potential hazard objects detected: " + ", ".join([x["label"] for x in top]) + ".")

    return {
        "ok": True,
        "counts": counts,
        "buckets": {k: [{"label": x["label"], "confidence": x["confidence"], "bbox_xyxy": x["bbox_xyxy"]} for x in v] for k, v in buckets.items()},
        "hazard_notes": hazard_msgs
    }
