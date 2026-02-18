import os
import json
from typing import List, Dict

def load_recent_scans(users_root: str, user_id: str, limit: int = 5) -> List[Dict]:
    scans_dir = os.path.join(users_root, user_id, "scans")
    if not os.path.isdir(scans_dir):
        return []

    files = sorted(os.listdir(scans_dir), reverse=True)
    results = []

    for f in files[:limit]:
        try:
            with open(os.path.join(scans_dir, f), "r") as fp:
                data = json.load(fp)
                if data.get("scan") and data["scan"].get("ok"):
                    results.append(data)
        except Exception:
            continue

    return results


def symmetry_trend(users_root: str, user_id: str) -> Dict:
    scans = load_recent_scans(users_root, user_id, limit=5)

    if len(scans) < 2:
        return {"status": "insufficient_data"}

    scores = [s["scan"]["symmetry_score"] for s in scans]

    latest = scores[0]
    previous_avg = sum(scores[1:]) / len(scores[1:])
    delta = latest - previous_avg

    if abs(delta) < 0.02:
        classification = "stable"
    elif delta > 0:
        classification = "improving"
    else:
        classification = "deteriorating"

    return {
        "status": "ok",
        "latest_score": latest,
        "previous_avg": previous_avg,
        "delta": delta,
        "classification": classification
    }
