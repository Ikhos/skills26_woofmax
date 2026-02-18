import os
import json
from typing import Dict, Any

def _path(users_root: str, user_id: str) -> str:
    return os.path.join(users_root, user_id, "object_overrides.json")

def load_overrides(users_root: str, user_id: str) -> Dict[str, Any]:
    p = _path(users_root, user_id)
    if not os.path.isfile(p):
        return {}
    try:
        with open(p, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def save_overrides(users_root: str, user_id: str, data: Dict[str, Any]):
    os.makedirs(os.path.join(users_root, user_id), exist_ok=True)
    with open(_path(users_root, user_id), "w") as f:
        json.dump(data, f, indent=2)

def set_override(users_root: str, user_id: str, track_id: int, correct_label: str, original_label: str):
    data = load_overrides(users_root, user_id)
    data[str(track_id)] = {"correct_label": correct_label, "original_label": original_label}
    save_overrides(users_root, user_id, data)

def apply_overrides(users_root: str, user_id: str, detections):
    data = load_overrides(users_root, user_id)
    if not data:
        return detections

    out = []
    for d in detections:
        tid = d.get("track_id")
        if tid is not None and str(tid) in data:
            nd = dict(d)
            nd["label_original"] = nd["label"]
            nd["label"] = data[str(tid)]["correct_label"]
            nd["corrected_by_user"] = True
            out.append(nd)
        else:
            out.append(d)
    return out
