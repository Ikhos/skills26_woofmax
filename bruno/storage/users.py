import json
from pathlib import Path
import time

def ensure_user_dirs(users_root: str, user_id: str):
    base = Path(users_root) / user_id
    (base / "scans").mkdir(parents=True, exist_ok=True)
    (base / "face" / "enroll").mkdir(parents=True, exist_ok=True)
    (base / "auth").mkdir(parents=True, exist_ok=True)
    return base

def save_scan_json(users_root: str, user_id: str, payload: dict) -> str:
    base = ensure_user_dirs(users_root, user_id)
    ts = payload.get("ts") or time.strftime("%Y-%m-%dT%H-%M-%S")
    path = base / "scans" / f"{ts}.json"
    path.write_text(json.dumps(payload, indent=2))
    return str(path)
