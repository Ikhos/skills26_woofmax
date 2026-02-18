import os
import json
import hashlib
from pathlib import Path

def _hash_pin(pin: str, salt: str) -> str:
    return hashlib.sha256((salt + pin).encode("utf-8")).hexdigest()

def pin_exists(users_root: str, user_id: str) -> bool:
    user_id = user_id.strip().lower()
    path = Path(users_root) / user_id / "auth" / "pin.json"
    return path.exists()

def set_pin(users_root: str, user_id: str, pin: str):
    user_id = user_id.strip().lower()
    root = Path(users_root) / user_id
    (root / "auth").mkdir(parents=True, exist_ok=True)

    salt = os.urandom(16).hex()
    data = {"salt": salt, "hash": _hash_pin(pin, salt)}
    (root / "auth" / "pin.json").write_text(json.dumps(data, indent=2))

def verify_pin(users_root: str, user_id: str, pin: str) -> bool:
    user_id = user_id.strip().lower()
    path = Path(users_root) / user_id / "auth" / "pin.json"
    if not path.exists():
        return False
    data = json.loads(path.read_text())
    return _hash_pin(pin, data["salt"]) == data["hash"]
