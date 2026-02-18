import requests
from bruno.bus.messages import PerceptionEvent, BrainCommand

def send_event(brain_url: str, evt: PerceptionEvent) -> BrainCommand:
    r = requests.post(f"{brain_url}/event", json=evt.model_dump(), timeout=8)
    r.raise_for_status()
    return BrainCommand(**r.json())
