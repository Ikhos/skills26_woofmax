from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class Detection(BaseModel):
    label: str
    confidence: float = 1.0
    track_id: Optional[int] = None
    box: List[int]  # [x1,y1,x2,y2]

class PerceptionEvent(BaseModel):
    ts: str
    device_id: str
    user_id: Optional[str] = None
    scene_summary: Dict[str, Any]
    detections: List[Detection] = []
    face_detected: bool = False
    notes: Dict[str, Any] = {}

class BrainCommand(BaseModel):
    say: str
    actions: List[str] = []
    safety: Dict[str, Any] = {}
