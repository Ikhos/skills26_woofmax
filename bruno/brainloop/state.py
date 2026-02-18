from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import time

Box = Tuple[int, int, int, int]

@dataclass
class PerceptionState:
    ts: float
    objects: List[str]
    people: List[Dict[str, Any]]  # {box, name, recognized}
    authorized_user: Optional[str]
    pose: Dict[str, Any]          # pass-through

def _unique_labels(detections: List[Dict[str, Any]]) -> List[str]:
    seen = set()
    out = []
    for d in detections or []:
        lab = d.get("label")
        if not lab:
            continue
        if lab not in seen:
            seen.add(lab)
            out.append(lab)
    return out

def build_state(
    detections: List[Dict[str, Any]],
    people: List[Dict[str, Any]],
    pose_info: Dict[str, Any],
    authorized_user: Optional[str],
) -> PerceptionState:
    ts = time.time()
    objects = _unique_labels(detections)

    # People already built in vision loop


    return PerceptionState(
        ts=ts,
        objects=objects,
        people=people,
        authorized_user=authorized_user,
        pose=pose_info or {},
    )
