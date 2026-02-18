from typing import List, Tuple, Dict, Any

# Indices match MediaPipe Face Landmarker landmark order (468 points).
MOUTH_LEFT = 61
MOUTH_RIGHT = 291
L_EYE_UP = 159
L_EYE_LOW = 145
R_EYE_UP = 386
R_EYE_LOW = 374

def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))

def compute_symmetry(pts: List[Tuple[int, int]], frame_h: int) -> Dict[str, Any]:
    """
    pts: one face's landmarks as [(x,y), ...] in pixels.
    Returns symmetry score 0..1 and flags.
    """
    if len(pts) <= max(MOUTH_LEFT, MOUTH_RIGHT, L_EYE_UP, L_EYE_LOW, R_EYE_UP, R_EYE_LOW):
        return {"ok": False, "reason": "not_enough_points"}

    mouth_l = pts[MOUTH_LEFT]
    mouth_r = pts[MOUTH_RIGHT]
    le_up = pts[L_EYE_UP]
    le_lo = pts[L_EYE_LOW]
    re_up = pts[R_EYE_UP]
    re_lo = pts[R_EYE_LOW]

    mouth_dy = mouth_l[1] - mouth_r[1]  # + means left mouth corner is lower
    mouth_asym = abs(mouth_dy) / max(1.0, 0.12 * frame_h)

    le_open = abs(le_lo[1] - le_up[1])
    re_open = abs(re_lo[1] - re_up[1])
    eye_asym = abs(le_open - re_open) / max(1.0, 0.06 * frame_h)

    mouth_asym = _clamp(mouth_asym)
    eye_asym = _clamp(eye_asym)

    score = 1.0 - (0.6 * mouth_asym + 0.4 * eye_asym)
    score = _clamp(score)

    flags = []
    if abs(mouth_dy) > 0.03 * frame_h:
        flags.append("mouth_left_lower" if mouth_dy > 0 else "mouth_right_lower")
    if abs(le_open - re_open) > 0.015 * frame_h:
        flags.append("eye_left_more_closed" if le_open < re_open else "eye_right_more_closed")

    return {
        "ok": True,
        "symmetry_score": score,
        "flags": flags,
        "details": {
            "mouth_dy_px": int(mouth_dy),
            "left_eye_open_px": int(le_open),
            "right_eye_open_px": int(re_open),
        },
    }
