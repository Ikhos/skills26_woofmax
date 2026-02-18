import cv2
import time

from bruno.utils.camera import open_camera, read_frame, close_camera
from bruno.perception.yolo import YOLOTracker
from bruno.perception.pose import PoseAnalyzer, draw_pose_skeleton_in_bbox
from bruno.auth.face_embed import FaceEmbedID
from bruno.auth.pin import verify_pin, set_pin, pin_exists
from bruno.storage.users import ensure_user_dirs, save_scan_json
from bruno.voice.tts import speak
from bruno.brain.orchestrator import think_sync
from bruno.brainloop.state import build_state
from bruno.brainloop.risk import score_risk
from bruno.brainloop.autopilot import Autopilot

USERS_ROOT = "data/users"


def draw_boxes(frame, detections, name_map=None):
    for d in detections:
        x1, y1, x2, y2 = d["box"]
        label = d["label"]
        tid = d.get("track_id")
        tag = f"{label}" if tid is None else f"{label} #{tid}"

        # Draw green box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw object label
        cv2.putText(
            frame,
            tag,
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        # Draw recognized name above PERSON box
        if name_map and label == "person":
            uid = name_map.get(tuple([x1, y1, x2, y2]))
            if uid:
                cv2.putText(
                    frame,
                    uid,
                    (x1, max(20, y1 - 30)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 255, 0),
                    2,
                )

    return frame



def center_of(box):
    x1,y1,x2,y2 = box
    return ((x1+x2)//2, (y1+y2)//2)

def point_in_box(px, py, box):
    x1,y1,x2,y2 = box
    return x1 <= px <= x2 and y1 <= py <= y2

def assign_names_to_person_boxes(detections, face_matches):
    # Returns dict: person_box_tuple -> user_id
    persons = [d for d in detections if d.get("label") == "person" and "box" in d]
    mapping = {}
    if not persons or not face_matches:
        return mapping

    for fm in face_matches:
        uid = fm.get("user_id")
        if not uid:
            continue
        fx1,fy1,fx2,fy2 = fm["bbox"]
        cx, cy = center_of([fx1,fy1,fx2,fy2])

        # pick the person box that contains the face center, prefer smallest area (closest fit)
        candidates = []
        for d in persons:
            pb = d["box"]
            if point_in_box(cx, cy, pb):
                x1,y1,x2,y2 = pb
                area = max(0,(x2-x1))*max(0,(y2-y1))
                candidates.append((area, tuple(pb)))
        if candidates:
            candidates.sort(key=lambda x: x[0])
            mapping[candidates[0][1]] = uid
    return mapping


def best_person_box(detections):
    persons = [d for d in detections if d.get("label") == "person" and "box" in d]
    if not persons:
        return None

    def area(d):
        x1, y1, x2, y2 = d["box"]
        return max(0, (x2 - x1)) * max(0, (y2 - y1))

    return max(persons, key=area)["box"]


def main():
    print("🐶 BRUNO booting...")
    cap = open_camera(index=0)
    yolo = YOLOTracker()
    pose = PoseAnalyzer()
    faceid = FaceEmbedID(USERS_ROOT)
    autopilot = Autopilot()
    autopilot_enabled = True

    print("BRUNO: Keys:")
    print("  n = new profile (create user + PIN)")
    print("  e = enroll face (for a user)")
    print("  u = unlock (face match -> PIN fallback)")
    print("  h = save scan (requires unlock)")
    print("  a = toggle autopilot")
    print("  q = quit")

    # Loop tuning
    frame_count = 0
    OBJ_EVERY_N = 4
    ID_EVERY_N = 4
    POSE_EVERY_N = 3

    # --- Identity smoothing ---
    ID_GRACE_SEC = 2.5          # keep last known name for a track for this long
    SPEAK_COOLDOWN_SEC = 2.0    # prevent spam
    track_identity = {}         # track_id -> {"name": str|None, "last_seen": float, "conf": float}
    last_spoken = {"t": 0.0, "name": None}


    # Detections cache
    last_detections = []

    # Face matches cache (for always-on display)
    last_face_matches = []

    # Pose cache (we keep fall_score in the data, but we do NOTHING with it)
    last_pose = {"detected": False, "fall_score": 0.0, "keypoints": None, "notes": {}}

    # Display identity (UI only, no access)
    display_user = None
    display_until = 0.0
    DISPLAY_TTL = 3.0
    stable_user = None
    stable_hits = 0
    STABLE_REQUIRED = 2

    last_spoken_user = None

    # Authorization identity (data access)
    authorized_user = None
    auth_until = 0.0
    AUTH_TTL = 25.0

    while True:
        frame = read_frame(cap)
        if frame is None:
            continue

        frame_count += 1

        # Object detection
        if frame_count % OBJ_EVERY_N == 0:
            try:
                result = yolo.track(frame)
                last_detections = result.get("detections", [])
            except Exception:
                pass

        # Background FaceID (display only, multi-face) - update cache
        if frame_count % ID_EVERY_N == 0:
            try:
                last_face_matches = faceid.match_faces(frame, threshold=0.35)
            except Exception:
                last_face_matches = []
# Background Pose
        if frame_count % POSE_EVERY_N == 0:
            try:
                ts_ms = int(time.time() * 1000)
                pr = pose.analyze_bgr_frame(frame, ts_ms)
                last_pose = {
                    "detected": pr.detected,
                    "fall_score": pr.fall_score,   # kept for later if you want, but unused now
                    "keypoints": pr.keypoints,
                    "notes": pr.notes
                }
            except Exception:
                pass

        # Draw boxes + name tag
        name_map = assign_names_to_person_boxes(last_detections, last_face_matches)

        # --- Smooth identities across frames using track_id ---
        now = time.time()

        # Update track_identity when we have a name for a person box
        for d in last_detections:
            if d.get("label") != "person":
                continue
            tid = d.get("track_id")
            if tid is None:
                continue
            box = tuple(d["box"])
            nm = name_map.get(box)
            if nm:
                track_identity[tid] = {"name": nm, "last_seen": now, "conf": 1.0}
            else:
                # refresh last_seen if track exists but no new match this frame
                if tid in track_identity:
                    track_identity[tid]["last_seen"] = track_identity[tid].get("last_seen", now)

        # Build a smoothed map: if no name this frame, keep last known name for grace period
        smoothed_name_map = {}
        for d in last_detections:
            if d.get("label") != "person":
                continue
            box = tuple(d["box"])
            tid = d.get("track_id")
            nm = name_map.get(box)
            if nm:
                smoothed_name_map[box] = nm
            elif tid is not None and tid in track_identity:
                age = now - float(track_identity[tid].get("last_seen", 0.0))
                if age <= ID_GRACE_SEC:
                    smoothed_name_map[box] = track_identity[tid].get("name")

        name_map = smoothed_name_map

        # --- Shared people list (UI + Brain use same recognition source) ---
        people = []
        for d in last_detections:
            if d.get("label") != "person":
                continue
            box = tuple(d["box"])
            tid = d.get("track_id")
            nm = name_map.get(box)
            people.append({
                "track_id": tid,
                "name": nm,
                "recognized": bool(nm),
            })

        frame = draw_boxes(frame, last_detections, name_map=name_map)

        # Stick figure ONLY inside PERSON bbox
        person_box = best_person_box(last_detections)
        if last_pose.get("detected") and last_pose.get("keypoints") and person_box:
            try:
                draw_pose_skeleton_in_bbox(frame, last_pose["keypoints"], person_box, min_vis=0.55)
            except Exception:
                pass


        # BrainLoop: build state -> risk -> autopilot
        if 'last_face_matches' in locals():
            pass
        state = build_state(
            detections=last_detections,
            people=people,
            pose_info=last_pose,
            authorized_user=authorized_user if (authorized_user and time.time() <= auth_until) else None,
        )
        risk = score_risk(state)
        if autopilot_enabled:
            out = autopilot.decide(state, risk)
            if out.say:
                try:
                    speak(out.say)
                except Exception:
                    pass

        
        # --- Primary identity speech gate (no spam + no instant 'not recognized') ---
        now = time.time()
        # pick the largest person box with a name (if any)
        primary_name = None
        best_area = -1
        for d in last_detections:
            if d.get("label") != "person":
                continue
            box = tuple(d["box"])
            nm = name_map.get(box)
            if not nm:
                continue
            x1,y1,x2,y2 = box
            area = max(0,(x2-x1)) * max(0,(y2-y1))
            if area > best_area:
                best_area = area
                primary_name = nm

        # speak only on changes, and don't say 'not recognized' instantly
        if primary_name and primary_name != last_spoken["name"]:
            if now - last_spoken["t"] >= SPEAK_COOLDOWN_SEC:
                speak(f"Hey {primary_name}.")
                last_spoken = {"t": now, "name": primary_name}

        # if no primary name, only speak "not recognized" if we've been unknown longer than grace
        if not primary_name and last_spoken["name"] is not None:
            if now - last_spoken["t"] >= (ID_GRACE_SEC + SPEAK_COOLDOWN_SEC):
                speak("I can't recognize you right now.")
                last_spoken = {"t": now, "name": None}

        cv2.imshow("BRUNO Vision", frame)

        key = cv2.waitKey(1) & 0xFF

        # Keys
        if key == ord("q"):
            break

        elif key == ord("a"):
            autopilot_enabled = not autopilot_enabled
            print("BRUNO: Autopilot", "ON" if autopilot_enabled else "OFF")

        elif key == ord("n"):
            uid = input("New user id: ").strip().lower()
            if not uid:
                print("BRUNO: Cancelled.")
                continue

            ensure_user_dirs(USERS_ROOT, uid)

            pin1 = input("Create PIN: ").strip()
            pin2 = input("Confirm PIN: ").strip()
            if not pin1 or pin1 != pin2:
                print("BRUNO: PIN mismatch. Not created.")
                continue

            set_pin(USERS_ROOT, uid, pin1)
            authorized_user = uid
            auth_until = time.time() + AUTH_TTL
            print(f"BRUNO: Created + unlocked as {authorized_user}. Now press e to enroll face.")

        elif key == ord("e"):
            uid = input("Enroll which user id: ").strip().lower()
            if not uid:
                print("BRUNO: Cancelled.")
                continue
            ensure_user_dirs(USERS_ROOT, uid)
            ok = faceid.enroll(uid, frame, n_samples=10)
            if ok:
                print(f"BRUNO: Face enrolled for {uid}.")
            else:
                print("BRUNO: Could not detect a clear face. Try better lighting.")

        elif key == ord("u"):
            matches = faceid.match_faces(frame, threshold=0.35)
            m = None
            # choose largest matched face
            best_area = -1
            for fm in matches:
                if not fm.get("user_id"):
                    continue
                x1,y1,x2,y2 = fm["bbox"]
                area = (x2-x1)*(y2-y1)
                if area > best_area:
                    best_area = area
                    m = fm
            if m and m.get('user_id'):
                authorized_user = m["user_id"]
                auth_until = time.time() + AUTH_TTL
                print(f"BRUNO: Unlocked as {authorized_user} (face match).")
                speak(f"Unlocked as {authorized_user}.")
            else:
                print("BRUNO: Face not confident. PIN required.")
                uid = input("User id: ").strip().lower()
                if not uid:
                    print("BRUNO: Cancelled.")
                    continue
                if not pin_exists(USERS_ROOT, uid):
                    print("BRUNO: No PIN set for that user. Create profile with n first.")
                    continue
                pin = input("PIN: ").strip()
                if verify_pin(USERS_ROOT, uid, pin):
                    authorized_user = uid
                    auth_until = time.time() + AUTH_TTL
                    print(f"BRUNO: Unlocked as {authorized_user} (PIN).")
                    speak(f"Unlocked as {authorized_user}.")
                else:
                    print("BRUNO: Incorrect PIN.")

        elif key == ord("h"):
            if not authorized_user or time.time() > auth_until:
                authorized_user = None
                print("BRUNO: Locked. Press u to unlock first.")
                continue

            ensure_user_dirs(USERS_ROOT, authorized_user)
            payload = {
                "ts": time.strftime("%Y-%m-%dT%H-%M-%S"),
                "user_id": authorized_user,
                "detections": last_detections,
                "note": "scan placeholder"
            }
            path = save_scan_json(USERS_ROOT, authorized_user, payload)
            print("BRUNO: Saved scan ->", path)

    pose.close()
    close_camera(cap)
    cv2.destroyAllWindows()
    print("BRUNO shutdown.")


if __name__ == "__main__":
    main()
