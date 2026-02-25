#!/usr/bin/env python3
"""
Run this script on your Raspberry Pi to find which import causes "Illegal Instruction".
Run from repo root: python3 scripts/check_pi_imports.py
If it crashes with "Illegal Instruction", the last [OK] line is the last successful step;
the next step is the one that triggered the crash. Set the corresponding env var (see
RASPBERRY_PI.md) and re-run the main app, or use BRUNO_PI=1 to disable all heavy libs.
"""
import sys
import os

# Ensure project root is on path (run from repo root or from scripts/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)

def step(name, fn):
    print(f"[  ] {name} ...", flush=True)
    fn()
    print(f"[OK] {name}", flush=True)

def main():
    print("BRUNO Pi import diagnostic (find Illegal Instruction source)\n", flush=True)

    step("stdlib", lambda: None)

    step("numpy", lambda: __import__("numpy"))
    step("cv2 (OpenCV)", lambda: __import__("cv2"))
    step("scipy", lambda: __import__("scipy"))

    step("mediapipe", lambda: __import__("mediapipe"))
    step("mediapipe.tasks.python.vision", lambda: __import__("mediapipe.tasks.python.vision"))

    step("insightface", lambda: __import__("insightface"))
    step("insightface.app.FaceAnalysis", lambda: __import__("insightface.app", fromlist=["FaceAnalysis"]))

    step("ultralytics (YOLO)", lambda: __import__("ultralytics"))
    step("torch", lambda: __import__("torch"))

    step("faster_whisper", lambda: __import__("faster_whisper"))

    step("sounddevice", lambda: __import__("sounddevice"))

    print("\nAll imports succeeded. If the main app still crashes, the error may happen")
    print("during model loading (e.g. FaceAnalysis.prepare(), YOLO(), PoseLandmarker).", flush=True)

if __name__ == "__main__":
    main()
