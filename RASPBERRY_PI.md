# Running BRUNO on Raspberry Pi 4 (Bookworm 64-bit)

The app runs on macOS/Windows but can hit **"Illegal Instruction"** on Raspberry Pi 4 (ARM). This happens when a library uses CPU instructions that the Pi’s ARM CPU doesn’t support (e.g. x86-only or wrong ARM build).

## 1. Find what’s crashing: diagnostic script

On the Pi (after cloning and installing deps), run:

```bash
cd /path/to/skills26_woofmax
python3 scripts/check_pi_imports.py
```

The script imports heavy libraries one by one. **When it crashes with "Illegal Instruction", the last `[OK]` line is the previous step; the next step is the one that triggered the crash.**

Typical culprits:

- **insightface** – often no proper ARM64 wheel; uses ONNX/x86-style builds.
- **mediapipe** – prebuilt binaries can use instructions not supported on Pi 4.
- **ultralytics / torch** – need ARM-compatible PyTorch.
- **faster_whisper** – CTranslate2 may not have a compatible build for your Pi.
- **numpy / scipy** – only if you got the wrong (e.g. x86) wheel by mistake.

## 2. Run with stubs (get the app up on Pi)

You can disable the heavy components and run with stubs so the rest of the app works (camera, UI, PIN, etc.) while face recognition, pose, object detection, or voice are off.

**Disable everything at once (fastest way to get a running app):**

```bash
BRUNO_PI=1 python3 -m bruno.run
```

**Or disable only what crashed** (from the diagnostic):

```bash
# If InsightFace crashed:
BRUNO_DISABLE_FACEID=1 python3 -m bruno.run

# If MediaPipe (pose) crashed:
BRUNO_DISABLE_POSE=1 python3 -m bruno.run

# If YOLO/Ultralytics crashed:
BRUNO_DISABLE_YOLO=1 python3 -m bruno.run

# If faster_whisper crashed:
BRUNO_DISABLE_WHISPER=1 python3 -m bruno.run
```

Combine as needed, e.g.:

```bash
BRUNO_DISABLE_FACEID=1 BRUNO_DISABLE_POSE=1 python3 -m bruno.run
```

With stubs:

- **Face ID**: no face detection or enrollment (unlock only via PIN).
- **Pose**: no skeleton/pose overlay.
- **YOLO**: no object/person boxes.
- **Whisper**: no voice triggers; other features still work.

## 3. Pi-friendly install tips

- **Use Pi 64-bit OS** (e.g. Raspberry Pi OS Bookworm 64-bit). Many wheels (including MediaPipe) only ship for `aarch64`, not 32-bit ARM.
- **Prefer piwheels** so you get ARM-built wheels when available:
  ```bash
  pip install --extra-index-url https://www.piwheels.org/simple numpy opencv-python scipy
  ```
- **System OpenCV** (optional): install with `sudo apt install python3-opencv` and use that Python/env if it’s compatible with your script.
- **InsightFace on ARM64**: there may be no official wheel. Options:
  - Use 32-bit OS and try piwheels’ InsightFace (if available for armv7l).
  - Or keep `BRUNO_DISABLE_FACEID=1` and use PIN-only auth on Pi.
- **PyTorch on Pi**: install the ARM build from the official PyTorch site or use a Pi-specific guide so you get a compatible `torch` (and thus `ultralytics`).

## 4. Summary

| Problem              | What to do |
|----------------------|------------|
| "Illegal Instruction" | Run `scripts/check_pi_imports.py` to see which import fails. |
| Get app running on Pi | Use `BRUNO_PI=1` or disable only the failing component with `BRUNO_DISABLE_*`. |
| Full face/pose/YOLO on Pi | Use 64-bit OS, piwheels, and ARM builds for InsightFace/MediaPipe/PyTorch as above. |
