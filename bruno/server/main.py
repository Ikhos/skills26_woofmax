from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2

app = FastAPI(title="BRUNO Backend", version="0.1.0")

# Allow Streamlit (and future UIs) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok", "service": "bruno-backend"}

@app.post("/scan")
async def scan_image(file: UploadFile = File(...)):
    # Read bytes -> decode image
    img_bytes = await file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return {"ok": False, "error": "Could not decode image. Upload a jpg/png."}

    h, w = img.shape[:2]

    # Dummy output for now (we’ll replace with YOLO + FaceMesh later)
    return {
        "ok": True,
        "image": {"width": int(w), "height": int(h)},
        "perception": {
            "detections": [],
            "face": {"detected": False},
            "quality": {"blur_score": None, "lighting": "unknown"},
        },
        "logic": {
            "risk_level": "unknown",
            "reasons": ["models_not_enabled_yet"],
            "questions": [],
            "urgent_if_yes": False,
        },
        "identity": {
            "status": "disabled",
            "name": None,
            "confidence": None,
        },
    }