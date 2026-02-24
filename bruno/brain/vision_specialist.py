import torch
import cv2
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

MODEL_NAME = "imfarzanansari/skintelligent-acne"

# Load model once at import
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def crop_center_square(frame):
    """
    Simple center square crop.
    You can later replace this with FaceEmbed bounding box.
    """
    h, w, _ = frame.shape
    size = min(h, w)
    start_x = w // 2 - size // 2
    start_y = h // 2 - size // 2
    return frame[start_y:start_y + size, start_x:start_x + size]


def analyze_scene(frame, routing=None):
    """
    Runs acne classification only when routing requires vision.
    Returns structured acne result.
    """

    if frame is None:
        return {"acne_stage": None, "confidence": 0.0}

    try:
        # Crop face region
        face_crop = crop_center_square(frame)

        # Convert BGR → RGB
        rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)

        inputs = processor(images=pil_image, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        conf, pred = torch.max(probs, dim=1)

        class_id = pred.item()
        # Manual label mapping (6 acne stages)
        ACNE_LABELS = { 0: "Clear Skin", 1: "Occasional Spots", 2: "Mild Acne", 3: "Moderate Acne", 4: "Severe Acne", 5: "Very Severe Acne" }
        label = ACNE_LABELS.get(class_id, "Unknown")
        confidence = float(conf.item())

        return {
            "acne_stage": label,
            "confidence": confidence
        }

    except Exception as e:
        print("Vision error:", e)
        return {"acne_stage": None, "confidence": 0.0}