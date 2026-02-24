import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import numpy as np

MODEL_NAME = "imfarzanansari/skintelligent-acne"

class AcneClassifier:
    def __init__(self):
        self.processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
        self.model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
        self.model.eval()

        self.labels = [
            "Clear Skin",
            "Occasional Spots",
            "Mild Acne",
            "Moderate Acne",
            "Severe Acne",
            "Very Severe Acne"
        ]

    def predict(self, frame):
        image = Image.fromarray(frame)
        inputs = self.processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=-1).item()

        return self.labels[predicted_class]