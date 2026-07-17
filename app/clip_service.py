import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from app.config import settings

print(f"Loading CLIP model... ({settings.MODEL_NAME})")
model = CLIPModel.from_pretrained(settings.MODEL_NAME)
processor = CLIPProcessor.from_pretrained(settings.MODEL_NAME)
model.eval()
print("✅ CLIP loaded")


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def encode_image(pil_image: Image.Image) -> np.ndarray:
    inputs = processor(images=pil_image, return_tensors="pt")
    with torch.no_grad():
        features = model.get_image_features(**inputs)
    return l2_normalize(features[0].numpy())


def encode_text(text: str) -> np.ndarray:
    inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        features = model.get_text_features(**inputs)
    return l2_normalize(features[0].numpy())