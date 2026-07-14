from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
import numpy as np
import io

app = FastAPI(title="Embedding Service")

# ---- Load CLIP model ONCE at startup ----
print("Loading CLIP model... (this takes ~30 seconds first time)")

MODEL_NAME = "openai/clip-vit-base-patch32"

model = CLIPModel.from_pretrained(MODEL_NAME)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)

model.eval()  # Inference mode (no training)

print("✅ CLIP model loaded and ready!")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "embedding-service",
        "model": MODEL_NAME
    }


@app.post("/embed")
async def embed(file: UploadFile = File(...)):
    # 1. Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an image"
        )

    # 2. Read image bytes → PIL Image
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image: {str(e)}"
        )

    # 3. Preprocess image for CLIP
    inputs = processor(images=image, return_tensors="pt")

    # 4. Get image embedding from CLIP (no gradient tracking → faster)
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)

    # 5. Convert to NumPy array
    vector = image_features[0].numpy()

    # 6. Normalize (unit vector) — Critical for cosine similarity
    vector = vector / np.linalg.norm(vector)

    # 7. Return embedding
    return {
        "vector": vector.tolist(),
        "dimension": len(vector),
        "model": MODEL_NAME
    }