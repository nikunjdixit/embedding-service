import io

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from app.clip_service import encode_image, encode_text
from app.config import settings
from app.models import TextRequest

app = FastAPI(title="Embedding Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "service": "embedding-service", "model": settings.MODEL_NAME}


@app.post("/embed")
async def embed(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")

    vector = encode_image(image)

    return {
        "vector": vector.tolist(),
        "dimension": len(vector),
        "model": settings.MODEL_NAME,
    }


@app.post("/embed/text")
async def embed_text(req: TextRequest):
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    try:
        vector = encode_text(text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text embedding failed: {str(e)}")

    return {
        "vector": vector.tolist(),
        "dimension": len(vector),
        "model": settings.MODEL_NAME,
    }