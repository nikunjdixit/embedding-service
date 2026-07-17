from pydantic import BaseModel


class TextRequest(BaseModel):
    text: str


class EmbeddingResponse(BaseModel):
    vector: list[float]
    dimension: int
    model: str