import os
from typing import List
from dotenv import load_dotenv

load_dotenv()


class Settings:
    MODEL_NAME: str = os.getenv("MODEL_NAME", "openai/clip-vit-base-patch32")
    CORS_ORIGINS: List[str] = os.getenv("CORS_ORIGINS", "*").split(",")
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))


settings = Settings()