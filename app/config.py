import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION: str = "image_embeddings"
    MODEL_NAME: str = "openai/clip-vit-base-patch32"
    UPLOAD_DIR: str = "data/uploads"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    @property
    def upload_path(self) -> Path:
        return Path(self.UPLOAD_DIR)

settings = Settings()

# Ensure the upload directory exists
settings.upload_path.mkdir(parents=True, exist_ok=True)
