"""
Centralised configuration using Pydantic BaseSettings.
All modules should import settings from here instead of calling os.getenv() directly.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


# Resolve project root regardless of working directory
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"


class Settings(BaseSettings):
    OLLAMA_HOST: str = "localhost:11434"
    LLM_MODEL: str = "qwen3:4b"
    EMBED_MODEL: str = "mxbai-embed-large"
    CHROMA_HOST: str = "localhost"
    CHROMA_PORT: int = 8200
    COLLECTION_NAME: str = "arxiv_papers"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )


settings = Settings()
