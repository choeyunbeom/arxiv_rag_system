"""
Config
- Centralised configuration loaded from environment variables
"""

import os


class Settings:
    OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "localhost:11434")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "qwen3:4b")
    EMBED_MODEL: str = os.getenv("EMBED_MODEL", "mxbai-embed-large")
    CHROMA_HOST: str = os.getenv("CHROMA_HOST", "localhost")
    CHROMA_PORT: int = int(os.getenv("CHROMA_PORT", "8200"))
    COLLECTION_NAME: str = "arxiv_papers"


settings = Settings()
