"""
Health Router
- GET /health — check status of all services
"""

import os
from fastapi import APIRouter
import httpx
import chromadb

from src.api.models.schemas import HealthResponse

router = APIRouter()

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "localhost:11434")
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8200"))


@router.get("/health", response_model=HealthResponse)
async def health():
    """Check health of all dependent services."""
    ollama_ok = False
    chroma_ok = False
    collection_count = 0

    # Check Ollama
    try:
        r = httpx.get(f"http://{OLLAMA_HOST}/api/tags", timeout=5.0)
        ollama_ok = r.status_code == 200
    except Exception:
        pass

    # Check ChromaDB
    try:
        client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        collection = client.get_collection("arxiv_papers")
        collection_count = collection.count()
        chroma_ok = True
    except Exception:
        pass

    status = "healthy" if (ollama_ok and chroma_ok) else "degraded"

    return HealthResponse(
        status=status,
        ollama=ollama_ok,
        chromadb=chroma_ok,
        collection_count=collection_count,
    )
