"""
Health Router
- GET /health — check status of all services
"""

from fastapi import APIRouter
import httpx
import chromadb

from src.api.core.config import settings
from src.api.models.schemas import HealthResponse

router = APIRouter()



@router.get("/health", response_model=HealthResponse)
async def health():
    """Check health of all dependent services."""
    ollama_ok = False
    chroma_ok = False
    collection_count = 0

    # Check Ollama
    try:
        r = httpx.get(f"http://{settings.OLLAMA_HOST}/api/tags", timeout=5.0)
        ollama_ok = r.status_code == 200
    except Exception:
        pass

    # Check ChromaDB
    try:
        client = chromadb.HttpClient(host=settings.CHROMA_HOST, port=settings.CHROMA_PORT)
        collection = client.get_collection(settings.COLLECTION_NAME)
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
