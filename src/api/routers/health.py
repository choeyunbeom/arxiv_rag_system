"""
Health Router
- GET /health — check status of all services
"""

import asyncio

import chromadb
import httpx
from fastapi import APIRouter

from src.api.core.config import settings
from src.api.models.schemas import HealthResponse

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Check system health",
    response_description="Health status of all dependent services",
)
async def health():
    """
    Check the health of all dependent services.

    Returns the status of:
    - **Ollama**: LLM and embedding model service
    - **ChromaDB**: vector store for document embeddings
    - **Collection count**: number of indexed chunks available for search

    Overall status is `healthy` when all services are reachable,
    or `degraded` when one or more services are unavailable.
    """

    async def _check_ollama() -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(f"http://{settings.OLLAMA_HOST}/api/tags")
                return r.status_code == 200
        except Exception:
            return False

    async def _check_chroma() -> tuple[bool, int]:
        try:
            def _sync_chroma():
                client = chromadb.HttpClient(host=settings.CHROMA_HOST, port=settings.CHROMA_PORT)
                collection = client.get_collection(settings.COLLECTION_NAME)
                return collection.count()

            count = await asyncio.to_thread(_sync_chroma)
            return True, count
        except Exception:
            return False, 0

    ollama_ok, (chroma_ok, collection_count) = await asyncio.gather(
        _check_ollama(),
        _check_chroma(),
    )

    status = "healthy" if (ollama_ok and chroma_ok) else "degraded"

    return HealthResponse(
        status=status,
        ollama=ollama_ok,
        chromadb=chroma_ok,
        collection_count=collection_count,
    )
