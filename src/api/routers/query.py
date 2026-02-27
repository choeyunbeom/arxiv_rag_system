"""
Query Router
- POST /query — answer a question using the RAG pipeline
"""

from fastapi import APIRouter, HTTPException, Depends
import httpx

from src.api.models.schemas import QueryRequest, QueryResponse, SourceInfo
from src.api.core.rag_chain import RAGChain

router = APIRouter()

# Lazy-initialised singleton to avoid loading models at import time
_rag_chain: RAGChain | None = None


def get_rag_chain() -> RAGChain:
    """Return the RAGChain singleton, creating it on first call."""
    global _rag_chain
    if _rag_chain is None:
        _rag_chain = RAGChain()
    return _rag_chain


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest, rag_chain: RAGChain = Depends(get_rag_chain)):
    """Answer a question using RAG over arXiv papers."""
    try:
        result = rag_chain.query(
            question=request.question,
            top_k=request.top_k,
        )

        sources = [
            SourceInfo(
                title=s.title,
                arxiv_id=s.arxiv_id,
                section=s.section,
                authors=s.authors,
                distance=s.distance,
            )
            for s in result.sources
        ]

        return QueryResponse(
            answer=result.answer,
            sources=sources,
            query=result.query,
        )
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="Ollama service unavailable")
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="LLM generation timed out")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {type(e).__name__}: {str(e)}")
