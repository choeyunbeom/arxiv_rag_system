"""
Query Router
- POST /query — answer a question using the RAG pipeline
"""

import httpx
from fastapi import APIRouter, HTTPException, Request

from src.api.models.schemas import (
    LatencyBreakdown,
    QueryRequest,
    QueryResponse,
    SourceInfo,
)

router = APIRouter()


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Ask a question about arXiv papers",
    response_description="Generated answer with source citations and latency breakdown",
    responses={
        400: {
            "description": "Invalid input (e.g. malformed question)",
            "content": {
                "application/json": {
                    "example": {"detail": "Invalid input: question cannot be processed"}
                }
            },
        },
        422: {
            "description": "Validation error (e.g. question too short, top_k out of range)",
        },
        503: {
            "description": "Ollama LLM service unavailable",
            "content": {
                "application/json": {
                    "example": {"detail": "Ollama service unavailable"}
                }
            },
        },
        504: {
            "description": "LLM generation timed out",
            "content": {
                "application/json": {
                    "example": {"detail": "LLM generation timed out"}
                }
            },
        },
    },
)
def query(request: QueryRequest, req: Request):
    """
    Answer a question using Retrieval-Augmented Generation over arXiv papers.

    The pipeline performs:
    1. **Hybrid retrieval**: dense vector search + BM25 keyword search
    2. **Reciprocal Rank Fusion**: merges results from both search methods
    3. **Cross-encoder reranking**: reranks top candidates for precision
    4. **Deduplication**: keeps the best chunk per paper section
    5. **LLM generation**: produces a cited answer using Qwen3 4B

    The response includes a latency breakdown showing time spent on
    retrieval vs generation, useful for performance monitoring.
    """
    rag_chain = req.app.state.rag_chain

    if rag_chain is None:
        raise HTTPException(
            status_code=503,
            detail="RAG pipeline is not initialized. Please ensure data is indexed and ChromaDB is running.",
        )

    try:
        result = rag_chain.query(
            question=request.question,
            top_k=request.top_k,
            include_vis=request.include_vis,
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
            latency=LatencyBreakdown(
                retrieval_ms=result.latency.retrieval_ms,
                generation_ms=result.latency.generation_ms,
                total_ms=result.latency.total_ms,
            ),
            vis_data=result.vis_data
        )
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="Ollama service unavailable")
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="LLM generation timed out")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {type(e).__name__}: {str(e)}")
