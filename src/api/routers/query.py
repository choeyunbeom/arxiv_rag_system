"""
Query Router
- POST /query — answer a question using the RAG pipeline
"""

from fastapi import APIRouter, HTTPException, Request
import httpx

from src.api.models.schemas import QueryRequest, QueryResponse, SourceInfo

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest, req: Request):
    """Answer a question using RAG over arXiv papers."""
    rag_chain = req.app.state.rag_chain

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
