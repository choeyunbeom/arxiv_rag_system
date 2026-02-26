"""
Query Router
- POST /query — answer a question using the RAG pipeline
"""

from fastapi import APIRouter, HTTPException

from src.api.models.schemas import QueryRequest, QueryResponse, SourceInfo
from src.api.core.rag_chain import RAGChain

router = APIRouter()
rag_chain = RAGChain()


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
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
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
