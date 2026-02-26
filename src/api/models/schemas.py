"""
API Schemas
- Pydantic models for request/response validation
"""

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, description="The question to answer")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve")


class SourceInfo(BaseModel):
    title: str
    arxiv_id: str
    section: str
    authors: str
    distance: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceInfo]
    query: str


class HealthResponse(BaseModel):
    status: str
    ollama: bool
    chromadb: bool
    collection_count: int
