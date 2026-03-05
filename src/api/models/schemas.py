"""
API Schemas
- Pydantic models for request/response validation
- Includes OpenAPI examples for Swagger UI documentation
"""

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Submit a question to the RAG pipeline for an answer grounded in arXiv papers."""

    question: str = Field(
        ...,
        min_length=3,
        description="Natural language question about AI research topics",
        json_schema_extra={"examples": ["What is QLoRA and how does it reduce memory usage?"]},
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of source chunks to retrieve and cite in the answer",
    )
    include_vis: bool = Field(
        default=False,
        description="Whether to return UMAP 2D coordinates for interactive embedding visualization",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "question": "What is QLoRA and how does it reduce memory usage?",
                    "top_k": 5,
                    "include_vis": True,
                }
            ]
        }
    }


class SourceInfo(BaseModel):
    """A single source chunk retrieved from the arXiv paper corpus."""

    title: str = Field(description="Title of the arXiv paper")
    arxiv_id: str = Field(description="arXiv identifier (e.g. 2305.14314)")
    section: str = Field(description="Section of the paper the chunk was extracted from")
    authors: str = Field(description="First 3 authors of the paper, comma-separated")
    distance: float = Field(
        description="Relevance distance (lower = more relevant, 0–1 after sigmoid normalisation)"
    )


class LatencyBreakdown(BaseModel):
    """Timing breakdown of the RAG pipeline stages in milliseconds."""

    retrieval_ms: float = Field(description="Time spent on hybrid retrieval (vector + BM25 + reranking)")
    generation_ms: float = Field(description="Time spent on LLM answer generation")
    total_ms: float = Field(description="Total end-to-end latency")


class VisPoint(BaseModel):
    """A single embedding point transformed into 2D via UMAP."""

    chunk_id: str | None = Field(default=None, description="The ID of the chunk, None for query")
    x: float = Field(description="3D X coordinate")
    y: float = Field(description="3D Y coordinate")
    z: float = Field(description="3D Z coordinate")
    title: str | None = Field(default=None, description="Title of the paper")
    section: str | None = Field(default=None, description="Section of the paper")
    text_preview: str | None = Field(default=None, description="Short snippet of the text")
    is_source: bool = Field(default=False, description="True if this chunk was one of the returned sources")
    is_query: bool = Field(default=False, description="True if this point is the user's query")


class VisData(BaseModel):
    """Visualization data to be rendered by Plotly."""
    points: list[VisPoint] = Field(description="List of all points to plot (background + query + sources)")


class QueryResponse(BaseModel):
    """RAG pipeline response containing the generated answer, source citations, and latency breakdown."""

    answer: str = Field(description="Generated answer grounded in retrieved arXiv paper sources")
    sources: list[SourceInfo] = Field(description="Retrieved source chunks cited in the answer")
    query: str = Field(description="Echo of the original question")
    latency: LatencyBreakdown = Field(description="Pipeline latency breakdown in milliseconds")
    vis_data: VisData | None = Field(default=None, description="Optional visualization data")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "answer": "QLoRA (Quantized Low-Rank Adaptation) reduces memory usage by quantizing the base model to 4-bit NormalFloat (NF4) precision while training only small low-rank adapter matrices in higher precision...",
                    "sources": [
                        {
                            "title": "QLoRA: Efficient Finetuning of Quantized LLMs",
                            "arxiv_id": "2305.14314",
                            "section": "Method",
                            "authors": "Dettmers, Pagnoni, Holtzman",
                            "distance": 0.12,
                        }
                    ],
                    "query": "What is QLoRA and how does it reduce memory usage?",
                    "latency": {
                        "retrieval_ms": 1340.5,
                        "generation_ms": 18200.3,
                        "total_ms": 19540.8,
                    },
                }
            ]
        }
    }


class HealthResponse(BaseModel):
    """Health status of the arXiv RAG System and its dependent services."""

    status: str = Field(description="Overall system status: 'healthy' or 'degraded'")
    ollama: bool = Field(description="Whether Ollama LLM service is reachable")
    chromadb: bool = Field(description="Whether ChromaDB vector store is reachable")
    collection_count: int = Field(description="Number of indexed chunks in the active collection")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "status": "healthy",
                    "ollama": True,
                    "chromadb": True,
                    "collection_count": 5234,
                }
            ]
        }
    }
