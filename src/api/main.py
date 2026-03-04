"""
arXiv RAG System API
- FastAPI application with lifespan for pre-loading heavy resources
- Mounts query and health routers
- Enhanced OpenAPI documentation with tag descriptions
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.core.rag_chain import RAGChain
from src.api.routers import health, query

# OpenAPI tag metadata for Swagger UI grouping
tags_metadata = [
    {
        "name": "Query",
        "description": "Ask questions and receive answers grounded in arXiv papers. "
        "The RAG pipeline performs hybrid retrieval, cross-encoder reranking, "
        "and LLM generation with source citations.",
    },
    {
        "name": "Health",
        "description": "Monitor the health of dependent services (Ollama, ChromaDB) "
        "and the number of indexed chunks available for search.",
    },
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load heavy resources at server startup, not on first request."""
    print("Loading RAG pipeline...")
    try:
        app.state.rag_chain = RAGChain()
        print("RAG pipeline ready.")
    except Exception as e:
        print(f"WARNING: RAG pipeline not available — {e}")
        print("API will start in degraded mode. Run 'make seed' to index data.")
        app.state.rag_chain = None
    yield
    print("Shutting down.")


app = FastAPI(
    title="arXiv RAG System",
    description=(
        "Question answering over arXiv papers using Retrieval-Augmented Generation.\n\n"
        "## Pipeline\n"
        "1. **Hybrid Search** — dense vector (mxbai-embed-large) + BM25 keyword search\n"
        "2. **RRF Fusion** — Reciprocal Rank Fusion to merge retrieval results\n"
        "3. **Cross-Encoder Reranking** — ms-marco-MiniLM-L-6-v2 for precision reranking\n"
        "4. **Deduplication** — best chunk per paper section\n"
        "5. **LLM Generation** — Qwen3 4B generates a cited answer\n\n"
        "## Corpus\n"
        "132 arXiv papers covering RAG, QLoRA, LoRA, LLM fine-tuning, "
        "and hallucination mitigation."
    ),
    version="0.1.0",
    lifespan=lifespan,
    openapi_tags=tags_metadata,
    contact={
        "name": "Yunbeom Choe",
        "url": "https://github.com/choeyunbeom/arxiv_rag_system",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
)

app.include_router(query.router, tags=["Query"])
app.include_router(health.router, tags=["Health"])


@app.get("/", tags=["Health"])
async def root():
    """Root endpoint returning basic API information and link to documentation."""
    return {
        "name": "arXiv RAG System",
        "version": "0.1.0",
        "docs": "/docs",
    }
