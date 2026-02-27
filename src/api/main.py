"""
arXiv RAG System API
- FastAPI application with lifespan for pre-loading heavy resources
- Mounts query and health routers
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI

from src.api.core.rag_chain import RAGChain
from src.api.routers import query, health


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load heavy resources at server startup, not on first request."""
    print("Loading RAG pipeline...")
    app.state.rag_chain = RAGChain()
    print("RAG pipeline ready.")
    yield
    # Cleanup (if needed)
    print("Shutting down.")


app = FastAPI(
    title="arXiv RAG System",
    description="Question answering over arXiv papers using Retrieval-Augmented Generation",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(query.router, tags=["Query"])
app.include_router(health.router, tags=["Health"])


@app.get("/")
async def root():
    return {
        "name": "arXiv RAG System",
        "version": "0.1.0",
        "docs": "/docs",
    }
