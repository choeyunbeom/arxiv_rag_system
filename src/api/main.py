"""
arXiv RAG System API
- FastAPI application entry point
- Mounts query and health routers
"""

from fastapi import FastAPI
from src.api.routers import query, health

app = FastAPI(
    title="arXiv RAG System",
    description="Question answering over arXiv papers using Retrieval-Augmented Generation",
    version="0.1.0",
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
