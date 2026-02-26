"""
Retriever
- Search ChromaDB for relevant chunks given a query
- Uses mxbai-embed-large for query embedding
- Returns ranked results with metadata
"""

import os
from dataclasses import dataclass

import chromadb
import httpx


CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8200"))
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "mxbai-embed-large")
COLLECTION_NAME = "arxiv_papers"


@dataclass
class RetrievedChunk:
    text: str
    arxiv_id: str
    title: str
    section: str
    authors: str
    published: str
    distance: float


class Retriever:
    def __init__(self):
        self.chroma_client = chromadb.HttpClient(
            host=CHROMA_HOST, port=CHROMA_PORT
        )
        self.collection = self.chroma_client.get_collection(COLLECTION_NAME)

    def _embed_query(self, query: str) -> list[float]:
        """Embed a query using Ollama."""
        response = httpx.post(
            f"http://{OLLAMA_HOST}/api/embed",
            json={"model": EMBED_MODEL, "input": [query]},
            timeout=30.0,
        )
        response.raise_for_status()
        return response.json()["embeddings"][0]

    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        """Search for relevant chunks."""
        query_embedding = self._embed_query(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

        chunks = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            chunk = RetrievedChunk(
                text=doc,
                arxiv_id=meta.get("arxiv_id", ""),
                title=meta.get("title", ""),
                section=meta.get("section", ""),
                authors=meta.get("authors", ""),
                published=meta.get("published", ""),
                distance=dist,
            )
            chunks.append(chunk)

        return chunks
