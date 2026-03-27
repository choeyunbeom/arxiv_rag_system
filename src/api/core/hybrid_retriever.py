"""
Hybrid Retriever with Reranker (v3)
- Stage 1: Vector search (ChromaDB) + BM25 keyword search
- Stage 2: RRF fusion to merge results
- Stage 3: Cross-encoder reranker for final ranking
- Deduplication: only the best chunk per arxiv_id is kept
"""

import asyncio
import json
import math
import pickle
import re
from dataclasses import dataclass

import chromadb
import httpx
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from src.api.core.config import DATA_DIR, settings

CHUNKS_FILE = DATA_DIR / "processed" / "chunks.json"
UMAP_MODEL_PATH = DATA_DIR / "processed" / "umap_model.pkl"
UMAP_BG_JSON_PATH = DATA_DIR / "processed" / "umap_bg.json"

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    arxiv_id: str
    title: str
    section: str
    authors: str
    published: str
    distance: float


class HybridRetriever:
    def __init__(self):
        # Vector search
        self.chroma_client = chromadb.HttpClient(host=settings.CHROMA_HOST, port=settings.CHROMA_PORT)
        self.collection = self.chroma_client.get_collection(settings.COLLECTION_NAME)

        # Persistent HTTP client for connection pooling
        self._http_client = httpx.AsyncClient(timeout=30.0)

        # BM25 search
        self._build_bm25_index()

        # Reranker
        print("  Loading reranker model...")
        self.reranker = CrossEncoder(RERANKER_MODEL)
        print(f"  Reranker loaded: {RERANKER_MODEL}")

        # UMAP Models
        self.umap_reducer = None
        self.umap_bg_data = None
        self._load_umap()

    def __del__(self):
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self._http_client.aclose())
            else:
                loop.run_until_complete(self._http_client.aclose())
        except Exception:
            pass

    def _load_umap(self):
        """Load precomputed UMAP model and background points if available."""
        if UMAP_MODEL_PATH.exists() and UMAP_BG_JSON_PATH.exists():
            print("  Loading UMAP model and background data...")
            with open(UMAP_MODEL_PATH, "rb") as f:
                self.umap_reducer = pickle.load(f)
            with open(UMAP_BG_JSON_PATH, "r", encoding="utf-8") as f:
                self.umap_bg_data = json.load(f)
            print(f"  UMAP loaded with {len(self.umap_bg_data)} background points.")
        else:
            print("  UMAP files not found. Visualization will be disabled.")

    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenizer for BM25."""
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s\-]", " ", text)
        return [w for w in text.split() if len(w) > 1]

    def _build_bm25_index(self):
        """Build BM25 index from chunks.json."""
        with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.chunks_data = data["chunks"]
        self.chunk_id_to_idx = {c["chunk_id"]: i for i, c in enumerate(self.chunks_data)}

        tokenized = [self._tokenize(c["text"]) for c in self.chunks_data]
        self.bm25 = BM25Okapi(tokenized)

    async def _embed_query(self, query: str) -> list[float]:
        """Embed a query using Ollama."""
        response = await self._http_client.post(
            f"http://{settings.OLLAMA_HOST}/api/embed",
            json={"model": settings.EMBED_MODEL, "input": [query]},
        )
        response.raise_for_status()
        return response.json()["embeddings"][0]

    def _vector_search(self, query_embedding: list[float], top_k: int) -> dict[str, float]:
        """Vector search via ChromaDB. Returns {chunk_id: rank}."""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

        ranked = {}
        for rank, cid in enumerate(results["ids"][0]):
            ranked[cid] = rank + 1
        return ranked

    def _bm25_search(self, query: str, top_k: int) -> dict[str, float]:
        """BM25 keyword search. Returns {chunk_id: rank}."""
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        ranked = {}
        for rank, idx in enumerate(top_indices):
            if scores[idx] > 0:
                chunk_id = self.chunks_data[idx]["chunk_id"]
                ranked[chunk_id] = rank + 1
        return ranked

    def _rrf_fusion(self, vector_ranks: dict, bm25_ranks: dict, k: int = 60) -> list[str]:
        """Reciprocal Rank Fusion to combine two ranked lists."""
        all_ids = set(vector_ranks.keys()) | set(bm25_ranks.keys())

        scores = {}
        for cid in all_ids:
            score = 0.0
            if cid in vector_ranks:
                score += 1.0 / (k + vector_ranks[cid])
            if cid in bm25_ranks:
                score += 1.0 / (k + bm25_ranks[cid])
            scores[cid] = score

        return sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    def _rerank(self, query: str, chunk_ids: list[str]) -> list[tuple[str, float]]:
        """Rerank candidates using cross-encoder."""
        pairs = []
        valid_ids = []
        for cid in chunk_ids:
            if cid in self.chunk_id_to_idx:
                idx = self.chunk_id_to_idx[cid]
                text = self.chunks_data[idx]["text"]
                truncated = " ".join(text.split()[:200])
                pairs.append([query, truncated])
                valid_ids.append(cid)

        if not pairs:
            return []

        scores = self.reranker.predict(pairs)

        scored = list(zip(valid_ids, scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def _deduplicate(self, scored_ids: list[tuple[str, float]], top_k: int) -> list[tuple[str, float]]:
        """Deduplicate by arxiv_id + section. Same paper, different sections are kept."""
        seen = set()
        deduped = []
        for cid, score in scored_ids:
            if cid not in self.chunk_id_to_idx:
                continue
            idx = self.chunk_id_to_idx[cid]
            arxiv_id = self.chunks_data[idx]["arxiv_id"]
            section = self.chunks_data[idx]["section"]
            key = f"{arxiv_id}::{section}"
            if key not in seen:
                seen.add(key)
                deduped.append((cid, score))
            if len(deduped) >= top_k:
                break
        return deduped

    async def search(self, query: str, top_k: int = 5, get_embeddings: bool = False) -> tuple[list[RetrievedChunk], list[float] | None]:
        """Hybrid search with reranking and deduplication."""
        # Stage 1: Fetch broad candidates
        fetch_k = top_k * 8  # Get 40 candidates for better coverage
        query_embedding = await self._embed_query(query)

        # Offload sync CPU-bound operations to threadpool to avoid blocking the event loop
        vector_ranks, bm25_ranks = await asyncio.gather(
            asyncio.to_thread(self._vector_search, query_embedding, fetch_k),
            asyncio.to_thread(self._bm25_search, query, fetch_k),
        )

        # Stage 2: RRF fusion
        fused_ids = self._rrf_fusion(vector_ranks, bm25_ranks)[:fetch_k]

        # Stage 3: Rerank all candidates (CPU-intensive ML inference)
        reranked = await asyncio.to_thread(self._rerank, query, fused_ids)

        # Stage 4: Deduplicate by arxiv_id
        deduped = self._deduplicate(reranked, top_k)

        # Build results
        chunks = []
        for cid, rerank_score in deduped:
            idx = self.chunk_id_to_idx[cid]
            c = self.chunks_data[idx]
            chunk = RetrievedChunk(
                chunk_id=cid,
                text=c["text"],
                arxiv_id=c["arxiv_id"],
                title=c["title"],
                section=c["section"],
                authors=", ".join(c["metadata"].get("authors", [])[:3]),
                published=c["metadata"].get("published", ""),
                distance=1.0 - (1.0 / (1.0 + math.exp(-rerank_score))),  # sigmoid normalisation
            )
            chunks.append(chunk)

        query_emb_out = query_embedding if get_embeddings else None
        return chunks, query_emb_out
