"""
Hybrid Retriever with Reranker (v3)
- Stage 1: Vector search (ChromaDB) + BM25 keyword search
- Stage 2: RRF fusion to merge results
- Stage 3: Cross-encoder reranker for final ranking
- Deduplication: only the best chunk per arxiv_id is kept
"""

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import chromadb
import httpx
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder


CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8200"))
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "mxbai-embed-large")
COLLECTION_NAME = "arxiv_papers"
CHUNKS_FILE = Path("data/processed/chunks.json")

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@dataclass
class RetrievedChunk:
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
        self.chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        self.collection = self.chroma_client.get_collection(COLLECTION_NAME)

        # BM25 search
        self._build_bm25_index()

        # Reranker
        print("  Loading reranker model...")
        self.reranker = CrossEncoder(RERANKER_MODEL)
        print(f"  Reranker loaded: {RERANKER_MODEL}")

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

    def _embed_query(self, query: str) -> list[float]:
        """Embed a query using Ollama."""
        response = httpx.post(
            f"http://{OLLAMA_HOST}/api/embed",
            json={"model": EMBED_MODEL, "input": [query]},
            timeout=30.0,
        )
        response.raise_for_status()
        return response.json()["embeddings"][0]

    def _vector_search(self, query: str, top_k: int) -> dict[str, float]:
        """Vector search via ChromaDB. Returns {chunk_id: rank}."""
        query_embedding = self._embed_query(query)
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

    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        """Hybrid search with reranking and deduplication."""
        # Stage 1: Fetch broad candidates
        fetch_k = top_k * 8  # Get 40 candidates for better coverage

        vector_ranks = self._vector_search(query, fetch_k)
        bm25_ranks = self._bm25_search(query, fetch_k)

        # Stage 2: RRF fusion
        fused_ids = self._rrf_fusion(vector_ranks, bm25_ranks)[:fetch_k]

        # Stage 3: Rerank all candidates
        reranked = self._rerank(query, fused_ids)

        # Stage 4: Deduplicate by arxiv_id
        deduped = self._deduplicate(reranked, top_k)

        # Build results
        chunks = []
        for cid, rerank_score in deduped:
            idx = self.chunk_id_to_idx[cid]
            c = self.chunks_data[idx]
            chunk = RetrievedChunk(
                text=c["text"],
                arxiv_id=c["arxiv_id"],
                title=c["title"],
                section=c["section"],
                authors=", ".join(c["metadata"].get("authors", [])[:3]),
                published=c["metadata"].get("published", ""),
                distance=1.0 - (rerank_score / 10.0),
            )
            chunks.append(chunk)

        return chunks
