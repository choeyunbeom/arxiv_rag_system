import logging

logger = logging.getLogger(__name__)
"""
ChromaDB Indexer (v4)
- Uses mxbai-embed-large for embeddings
- Batch indexing with individual fallback on failure
- If a batch fails, tries each chunk individually and skips only the broken ones
"""

import json
import time

import chromadb
import httpx

from src.api.core.config import settings, DATA_DIR


PROCESSED_DIR = DATA_DIR / "processed"
CHUNKS_FILE = PROCESSED_DIR / "chunks.json"

BATCH_SIZE = 32


def embed_single(text: str) -> list[float] | None:
    """Embed a single text. Returns None on failure."""
    try:
        response = httpx.post(
            f"http://{settings.OLLAMA_HOST}/api/embed",
            json={"model": settings.EMBED_MODEL, "input": [text]},
            timeout=60.0,
        )
        if response.status_code == 200:
            return response.json()["embeddings"][0]
    except Exception as e:
        logger.warning(f"embed_single failed: {type(e).__name__}: {e}")
    return None


def embed_batch(texts: list[str]) -> list[list[float]] | None:
    """Embed a batch. Returns None on failure."""
    try:
        response = httpx.post(
            f"http://{settings.OLLAMA_HOST}/api/embed",
            json={"model": settings.EMBED_MODEL, "input": texts},
            timeout=120.0,
        )
        if response.status_code == 200:
            return response.json()["embeddings"]
    except Exception as e:
        logger.warning(f"embed_batch failed: {type(e).__name__}: {e}")
    return None


def index_chunks():
    """Index all chunks into ChromaDB."""
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    chunks = [c for c in data["chunks"] if c.get("text") and len(c["text"].strip()) > 10]
    print(f"Loading {len(chunks)} chunks (filtered from {len(data['chunks'])})")

    client = chromadb.HttpClient(host=settings.CHROMA_HOST, port=settings.CHROMA_PORT)

    try:
        client.delete_collection(settings.COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=settings.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    total_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE
    indexed = 0
    skipped = 0

    for batch_idx in range(total_batches):
        start = batch_idx * BATCH_SIZE
        end = min(start + BATCH_SIZE, len(chunks))
        batch = chunks[start:end]

        texts = [c["text"].strip() for c in batch]
        ids = [c["chunk_id"] for c in batch]
        metadatas = [
            {
                "arxiv_id": c["arxiv_id"],
                "title": c["title"],
                "section": c["section"],
                "word_count": c["word_count"],
                "authors": ", ".join(c["metadata"].get("authors", [])[:3]),
                "published": c["metadata"].get("published", ""),
            }
            for c in batch
        ]

        # Try batch first
        embeddings = embed_batch(texts)

        if embeddings:
            collection.add(
                ids=ids, embeddings=embeddings,
                documents=texts, metadatas=metadatas,
            )
            indexed += len(batch)
            print(f"  Batch {batch_idx + 1}/{total_batches} indexed ({indexed}/{len(chunks)})")
        else:
            # Fallback: embed individually
            ok = 0
            for i in range(len(batch)):
                emb = embed_single(texts[i])
                if emb:
                    collection.add(
                        ids=[ids[i]], embeddings=[emb],
                        documents=[texts[i]], metadatas=[metadatas[i]],
                    )
                    indexed += 1
                    ok += 1
                else:
                    skipped += 1

            print(f"  Batch {batch_idx + 1}/{total_batches} fallback ({ok} ok, {len(batch)-ok} skipped)")

    count = collection.count()
    print(f"\n  Indexed {count} chunks in collection '{settings.COLLECTION_NAME}'")
    print(f"  Skipped {skipped} chunks (embedding failed)")

    # Test query
    print("\n  Test query: 'What is Retrieval Augmented Generation?'")
    test_emb = embed_single("What is Retrieval Augmented Generation?")
    if test_emb:
        results = collection.query(query_embeddings=[test_emb], n_results=5)
        for i, (doc, meta, dist) in enumerate(zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        )):
            print(f"\n  [{i+1}] dist={dist:.4f} | {meta['title'][:50]}")
            print(f"      Section: {meta['section']}")
            print(f"      Preview: {doc[:100]}...")


if __name__ == "__main__":
    index_chunks()