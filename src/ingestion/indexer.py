"""
ChromaDB Indexer (v3)
- Vectorise chunks using Ollama embeddings
- Uses nomic-embed-text prefixes for proper retrieval
  (search_document: for docs, search_query: for queries)
- Store in ChromaDB with metadata
"""

import json
import os
import time
from pathlib import Path

import chromadb
import httpx


PROCESSED_DIR = Path("data/processed")
CHUNKS_FILE = PROCESSED_DIR / "chunks.json"

CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8200"))
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
COLLECTION_NAME = "arxiv_papers"

BATCH_SIZE = 32


def get_embeddings(texts: list[str], prefix: str = "", max_retries: int = 3) -> list[list[float]]:
    """Generate embeddings via Ollama API with optional prefix."""
    url = f"http://{OLLAMA_HOST}/api/embed"

    # Add prefix for nomic-embed-text
    if prefix:
        texts = [f"{prefix}{t}" for t in texts]

    for attempt in range(max_retries):
        try:
            response = httpx.post(
                url,
                json={"model": EMBED_MODEL, "input": texts},
                timeout=120.0,
            )
            response.raise_for_status()
            return response.json()["embeddings"]
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"    Retry {attempt + 1}/{max_retries}: {e}")
                time.sleep(2)
            else:
                raise


def index_chunks():
    """Index all chunks into ChromaDB."""
    # Load chunks
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Filter out empty or very short texts
    chunks = [c for c in data["chunks"] if c.get("text") and len(c["text"].strip()) > 10]
    print(f"Loading {len(chunks)} chunks (filtered from {len(data['chunks'])})")

    # Connect to ChromaDB
    client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

    # Recreate collection (drop if exists)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # Batch indexing
    total_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE
    indexed = 0

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

        try:
            # Embed with document prefix
            embeddings = get_embeddings(texts, prefix="search_document: ")

            # Add to ChromaDB
            collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )

            indexed += len(batch)
            print(f"  Batch {batch_idx + 1}/{total_batches} indexed ({indexed}/{len(chunks)})")

        except Exception as e:
            print(f"  Batch {batch_idx + 1}/{total_batches} FAILED: {e}")
            continue

    # Verify
    count = collection.count()
    print(f"\n  Indexed {count} chunks in collection '{COLLECTION_NAME}'")

    # Test query (with query prefix)
    print("\n  Test query: 'What is Retrieval Augmented Generation?'")
    test_embedding = get_embeddings(
        ["What is Retrieval Augmented Generation?"],
        prefix="search_query: "
    )
    results = collection.query(
        query_embeddings=test_embedding,
        n_results=5,
    )
    for i, (doc, meta, dist) in enumerate(zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    )):
        print(f"\n  [{i+1}] dist={dist:.4f} | {meta['title'][:50]}")
        print(f"      Section: {meta['section']}")
        print(f"      Preview: {doc[:100]}...")


if __name__ == "__main__":
    index_chunks()