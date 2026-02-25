"""
ChromaDB Indexer
- Vectorise chunks using Ollama embeddings
- Store in ChromaDB with metadata
- Supports filtered retrieval
"""

import json
import os
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

BATCH_SIZE = 32  # ChromaDB batch size


def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Generate embeddings via Ollama API."""
    url = f"http://{OLLAMA_HOST}/api/embed"
    response = httpx.post(
        url,
        json={"model": EMBED_MODEL, "input": texts},
        timeout=120.0,
    )
    response.raise_for_status()
    return response.json()["embeddings"]


def index_chunks():
    """Index all chunks into ChromaDB."""
    # Load chunks
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    chunks = data["chunks"]
    print(f"Loading {len(chunks)} chunks...")

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

    for batch_idx in range(total_batches):
        start = batch_idx * BATCH_SIZE
        end = min(start + BATCH_SIZE, len(chunks))
        batch = chunks[start:end]

        texts = [c["text"] for c in batch]
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

        # Generate embeddings
        embeddings = get_embeddings(texts)

        # Add to ChromaDB
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

        print(f"  Batch {batch_idx + 1}/{total_batches} indexed ({end}/{len(chunks)})")

    # Verify
    count = collection.count()
    print(f"\n  Indexed {count} chunks in collection '{COLLECTION_NAME}'")

    # Test query
    print("\n  Test query: 'What is Retrieval Augmented Generation?'")
    test_embedding = get_embeddings(["What is Retrieval Augmented Generation?"])
    results = collection.query(
        query_embeddings=test_embedding,
        n_results=3,
    )
    for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
        print(f"\n  [{i+1}] {meta['title'][:60]}")
        print(f"      Section: {meta['section']}")
        print(f"      Preview: {doc[:100]}...")


if __name__ == "__main__":
    index_chunks()
