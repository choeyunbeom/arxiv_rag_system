"""
Precompute UMAP 2D coordinates for all existing embeddings in ChromaDB.
This script will output `data/processed/umap_bg.json` containing the background points
and `data/processed/umap_model.pkl` containing the fitted UMAP reducer for transforming
new query embeddings in real time.
"""
import json
import os
import pickle
from pathlib import Path

import chromadb
import numpy as np
import umap

from src.api.core.config import DATA_DIR, settings

UMAP_MODEL_PATH = DATA_DIR / "processed" / "umap_model.pkl"
UMAP_BG_JSON_PATH = DATA_DIR / "processed" / "umap_bg.json"


def main():
    print(f"Connecting to ChromaDB at {settings.CHROMA_HOST}:{settings.CHROMA_PORT}...")
    chroma_client = chromadb.HttpClient(host=settings.CHROMA_HOST, port=settings.CHROMA_PORT)

    try:
        collection = chroma_client.get_collection(settings.COLLECTION_NAME)
    except Exception as e:
        print(f"Error accessing collection: {e}")
        return

    print("Fetching all embeddings from ChromaDB (this may take a few seconds)...")
    # ChromaDB get() without arguments returns all entries
    data = collection.get(include=["embeddings", "metadatas", "documents"])

    embeddings = data.get("embeddings")
    if embeddings is None or len(embeddings) == 0:
        print("No embeddings found in the collection.")
        return
    ids = data["ids"]
    metadatas = data["metadatas"]
    documents = data["documents"]

    print(f"Loaded {len(embeddings)} embeddings.")
    # Fit UMAP
    print("Fitting UMAP model (3D) (this may take a minute depending on count)...")
    reducer = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42)
    # converting to numpy array
    X = np.array(embeddings)
    embedding_2d = reducer.fit_transform(X)
    print("UMAP fitted.")
    # Save the model
    UMAP_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(UMAP_MODEL_PATH, "wb") as f:
        pickle.dump(reducer, f)
    print(f"UMAP model saved to {UMAP_MODEL_PATH}")

    # Save background points
    bg_data = []
    for i in range(len(ids)):
        meta = metadatas[i] or {}
        # Authors are typically a list in metadata, but Chroma returns simple types
        # depending on version, so we handle it safely.
        title = meta.get("title", "Unknown Title")
        section = meta.get("section", "Unknown Section")
        # truncated document text for tooltip
        text = documents[i]
        text_preview = " ".join(text.split()[:30]) + "..."
        bg_data.append({
            "chunk_id": ids[i],
            "x": float(embedding_2d[i][0]),
            "y": float(embedding_2d[i][1]),
            "z": float(embedding_2d[i][2]),
            "title": title,
            "section": section,
            "text_preview": text_preview
        })
    with open(UMAP_BG_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(bg_data, f, ensure_ascii=False)
    print(f"Background coordinates saved to {UMAP_BG_JSON_PATH}")
    print("Done!")

if __name__ == "__main__":
    main()
