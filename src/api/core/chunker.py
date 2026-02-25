"""
Text Chunker
- Split parsed paper text into chunks for RAG retrieval
- Section-aware chunking (does not cross section boundaries)
- Overlap to preserve context
"""

import json
import hashlib
from pathlib import Path
from dataclasses import dataclass, asdict


PROCESSED_DIR = Path("data/processed")
PARSED_FILE = PROCESSED_DIR / "papers_parsed.json"
CHUNKS_FILE = PROCESSED_DIR / "chunks.json"

CHUNK_SIZE = 512  # Approximate token count (word-based)
CHUNK_OVERLAP = 64


@dataclass
class Chunk:
    chunk_id: str
    arxiv_id: str
    title: str
    section: str
    text: str
    word_count: int
    metadata: dict


def generate_chunk_id(arxiv_id: str, section: str, index: int) -> str:
    """Generate a deterministic chunk ID."""
    raw = f"{arxiv_id}:{section}:{index}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into chunks by word count."""
    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def chunk_paper(paper: dict) -> list[Chunk]:
    """Split a single paper into chunks."""
    chunks = []
    sections = paper.get("sections", {})

    # Fallback to full text if no sections found
    if not sections or "full_text" in sections:
        sections = {"full_text": paper.get("full_text", "")}

    for section_name, section_text in sections.items():
        if not section_text or len(section_text.split()) < 20:
            continue

        text_chunks = chunk_text(section_text)

        for i, text in enumerate(text_chunks):
            chunk_id = generate_chunk_id(paper["arxiv_id"], section_name, i)
            chunk = Chunk(
                chunk_id=chunk_id,
                arxiv_id=paper["arxiv_id"],
                title=paper["title"],
                section=section_name,
                text=text,
                word_count=len(text.split()),
                metadata={
                    "authors": paper.get("authors", []),
                    "published": paper.get("published", ""),
                    "categories": paper.get("categories", []),
                    "chunk_index": i,
                    "total_chunks_in_section": len(text_chunks),
                },
            )
            chunks.append(chunk)

    return chunks


def chunk_all_papers():
    """Chunk all parsed papers."""
    with open(PARSED_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_chunks = []
    for paper in data["papers"]:
        paper_chunks = chunk_paper(paper)
        all_chunks.extend(paper_chunks)
        print(f"[{paper['arxiv_id']}] {len(paper_chunks)} chunks - {paper['title'][:50]}")

    # Save
    output = {
        "chunked_at": __import__("datetime").datetime.now().isoformat(),
        "total_chunks": len(all_chunks),
        "chunk_config": {
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
        },
        "chunks": [asdict(c) for c in all_chunks],
    }

    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n  Total chunks: {len(all_chunks)}")
    print(f"  Saved to {CHUNKS_FILE}")


if __name__ == "__main__":
    chunk_all_papers()
