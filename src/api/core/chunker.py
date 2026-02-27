"""
Text Chunker (v5)
- Split parsed paper text into chunks for RAG retrieval
- Section-aware chunking (does not cross section boundaries)
- Filters out references, appendix, and acknowledgments
- For full_text sections: strips everything after references header
- Strong citation/bibliography content detection and filtering
- Quality filtering for meaningful chunks
"""

import json
import re
from transformers import AutoTokenizer

_tokenizer = AutoTokenizer.from_pretrained("mixedbread-ai/mxbai-embed-large-v1")
import hashlib
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime


PROCESSED_DIR = Path("data/processed")
PARSED_FILE = PROCESSED_DIR / "papers_parsed.json"
CHUNKS_FILE = PROCESSED_DIR / "chunks.json"

CHUNK_SIZE = 450  # tokens, not words (model limit: 512)
CHUNK_OVERLAP = 64

EXCLUDED_SECTIONS = {
    "references", "bibliography", "appendix",
    "acknowledgments", "acknowledgements",
}


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
    raw = f"{arxiv_id}:{section}:{index}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def strip_references_from_text(text: str) -> str:
    """Remove everything after the references section in full_text."""
    # Header-based detection
    patterns = [
        r"(?m)^\*\*\d*\.?\*\*\s*\*\*[Rr]eferences\*\*",
        r"(?m)^#+\s*(?:\d+\.?\s+)?[Rr]eferences",
        r"(?m)^\*\*[Rr]eferences\*\*",
        r"(?m)^[Rr]eferences\s*$",
        r"(?m)^\*\*\d*\.?\*\*\s*\*\*[Bb]ibliography\*\*",
        r"(?m)^#+\s*(?:\d+\.?\s+)?[Bb]ibliography",
        r"(?m)^\*\*[Aa]cknowledg\w*\*\*",
    ]

    earliest_pos = len(text)
    for pattern in patterns:
        match = re.search(pattern, text)
        if match and match.start() < earliest_pos:
            earliest_pos = match.start()

    if earliest_pos < len(text):
        return text[:earliest_pos].strip()

    return text


def is_citation_text(text: str) -> bool:
    """Detect if text is primarily citation/bibliography content."""
    # Count DOI/URL patterns
    doi_count = len(re.findall(r"https?://doi\.org|doi\.org|arxiv\.org", text))
    
    # Count year patterns like (2020) or 2020.
    year_count = len(re.findall(r"\b(19|20)\d{2}\b", text))
    
    # Count patterns like "Author Name. 2023." or "Author Name, 2023"
    author_year = len(re.findall(r"[A-Z][a-z]+[\.,]\s*(19|20)\d{2}", text))
    
    # Count "In _Conference" or "In Proceedings" patterns
    conf_count = len(re.findall(r"(?:In\s+_|In\s+Proc|In\s+\*)", text))
    
    # Count "pp." or "pages" patterns
    page_count = len(re.findall(r"\bpp\.\s*\d|pages?\s*\d", text, re.IGNORECASE))

    words = text.split()
    word_count = len(words)
    
    if word_count == 0:
        return True

    # Strong signals: many DOIs or conference references
    if doi_count >= 2:
        return True
    if conf_count >= 2:
        return True
    
    # Combined signal: multiple year references + author-year patterns
    citation_score = doi_count * 3 + author_year * 2 + conf_count * 2 + page_count * 2 + year_count
    
    # If citation density is high relative to text length
    if citation_score > 8 and word_count < 300:
        return True
    if citation_score > 15:
        return True

    return False


def clean_chunk_text(text: str) -> str:
    """Aggressively clean text for embedding quality."""
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
    text = re.sub(r"\[([^\]]*)\]\(.*?\)", r"\1", text)
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"[�\ufffd]", "", text)
    # Remove markdown table remnants
    text = re.sub(r"\|[^|\n]{0,30}\|[^|\n]{0,30}\|", "", text)
    text = re.sub(r"\|Col\d+", "", text)
    text = re.sub(r"^\|.*\|\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[-|:]+$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\[[\w\s]*\]", "", text)
    text = re.sub(r"_([^_]{1,3})_", "", text)
    text = re.sub(r"\\[a-zA-Z]+\{[^}]*\}", "", text)
    text = re.sub(r"\$[^$]*\$", "", text)

    lines = text.split("\n")
    clean_lines = []
    for line in lines:
        alpha_count = sum(1 for c in line if c.isalnum() or c.isspace())
        if len(line) > 0 and alpha_count / len(line) > 0.5:
            clean_lines.append(line)
        elif len(line.strip()) == 0:
            clean_lines.append("")

    text = "\n".join(clean_lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)

    return text.strip()


def is_quality_chunk(text: str, min_words: int = 30) -> bool:
    """Check if a chunk has enough meaningful content."""
    words = text.split()
    if len(words) < min_words:
        return False

    alpha_words = sum(1 for w in words if any(c.isalpha() for c in w))
    if len(words) > 0 and alpha_words / len(words) < 0.6:
        return False

    # Reject citation-heavy chunks
    if is_citation_text(text):
        return False

    return True


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into chunks based on token count (not word count) to respect model context limits."""
    tokens = _tokenizer.encode(text, add_special_tokens=False)

    if len(tokens) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        decoded = _tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        if len(decoded.split()) >= 10:
            chunks.append(decoded.strip())
        start += chunk_size - overlap

    return chunks


def chunk_paper(paper: dict) -> list[Chunk]:
    chunks = []
    sections = paper.get("sections", {})

    if not sections or "full_text" in sections:
        full = paper.get("full_text", "")
        full = strip_references_from_text(full)
        sections = {"full_text": full}

    for section_name, section_text in sections.items():
        if section_name.lower() in EXCLUDED_SECTIONS:
            continue

        if not section_text:
            continue

        cleaned = clean_chunk_text(section_text)

        if len(cleaned.split()) < 20:
            continue

        text_chunks = chunk_text(cleaned)

        for i, text in enumerate(text_chunks):
            if not is_quality_chunk(text):
                continue

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
    with open(PARSED_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_chunks = []
    skipped_papers = 0

    for paper in data["papers"]:
        paper_chunks = chunk_paper(paper)
        if paper_chunks:
            all_chunks.extend(paper_chunks)
            print(f"[{paper['arxiv_id']}] {len(paper_chunks)} chunks - {paper['title'][:50]}")
        else:
            skipped_papers += 1

    output = {
        "chunked_at": datetime.now().isoformat(),
        "total_chunks": len(all_chunks),
        "chunk_config": {
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "excluded_sections": list(EXCLUDED_SECTIONS),
        },
        "chunks": [asdict(c) for c in all_chunks],
    }

    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n  Total chunks: {len(all_chunks)}")
    print(f"  Skipped papers (no quality chunks): {skipped_papers}")
    print(f"  Saved to {CHUNKS_FILE}")


if __name__ == "__main__":
    chunk_all_papers()