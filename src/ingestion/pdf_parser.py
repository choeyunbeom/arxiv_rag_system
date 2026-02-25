"""
PDF Parser
- Extract text from PDFs using PyMuPDF (fitz)
- Split into sections (Abstract, Introduction, Method, etc.)
- Combine with paper metadata
"""

import json
import re
from pathlib import Path

import fitz  # PyMuPDF


PROCESSED_DIR = Path("data/processed")
METADATA_FILE = PROCESSED_DIR / "papers_metadata.json"
PARSED_FILE = PROCESSED_DIR / "papers_parsed.json"


def extract_text_from_pdf(pdf_path: str) -> str | None:
    """Extract full text from a PDF file."""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text.strip()
    except Exception as e:
        print(f"  [FAIL] {pdf_path}: {e}")
        return None


def clean_text(text: str) -> str:
    """Clean extracted text."""
    # Remove excessive newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Rejoin hyphenated words (e.g., "lan-\nguage" -> "language")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # Replace single newlines with spaces (within paragraphs)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    # Collapse multiple spaces
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def split_sections(text: str) -> dict[str, str]:
    """Split paper text into sections (best-effort heuristic)."""
    section_patterns = [
        r"(?i)\babstract\b",
        r"(?i)\b1[\.\s]+introduction\b",
        r"(?i)\b2[\.\s]+(?:related\s+work|background)\b",
        r"(?i)\b3[\.\s]+(?:method|approach|model)\b",
        r"(?i)\b4[\.\s]+(?:experiment|evaluation|result)\b",
        r"(?i)\b5[\.\s]+(?:discussion|analysis)\b",
        r"(?i)\b(?:conclusion|summary)\b",
        r"(?i)\breferences\b",
    ]

    sections = {}
    positions = []

    for pattern in section_patterns:
        match = re.search(pattern, text)
        if match:
            positions.append((match.start(), match.group(), pattern))

    # Sort by position
    positions.sort(key=lambda x: x[0])

    for i, (start, name, _) in enumerate(positions):
        end = positions[i + 1][0] if i + 1 < len(positions) else len(text)
        section_text = text[start:end].strip()
        section_name = re.sub(r"[\d\.\s]+", " ", name).strip().lower()
        sections[section_name] = section_text

    # Fallback: return full text if section splitting fails
    if not sections:
        sections["full_text"] = text

    return sections


def parse_all_papers():
    """Parse all downloaded papers."""
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    parsed_papers = []
    total = len(metadata["papers"])

    print(f"Parsing {total} papers...\n")

    for i, paper in enumerate(metadata["papers"]):
        pdf_path = paper.get("local_pdf_path")
        if not pdf_path or not Path(pdf_path).exists():
            print(f"[{i+1}/{total}] SKIP - no PDF: {paper['title'][:50]}")
            continue

        print(f"[{i+1}/{total}] {paper['title'][:60]}...")

        raw_text = extract_text_from_pdf(pdf_path)
        if not raw_text:
            continue

        cleaned = clean_text(raw_text)
        sections = split_sections(cleaned)

        parsed = {
            "arxiv_id": paper["arxiv_id"],
            "title": paper["title"],
            "abstract": paper["abstract"],
            "authors": paper["authors"],
            "published": paper["published"],
            "categories": paper["categories"],
            "sections": sections,
            "full_text": cleaned,
            "char_count": len(cleaned),
            "word_count": len(cleaned.split()),
        }
        parsed_papers.append(parsed)

    # Save
    output = {
        "parsed_at": __import__("datetime").datetime.now().isoformat(),
        "total_parsed": len(parsed_papers),
        "papers": parsed_papers,
    }

    with open(PARSED_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n  Parsed {len(parsed_papers)}/{total} papers")
    print(f"  Saved to {PARSED_FILE}")


if __name__ == "__main__":
    parse_all_papers()
