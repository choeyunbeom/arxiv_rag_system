"""
PDF Parser (v3)
- Uses pymupdf4llm for proper two-column layout handling
- Section detection tuned for actual pymupdf4llm output format
  (bold numbered sections like **1** **Introduction**)
- Filters references at parse stage
"""

import json
import re
from datetime import datetime
from pathlib import Path

import pymupdf4llm

from src.api.core.config import DATA_DIR


PROCESSED_DIR = DATA_DIR / "processed"
METADATA_FILE = PROCESSED_DIR / "papers_metadata.json"
PARSED_FILE = PROCESSED_DIR / "papers_parsed.json"

# Sections to exclude at parse stage
EXCLUDED_SECTIONS = {
    "references", "bibliography", "appendix",
    "acknowledgments", "acknowledgements", "acknowledgement",
}


def extract_text_from_pdf(pdf_path: str) -> str | None:
    """Extract text from PDF using pymupdf4llm."""
    try:
        md_text = pymupdf4llm.to_markdown(pdf_path)
        return md_text.strip()
    except Exception as e:
        print(f"  [FAIL] {pdf_path}: {e}")
        return None


def clean_text(text: str) -> str:
    """Clean extracted markdown text."""
    # Remove image placeholders
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
    # Remove excessive newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse multiple spaces
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def split_sections(text: str) -> dict[str, str]:
    """
    Split paper into sections. Handles multiple header formats:
    1. Markdown headers: ## 1 Introduction
    2. Bold numbered: **1** **Introduction**
    3. Plain numbered: 1. Introduction / 1 Introduction
    """

    # Patterns ordered by priority (most specific first)
    section_markers = [
        # Pattern: **1** **Introduction** or **1.** **Introduction**
        (r"(?m)^\*\*\d+\.?\*\*\s*\*\*([Aa]bstract)\*\*", "abstract"),
        (r"(?m)^\*\*\d+\.?\*\*\s*\*\*([Ii]ntroduction)\*\*", "introduction"),
        (r"(?m)^\*\*\d+\.?\*\*\s*\*\*([Rr]elated\s*[Ww]ork|[Bb]ackground|[Pp]reliminar)\w*\*\*", "related_work"),
        (r"(?m)^\*\*\d+\.?\*\*\s*\*\*([Mm]ethod|[Aa]pproach|[Mm]odel|[Ff]ramework|[Ss]ystem|[Pp]roposed)\w*\*\*", "method"),
        (r"(?m)^\*\*\d+\.?\*\*\s*\*\*([Ee]xperiment|[Ee]valuation|[Rr]esult|[Ss]etup)\w*\*\*", "experiments"),
        (r"(?m)^\*\*\d+\.?\*\*\s*\*\*([Dd]iscussion|[Aa]nalysis|[Aa]blation)\w*\*\*", "discussion"),
        (r"(?m)^\*\*\d+\.?\*\*\s*\*\*([Cc]onclusion|[Ss]ummary)\w*\*\*", "conclusion"),
        (r"(?m)^\*\*\d+\.?\*\*\s*\*\*([Rr]eferences|[Bb]ibliography)\*\*", "references"),
        (r"(?m)^\*\*\d+\.?\*\*\s*\*\*([Aa]cknowledg)\w*\*\*", "acknowledgments"),
        (r"(?m)^\*\*\d+\.?\*\*\s*\*\*([Aa]ppendix)\w*\*\*", "appendix"),

        # Pattern: ## 1. Introduction or ## Introduction
        (r"(?m)^#+\s*(?:\d+\.?\s+)?[Aa]bstract", "abstract"),
        (r"(?m)^#+\s*(?:\d+\.?\s+)?[Ii]ntroduction", "introduction"),
        (r"(?m)^#+\s*(?:\d+\.?\s+)?(?:[Rr]elated\s*[Ww]ork|[Bb]ackground)", "related_work"),
        (r"(?m)^#+\s*(?:\d+\.?\s+)?(?:[Mm]ethod|[Aa]pproach|[Mm]odel|[Ff]ramework)", "method"),
        (r"(?m)^#+\s*(?:\d+\.?\s+)?(?:[Ee]xperiment|[Ee]valuation|[Rr]esult)", "experiments"),
        (r"(?m)^#+\s*(?:\d+\.?\s+)?(?:[Dd]iscussion|[Aa]nalysis|[Aa]blation)", "discussion"),
        (r"(?m)^#+\s*(?:\d+\.?\s+)?(?:[Cc]onclusion|[Ss]ummary)", "conclusion"),
        (r"(?m)^#+\s*(?:\d+\.?\s+)?(?:[Rr]eferences|[Bb]ibliography)", "references"),
        (r"(?m)^#+\s*(?:\d+\.?\s+)?[Aa]cknowledg", "acknowledgments"),
        (r"(?m)^#+\s*(?:\d+\.?\s+)?[Aa]ppendix", "appendix"),

        # Pattern: standalone **Abstract** or **References**
        (r"(?m)^\*\*[Aa]bstract\*\*", "abstract"),
        (r"(?m)^\*\*[Rr]eferences\*\*", "references"),
        (r"(?m)^\*\*[Aa]cknowledg\w*\*\*", "acknowledgments"),

        # Pattern: plain text headers (last resort)
        (r"(?m)^Abstract\s*$", "abstract"),
        (r"(?m)^References\s*$", "references"),
    ]

    positions = []
    seen_sections = set()

    for pattern, name in section_markers:
        match = re.search(pattern, text)
        if match and name not in seen_sections:
            positions.append((match.start(), name))
            seen_sections.add(name)

    # Sort by position
    positions.sort(key=lambda x: x[0])

    if not positions:
        return {"full_text": text}

    sections = {}
    for i, (start, name) in enumerate(positions):
        end = positions[i + 1][0] if i + 1 < len(positions) else len(text)
        section_text = text[start:end].strip()
        if len(section_text) > 50:
            sections[name] = section_text

    # If no content sections found, fallback
    content_sections = {k: v for k, v in sections.items() if k not in EXCLUDED_SECTIONS}
    if not content_sections:
        return {"full_text": text}

    return sections


def parse_all_papers():
    """Parse all downloaded papers."""
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    parsed_papers = []
    total = len(metadata["papers"])
    section_stats = {"sectioned": 0, "full_text_only": 0}

    print(f"Parsing {total} papers with pymupdf4llm...\n")

    for i, paper in enumerate(metadata["papers"]):
        pdf_path = paper.get("local_pdf_path")
        if not pdf_path or not Path(pdf_path).exists():
            print(f"[{i+1}/{total}] SKIP - no PDF: {paper['title'][:50]}")
            continue

        print(f"[{i+1}/{total}] {paper['title'][:60]}...", end="")

        raw_text = extract_text_from_pdf(pdf_path)
        if not raw_text:
            continue

        cleaned = clean_text(raw_text)
        sections = split_sections(cleaned)

        # Track stats
        if "full_text" in sections:
            section_stats["full_text_only"] += 1
            print(f" [full_text]")
        else:
            section_stats["sectioned"] += 1
            print(f" [{', '.join(sections.keys())}]")

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
        "parsed_at": datetime.now().isoformat(),
        "total_parsed": len(parsed_papers),
        "papers": parsed_papers,
    }

    with open(PARSED_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n  Parsed {len(parsed_papers)}/{total} papers")
    print(f"  Sectioned: {section_stats['sectioned']}, Full text only: {section_stats['full_text_only']}")
    print(f"  Saved to {PARSED_FILE}")


if __name__ == "__main__":
    parse_all_papers()