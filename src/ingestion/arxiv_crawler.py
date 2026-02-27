"""
arXiv Paper Crawler
- Search arXiv API for LLM/RAG related papers
- Save metadata (JSON)
- Download PDFs
"""

import arxiv
import json
import time
from datetime import datetime

from src.api.core.config import DATA_DIR


# ─── Config ─────────────────────────────────────────────
QUERIES = [
    "Retrieval Augmented Generation",
    "Large Language Model fine-tuning",
    "QLoRA",
    "LoRA low rank adaptation",
    "RAG evaluation",
    "vector database embedding",
    "LLM hallucination mitigation",
    "instruction tuning",
    "prompt engineering techniques",
    "small language model efficiency",
]

MAX_RESULTS_PER_QUERY = 15
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
METADATA_FILE = PROCESSED_DIR / "papers_metadata.json"


def search_papers(query: str, max_results: int = MAX_RESULTS_PER_QUERY) -> list[dict]:
    """Search arXiv API for papers matching the query with retry logic."""
    client = arxiv.Client(
        page_size=max_results,
        delay_seconds=5.0,
        num_retries=5,
    )
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    papers = []
    try:
        for result in client.results(search):
            paper = {
                "arxiv_id": result.entry_id.split("/")[-1],
                "title": result.title,
                "abstract": result.summary,
                "authors": [a.name for a in result.authors],
                "published": result.published.isoformat(),
                "updated": result.updated.isoformat(),
                "categories": result.categories,
                "pdf_url": result.pdf_url,
                "primary_category": result.primary_category,
                "query_source": query,
            }
            papers.append(paper)
    except Exception as e:
        print(f"   [ERROR] {e}")
        print(f"   Collected {len(papers)} papers before error")

    return papers


def deduplicate(papers: list[dict]) -> list[dict]:
    """Remove duplicates based on arxiv_id."""
    seen = set()
    unique = []
    for p in papers:
        if p["arxiv_id"] not in seen:
            seen.add(p["arxiv_id"])
            unique.append(p)
    return unique


def download_pdf(paper: dict, output_dir: Path) -> str | None:
    """Download PDF and return the file path."""
    filename = f"{paper['arxiv_id'].replace('/', '_')}.pdf"
    filepath = output_dir / filename

    if filepath.exists():
        print(f"  [SKIP] {filename} already exists")
        return str(filepath)

    try:
        import urllib.request
        urllib.request.urlretrieve(paper["pdf_url"], filepath)
        print(f"  [OK]   {filename}")
        return str(filepath)
    except Exception as e:
        print(f"  [FAIL] {filename}: {e}")
        return None


def main():
    # Create directories
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing metadata if available (resume support)
    existing_papers = []
    if METADATA_FILE.exists():
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            existing = json.load(f)
            existing_papers = existing.get("papers", [])
        print(f"  Found existing metadata with {len(existing_papers)} papers")

    # 1. Search papers
    print("Searching arXiv papers...")

    all_papers = list(existing_papers)
    for query in QUERIES:
        print(f"\n  Query: '{query}'")
        papers = search_papers(query)
        print(f"   Found {len(papers)} papers")
        all_papers.extend(papers)
        print(f"   Waiting 5s before next query...")
        time.sleep(5)  # Longer delay to avoid rate limits

    # Deduplicate
    unique_papers = deduplicate(all_papers)
    print(f"\n  Total: {len(all_papers)} -> Deduplicated: {len(unique_papers)}")

    # 2. Download PDFs
    print("Downloading PDFs...")

    for i, paper in enumerate(unique_papers):
        print(f"\n[{i+1}/{len(unique_papers)}] {paper['title'][:60]}...")
        pdf_path = download_pdf(paper, RAW_DIR)
        paper["local_pdf_path"] = pdf_path
        time.sleep(1)  # Download rate limit

    # 3. Save metadata
    successful = [p for p in unique_papers if p.get("local_pdf_path")]
    print(f"\n  Successfully downloaded: {len(successful)}/{len(unique_papers)}")

    metadata = {
        "collected_at": datetime.now().isoformat(),
        "total_papers": len(successful),
        "queries": QUERIES,
        "papers": successful,
    }

    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"  Metadata saved to {METADATA_FILE}")
    print(f"  PDFs saved to {RAW_DIR}")


if __name__ == "__main__":
    main()
