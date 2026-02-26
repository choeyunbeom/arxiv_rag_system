# Weekly Development Log

## Week 1 — Infrastructure & Data Pipeline

### Day 1 (2025-02-25)

#### Environment Setup
- Installed Ollama natively on host (not Docker) to leverage Apple Silicon Metal GPU acceleration
- Downloaded models: Qwen3 4B (main LLM), nomic-embed-text (embedding — later replaced)
- Verified Metal GPU detection: Apple M4 Pro, 37.4 GiB VRAM available
- Set up Python 3.11 virtual environment using `uv` package manager
- Installed Docker Desktop, confirmed ChromaDB running on port 8200
- Configured Git + SSH authentication, first push to GitHub

#### Project Scaffold
- Created full project structure: `src/api/`, `src/ingestion/`, `src/evaluation/`, `tests/`, `docs/`, `scripts/`
- Configured `.env.example`, `.gitignore`, `pyproject.toml`
- Initial `README.md` with tech stack overview

#### Data Collection (`src/ingestion/arxiv_crawler.py`)
- Built arXiv API crawler with 10 search queries (RAG, QLoRA, LoRA, hallucination, etc.)
- Collected 132 unique papers after deduplication (from 150 raw results)
- Downloaded all 132 PDFs successfully
- Encountered arXiv API rate limit (HTTP 429) — added retry logic with `delay_seconds=5.0`, `num_retries=5`, and 5s inter-query delay
- Added resume support: if metadata file exists, appends new results instead of starting over

#### PDF Parsing (`src/ingestion/pdf_parser.py`)
- **v1**: Used `pymupdf` `page.get_text()` — failed on two-column arXiv layouts (text interleaving)
- **v2**: Migrated to `pymupdf4llm.to_markdown()` — proper two-column handling, Markdown output
- **v3**: Tuned section detection for actual pymupdf4llm output format
  - arXiv papers use bold numbered headers (`**1** **Introduction**`), not Markdown `#` headers
  - Built regex patterns for both formats: bold numbered, Markdown `#`, and standalone `**Abstract**`
  - Result: 82/132 papers successfully sectioned, 50 fell back to full_text

#### Chunking (`src/api/core/chunker.py`)
- Section-aware chunking: does not cross section boundaries
- Excluded sections: References, Bibliography, Appendix, Acknowledgments
- Stripped references from unsectioned `full_text` using header detection + citation pattern matching
- **Citation content detection**: scores chunks based on DOI count, author-year patterns, conference mentions, page references — rejects citation-heavy chunks
- **Text cleaning**: removes broken Unicode, LaTeX remnants, math placeholders, low-content lines
- **Quality filter**: minimum 30 words, 60%+ alphabetic words required
- Final chunk size: 128 words (tuned for mxbai-embed-large context window)

#### Indexing (`src/ingestion/indexer.py`)
- ChromaDB with cosine distance metric
- Batch indexing (32 chunks per batch) with retry logic
- Failed batches are skipped (not crash) — ensures pipeline completes
- Metadata stored per chunk: arxiv_id, title, section, authors, published date

#### Embedding Model Debugging (Critical Learning)

This was the most significant engineering decision of the day.

**Problem**: After indexing ~4000 chunks, test query `"What is Retrieval Augmented Generation?"` returned completely irrelevant results (Software Engineering, Web Engineering papers).

**Debugging steps**:

1. Verified RAG-related chunks existed in the index ✓
2. Checked retrieval distances — all 0.34–0.40 (too high) ✓
3. **Sanity check**: computed direct cosine similarity between query and known-relevant vs known-irrelevant chunks

**Results with `nomic-embed-text`**:

| Pair | Cosine Similarity |
|------|-------------------|
| Query ↔ RAG chunk | 0.41 |
| Query ↔ Irrelevant chunk | 0.60 |

The irrelevant document scored **higher** — the vector space was inverted.

**Results with `mxbai-embed-large`**:

| Pair | Cosine Similarity |
|------|-------------------|
| Query ↔ RAG chunk | **0.76** |
| Query ↔ Irrelevant chunk | **0.49** |

Correct ranking restored. Switched to `mxbai-embed-large` and re-indexed.

**Root cause**: `nomic-embed-text` in Ollama's GGUF/quantised format does not properly handle task-specific retrieval behaviour. The Hugging Face version may work, but the Ollama-served variant produces degraded embeddings.

**Final retrieval results**:
```
[1] dist=0.1668 | RAGPart & RAGMask: Retrieval-Stage Defenses Against...
[2] dist=0.1724 | RAG-Gym: Systematic Optimization of Language Agents...
[3] dist=0.1840 | Engineering the RAG Stack: A Comprehensive Review...
[4] dist=0.1853 | MultiHop-RAG: Benchmarking Retrieval-Augmented Gen...
[5] dist=0.1909 | T-RAG: Lessons from the LLM Trenches
```

**Key takeaway**: Never trust an embedding model blindly. A 3-line cosine similarity test caught a failure that would have made the entire RAG system useless.

> For full debugging data and analysis, see [Embedding Model Debugging Log](embedding_model_debugging.md).

#### End of Day Status
- 132 papers collected, parsed, chunked, and indexed
- ~3500 quality-filtered chunks in ChromaDB
- Test queries returning accurate results (distance 0.16–0.19)
- All code pushed to GitHub

---

### Day 2 (2025-02-26)

#### RAG Pipeline (`src/api/core/`)

Built the core question-answering pipeline with three modules:

**Retriever** (`retriever.py`)
- Wraps ChromaDB collection with query embedding via `mxbai-embed-large`
- Returns ranked `RetrievedChunk` objects with text, metadata, and cosine distance
- Configurable `top_k` parameter (default: 5)

**LLM Client** (`llm_client.py`)
- Thin wrapper around Ollama `/api/generate` endpoint
- Calls Qwen3 4B with configurable temperature (default: 0.3 for factual answers)
- Added `/no_think` flag to suppress Qwen3's thinking mode — without this, the model would reason inside `<think>` tags and return an empty response for long prompts

**RAG Chain** (`rag_chain.py`)
- Orchestrates the full pipeline: retrieve → format context → generate answer
- System prompt constrains LLM to answer ONLY from retrieved context and cite paper titles
- Context template formats each chunk with paper title, section, and content
- Deduplicates sources by `arxiv_id` (same paper may contribute multiple chunks)

#### FastAPI Application (`src/api/`)

**Endpoints**:
- `POST /query` — accepts `{question, top_k}`, returns `{answer, sources, query}`
- `GET /health` — checks Ollama and ChromaDB connectivity, returns collection count
- `GET /` — root info endpoint

**Supporting files**:
- `models/schemas.py` — Pydantic models for request/response validation
- `routers/query.py` — query endpoint router
- `routers/health.py` — health check router
- `core/config.py` — centralised settings from environment variables

**Validation**:
- Swagger UI confirmed at `http://localhost:8000/docs`
- Test query `"What is QLoRA and how does it work?"` returned accurate answer citing the QLoRA paper (distance 0.32)
- Sources correctly ranked: QLoRA original paper → clinical QLoRA application → radiology QLoRA → parliamentary QLoRA

#### Qwen3 Thinking Mode Fix

**Problem**: Qwen3 4B has a "thinking mode" where it reasons inside `<think>...</think>` tags before answering. For long RAG prompts, the model would spend all its tokens thinking and return an empty `response` field.

**Solution**: Appended `/no_think` flag to all prompts, which disables the internal reasoning and forces direct answers. This is a Qwen3-specific feature.

#### End of Day Status
- Full RAG pipeline operational: question → retrieval → LLM answer + cited sources
- FastAPI serving on port 8000 with Swagger UI
- All code pushed to GitHub

---

## Week 1 — Day 3 (TBD)

**Planned**:
- Streamlit UI for interactive demo
- Additional testing and edge cases

---

## Week 2 (TBD)

**Planned**:
- Evaluation metrics (RAGAS or custom)
- Baseline performance benchmarks
- Reranking experiments

---

## Week 3 (TBD)

**Planned**:
- QLoRA fine-tuning on synthetic Q&A dataset
- Compare base vs fine-tuned model performance

---

## Week 4 (TBD)

**Planned**:
- Evaluation pipeline with quantitative metrics
- Unit + integration tests
- CI/CD setup
- Final README and documentation
