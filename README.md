# arXiv RAG System

A Retrieval-Augmented Generation system for querying academic papers from arXiv. Built as a portfolio project demonstrating end-to-end ML engineering: data pipeline, hybrid retrieval, LLM fine-tuning, and systematic evaluation.

Ask a question in natural language → the system retrieves relevant papers → generates a cited, grounded answer.

## Demo

![Query → answer flow with source citations and latency breakdown](docs/ui_demo.gif)
![Streamlit UI — interactive Q&A interface with source cards and latency breakdown](docs/main_demo.png)
![FastAPI Swagger UI — interactive API documentation with example requests and responses](docs/swagger_demo.png)

> The UI shows the full query flow: enter a question → hybrid retrieval searches 132 arXiv papers → cross-encoder reranks results → Qwen3 4B generates a cited answer. Latency breakdown shows retrieval vs generation time.

## Architecture

```
                          ┌──────────────────────────────────────┐
                          │              User Query              │
                          └──────────────────┬───────────────────┘
                                             │
                          ┌──────────────────▼───────────────────┐
                          │           FastAPI Backend            │
                          │         POST /query {question}       │
                          └──────────────────┬───────────────────┘
                                             │
                     ┌───────────────────────┼───────────────────┐
                     │                       │                   │
          ┌──────────▼──────────┐ ┌──────────▼──────────┐        │
          │   ChromaDB Vector   │ │    BM25 Keyword     │        │
          │   Search (Top-40)   │ │   Search (Top-40)   │        │
          │  mxbai-embed-large  │ │     rank_bm25       │        │
          └──────────┬──────────┘ └──────────┬──────────┘        │
                     │                       │                   │
                     └───────────┬───────────┘                   │
                                 │                               │
                     ┌───────────▼───────────┐                   │
                     │  Reciprocal Rank      │                   │
                     │  Fusion (k=60)        │                   │
                     └───────────┬───────────┘                   │
                                 │                               │
                     ┌───────────▼───────────┐                   │
                     │  Cross-Encoder        │                   │
                     │  Reranker (Top-5)     │                   │
                     │  ms-marco-MiniLM-L6   │                   │
                     └───────────┬───────────┘                   │
                                 │                               │
                     ┌───────────▼───────────┐                   │
                     │  Deduplication by     │                   │
                     │  arxiv_id::section    │                   │
                     └───────────┬───────────┘                   │
                                 │                               │
                          ┌──────▼───────────────────────────────▼──┐
                          │         Qwen3 4B (via Ollama)           │
                          │    System prompt + Retrieved context    │
                          │         → Cited answer generation       │
                          └──────────────────┬──────────────────────┘
                                             │
                          ┌──────────────────▼──────────────────────┐
                          │         Streamlit Frontend              │
                          │  Answer + Source cards with arXiv links │
                          └─────────────────────────────────────────┘
```

## Key Results

### Retrieval Optimisation

| Stage | Hit Rate | MRR | Key Change |
|-------|----------|-----|------------|
| Baseline | 60% | 0.51 | 128-word chunks, dense vector only |
| + Chunk optimisation | 67% | 0.42 | 200-word chunks, fault-tolerant indexer |
| + BM25 Hybrid Search | 73% | 0.52 | Reciprocal Rank Fusion with keyword search |
| **+ Reranker + Dedup** | **100%** | **0.82** | Cross-encoder reranking, section-level dedup |

### Prompt Engineering vs Fine-Tuning

Compared three answer generation strategies on the same 15-question benchmark under identical retrieval conditions:

| Metric | Zero-Shot | Few-Shot | Fine-Tuned |
|--------|-----------|----------|------------|
| Keyword Coverage | 76.4% | **78.0%** | 48.0% |
| Source Hit Rate | 100% | 100% | 100% |
| Substantive Rate | 100% | 100% | 100% |
| Avg Word Count | 175 | 177 | 1,614 |
| Avg Latency | 20.0s | 20.8s | 47.7s |

Few-shot prompt engineering outperformed both zero-shot and fine-tuning. The fine-tuned model suffered from training data contamination — see [Fine-Tuning Analysis](#why-fine-tuning-didnt-improve-metrics) for the full breakdown.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Qwen3 4B (Ollama, Apple Silicon Metal) |
| Embeddings | mxbai-embed-large (Ollama) |
| Vector Store | ChromaDB (Docker) |
| Sparse Search | rank_bm25 |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Backend | FastAPI |
| Frontend | Streamlit |
| Fine-Tuning | LoRA via PEFT + trl (SFTTrainer) |
| CI/CD | GitHub Actions (ruff + pytest) |
| Deployment | Docker Compose |
| Testing | pytest (104 tests) |
| Config | Pydantic Settings |

## Project Structure

```
arxiv_rag_system/
├── .github/workflows/ci.yml        # GitHub Actions: lint + test on push/PR
├── src/
│   ├── api/
│   │   ├── core/
│   │   │   ├── config.py            # Centralised Pydantic settings
│   │   │   ├── hybrid_retriever.py  # Dense + BM25 + reranker pipeline
│   │   │   ├── llm_client.py        # Ollama API wrapper
│   │   │   ├── rag_chain.py         # Retrieval → Generation orchestrator
│   │   │   ├── prompts.py           # Prompt templates (zero-shot, few-shot)
│   │   │   └── chunker.py           # Token-aware chunking with quality filters
│   │   ├── models/schemas.py        # Request/response Pydantic models
│   │   ├── routers/
│   │   │   ├── query.py             # POST /query endpoint
│   │   │   └── health.py            # GET /health endpoint
│   │   └── main.py                  # App entry with lifespan pre-loading
│   ├── ingestion/
│   │   ├── arxiv_crawler.py         # arXiv API crawler with retry logic
│   │   ├── pdf_parser.py            # PDF → Markdown (pymupdf4llm)
│   │   └── indexer.py               # ChromaDB batch indexer
│   ├── evaluation/
│   │   ├── eval_dataset.py          # 15-question benchmark dataset
│   │   └── evaluate.py              # Automated retrieval + answer metrics
│   └── finetuning/
│       ├── generate_qa_dataset.py   # Synthetic Q&A generation pipeline
│       └── finetune_lora.ipynb      # LoRA training notebook
├── scripts/
│   └── run_fewshot_experiment.py    # 3-way prompt comparison orchestrator
├── tests/
│   ├── test_chunker.py              # 33 unit tests
│   ├── test_llm_client.py           # 16 unit tests
│   ├── test_rag_chain.py            # 17 unit tests
│   ├── test_hybrid_retriever.py     # 19 unit tests
│   └── test_api_integration.py      # 19 integration tests
├── ui/app.py                        # Streamlit frontend
├── data/
│   ├── raw/                         # 132 arXiv PDFs
│   ├── processed/                   # Chunks, metadata, eval results
│   ├── base_model/                  # Qwen3-4B weights (git-ignored)
│   └── finetuned_lora/              # LoRA adapter + checkpoints (git-ignored)
├── Dockerfile.api                   # FastAPI backend container
├── Dockerfile.ui                    # Streamlit frontend container
├── docker-compose.yml               # Full-stack deployment
├── Makefile                         # Dev shortcuts
└── docs/                            # Detailed experiment logs
```

## Setup

### Prerequisites

- Python 3.11+
- Docker Desktop
- Ollama (for LLM + embeddings)

### Quick Start (Docker Compose)

```bash
# Clone the repository
git clone https://github.com/choeyunbeom/arxiv_rag_system.git
cd arxiv_rag_system

# Pull Ollama models (must run on host for Metal GPU)
ollama pull qwen3:4b
ollama pull mxbai-embed-large

# Start all services
docker compose up --build
```

The API is available at `http://localhost:8000/docs` and the UI at `http://localhost:8501`.

### Local Development

```bash
# Create virtual environment
uv venv && source .venv/bin/activate

# Install dependencies
uv pip install -e ".[dev]"

# Start ChromaDB only
docker compose up chromadb -d

# Start the API server
uvicorn src.api.main:app --reload

# Start the UI (in another terminal)
streamlit run ui/app.py

# Run tests
pytest tests/ -v
```

### Data Pipeline

```bash
# 1. Crawl arXiv papers
python -m src.ingestion.arxiv_crawler

# 2. Parse PDFs to Markdown
python -m src.ingestion.pdf_parser

# 3. Index chunks into ChromaDB
python -m src.ingestion.indexer
```

## Engineering Decisions

### Embedding Model Selection

Initial choice of `nomic-embed-text` via Ollama produced an inverted vector space — irrelevant documents scored higher than relevant ones. A 3-line cosine similarity sanity check caught the failure:

| Pair | nomic-embed-text | mxbai-embed-large |
|------|-----------------|-------------------|
| Query ↔ Relevant chunk | 0.41 | **0.76** |
| Query ↔ Irrelevant chunk | **0.60** | 0.49 |

Root cause: Ollama's GGUF-quantised `nomic-embed-text` does not preserve task-specific retrieval behaviour from the original Hugging Face model. Switched to `mxbai-embed-large` which correctly ranks relevant documents higher.

**Lesson**: Never trust an embedding model without a basic sanity check. Three lines of code prevented a completely broken RAG system.

### Token-Based Chunking

Word-count chunking (200 words) caused 2.2% of chunks to fail embedding due to token overflow. Academic text has a 2.27x token-to-word ratio (vs 1.27x for normal text) because of LaTeX, markdown tables, and special characters. Switching to BPE tokeniser-based splitting at 450 tokens reduced failures from 116 to 1.

### Hybrid Retrieval

Academic queries contain domain-specific terms (QLoRA, NF4, RAGAS) where exact keyword matching outperforms semantic similarity. Combining BM25 sparse search with dense vector search via Reciprocal Rank Fusion captures both semantic meaning and keyword precision, improving Hit Rate from 67% to 73%.

### Cross-Encoder Reranking

A bi-encoder retriever scores query-document pairs independently. A cross-encoder jointly attends to both, producing much more accurate relevance scores at the cost of speed. By using the cross-encoder only on the top-40 candidates from hybrid search, we get high-quality reranking with minimal latency overhead (+1.3s for a transformative quality improvement from 73% → 100% Hit Rate).

### Few-Shot Prompt Engineering

Designed token-optimised few-shot examples (~350 tokens overhead) covering three RAG behaviours: context-grounded answering, multi-paper synthesis, and refusal. This approach improved keyword coverage by +1.6%p with only +0.8s latency — more effective and far cheaper than 6.8 hours of LoRA fine-tuning that produced a -28.4%p regression.

## Why Fine-Tuning Didn't Improve Metrics

### What I Tried

Generated 1,997 synthetic Q&A training examples across three categories designed to address specific base model weaknesses:

| Type | Count | Purpose |
|------|-------|---------|
| Grounded (60%) | 1,200 | Context-only answering with paper attribution |
| Synthesis (20%) | 397 | Multi-paper comparison in prose (no markdown) |
| Refusal (20%) | 400 | Proper refusal when context is insufficient |

Training configuration: LoRA (r=16, α=32) on all attention + MLP projections, 3 epochs with cosine LR schedule, bf16 on Apple M4 Pro. Best checkpoint at epoch 2 (val_loss: 1.060).

### What Happened

The fine-tuned model scored drastically lower on keyword coverage (-28.4%p) with 8x higher word counts and 2.4x higher latency. Inspection of responses revealed the dominant failure mode: **every response begins by repeating the system prompt instructions verbatim**, inflating word counts to ~1,600 and displacing actual answer content.

### Root Cause: Training Data Contamination

The synthetic data generation pipeline (Day 4) used Qwen3's `format: json` with thinking mode enabled. The model's `thinking` field contained system prompt fragments mixed with reasoning. When these were extracted as training answers, the model learned to reproduce instruction text as part of its response — a form of **training data contamination** where the model memorised prompt scaffolding rather than learning the intended answering behaviour.

Additional contributing factors include catastrophic forgetting in the 4B model, evaluation metric mismatch (keyword matching penalises concise answers), and quantisation differences between base (Q4_K_M) and fine-tuned (Q8_0) models.

### What I Would Do Differently

- **Validate training data for instruction leakage**: Add automated checks that reject any training answer containing system prompt fragments
- **Use a separate model for data generation**: Avoid the thinking mode contamination issue by generating data with a different model
- **Start with few-shot baseline before fine-tuning**: Establish prompt engineering ceiling first, then fine-tune only if there is a clear gap
- **Add semantic evaluation metrics** (BERTScore, GPT-as-judge) alongside keyword matching
- **Use a larger base model (7B+)** to reduce catastrophic forgetting risk

## Development Timeline

| Day | Focus | Key Outcomes |
|-----|-------|-------------|
| 1 | Infrastructure | 132 papers crawled, parsed, chunked, indexed. Caught embedding model failure via cosine similarity test. |
| 2 | RAG Pipeline | FastAPI + Streamlit serving. Evaluation pipeline with 15-question benchmark. Qwen3 thinking mode fix. |
| 3 | Retrieval Optimisation | Hit Rate 60% → 100%, MRR 0.51 → 0.82. Hybrid search + cross-encoder reranking. |
| 4 | Fine-Tuning Prep | 1,997 synthetic Q&A pairs generated. Code quality refactoring (9 fixes). |
| 5 | Fine-Tuning & Eval | LoRA training, GGUF conversion, Ollama deployment. Honest evaluation showing regression — analysed root causes. |
| 6 | Testing & CI/CD | 104 tests (unit + integration). GitHub Actions CI. Docker Compose full-stack deployment. Few-shot experiment revealing training data contamination as fine-tuning root cause. |

## Detailed Logs

For full experiment data and debugging notes:

- [Development Log](docs/Development_log.md)
- [Embedding Model Debugging](docs/embedding_model_debugging.md)
- [Retrieval Optimisation Experiments](docs/retrieval_optimisation.md)
- [Fine-Tuning Experiment Log](docs/finetuning_experiment.md)

## Known Limitations & Scaling Considerations

- **In-memory BM25**: All chunks loaded into memory. Sufficient for 132 papers (~5K chunks), 
  but would require ElasticSearch/OpenSearch for larger corpora.
- **Synchronous Ollama calls**: Embedding and generation use blocking `httpx.Client`. 
  Adequate for single-user demo; multi-user serving would need `httpx.AsyncClient` with async/await.
- **Ollama not containerised**: Runs on host for Apple Silicon Metal GPU access. 
  For cloud deployment, would need a GPU-enabled container or API-based LLM service.

## License

MIT