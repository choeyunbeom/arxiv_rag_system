# arXiv RAG System

A Retrieval-Augmented Generation system for querying academic papers from arXiv. Built as a portfolio project demonstrating end-to-end ML engineering: data pipeline, hybrid retrieval, LLM fine-tuning, and systematic evaluation.

Ask a question in natural language → the system retrieves relevant papers → generates a cited, grounded answer.

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

### LoRA Fine-Tuning

Fine-tuned Qwen3 4B on 1,997 synthetic Q&A pairs targeting three RAG-specific behaviours: context-grounded answering, multi-paper synthesis, and proper refusal handling.

| Metric | Base Model | Fine-Tuned | Change |
|--------|-----------|------------|--------|
| Keyword Coverage | 80.2% | 71.1% | -9.1%p |
| Substantive Rate | 100% | 93.3% | -6.7%p |
| Source Hit Rate | 100% | 100% | — |
| Avg Latency | 20.3s | 22.0s | +1.7s |

**The fine-tuned model underperformed the base model.** This is a genuine result that I analyse in detail below — understanding failure modes is as important as achieving improvements. See [Fine-Tuning Analysis](#why-fine-tuning-didnt-improve-metrics) for the full breakdown.

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
| Config | Pydantic Settings |

## Project Structure

```
arxiv_rag_system/
├── src/
│   ├── api/
│   │   ├── core/
│   │   │   ├── config.py            # Centralised Pydantic settings
│   │   │   ├── hybrid_retriever.py  # Dense + BM25 + reranker pipeline
│   │   │   ├── llm_client.py        # Ollama API wrapper
│   │   │   ├── rag_chain.py         # Retrieval → Generation orchestrator
│   │   │   └── chunker.py           # Token-aware chunking with quality filters
│   │   ├── models/schemas.py        # Request/response Pydantic models
│   │   ├── routers/                 # FastAPI route handlers
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
│       └── finetune_lora.ipynb     # LoRA training notebook
├── ui/app.py                        # Streamlit frontend
├── data/
│   ├── raw/                         # 132 arXiv PDFs
│   ├── processed/                   # Chunks, metadata, eval results
│   ├── base_model/                  # Qwen3-4B weights (git-ignored)
│   └── finetuned_lora/              # LoRA adapter + metrics (git-ignored)
└── docs/                            # Detailed experiment logs
```

## Setup

### Prerequisites

- Python 3.11+
- Docker Desktop (for ChromaDB)
- Ollama (for LLM + embeddings)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/arxiv_rag_system.git
cd arxiv_rag_system

# Create virtual environment
uv venv && source .venv/bin/activate

# Install dependencies
uv pip install -e .

# Start ChromaDB
docker compose up -d

# Pull models
ollama pull qwen3:4b
ollama pull mxbai-embed-large
```

### Running

```bash
# Start the API server
uvicorn src.api.main:app --reload

# Start the UI (in another terminal)
streamlit run ui/app.py
```

The API is available at `http://localhost:8000/docs` and the UI at `http://localhost:8501`.

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

The fine-tuned model scored lower on keyword coverage (-9.1%p) and substantive rate (-6.7%p) compared to the base model. Per-question analysis revealed:

- **1 empty response** (Ragas evaluation question): The fine-tuned model likely exhausted output tokens on internal reasoning (`<think>` tags), returning a 3-word truncated answer. The base model's `/no_think` suppression worked more reliably.
- **Lower keyword coverage across the board**: The model learned to be more concise (as intended), but the keyword-matching evaluation metric penalises shorter answers that omit synonyms or related terms present in the expected keyword lists.

### Root Cause Analysis

1. **Catastrophic forgetting in small models**: At 4B parameters, LoRA fine-tuning on 2,000 examples is enough to shift response style but also degrades the model's ability to comprehensively cover a topic. Larger models (7B+) are more resilient to this tradeoff.

2. **Evaluation metric mismatch**: Keyword Coverage measures whether specific terms appear in the answer. The fine-tuned model was trained to produce concise, prose-style answers — exactly the behaviour that reduces keyword recall. A semantic similarity metric (e.g., BERTScore) would better capture answer quality improvements.

3. **Quantisation gap**: The base model runs as Ollama's default `qwen3:4b` (Q4_K_M quantisation), while the fine-tuned model was converted to Q8_0 GGUF. Different quantisation methods can affect generation behaviour independently of the fine-tuning itself.

4. **Thinking mode interaction**: Qwen3's `<think>` reasoning mode behaves differently after fine-tuning. The base model's `/no_think` prompt injection was calibrated for the original model weights and may be less effective after LoRA modification.

### What I Would Do Differently

- **Use a larger base model (7B+)** to reduce catastrophic forgetting risk
- **Add semantic evaluation metrics** (BERTScore, GPT-as-judge) alongside keyword matching
- **Train with thinking mode explicitly disabled** by including `/no_think` tokens in training data
- **Use fewer epochs (1-2) with lower learning rate** to minimise forgetting while still imparting style changes
- **A/B test with human evaluation** to capture qualitative improvements that automated metrics miss

## Development Timeline

| Day | Focus | Key Outcomes |
|-----|-------|-------------|
| 1 | Infrastructure | 132 papers crawled, parsed, chunked, indexed. Caught embedding model failure via cosine similarity test. |
| 2 | RAG Pipeline | FastAPI + Streamlit serving. Evaluation pipeline with 15-question benchmark. Qwen3 thinking mode fix. |
| 3 | Retrieval Optimisation | Hit Rate 60% → 100%, MRR 0.51 → 0.82. Hybrid search + cross-encoder reranking. |
| 4 | Fine-Tuning Prep | 1,997 synthetic Q&A pairs generated. Code quality refactoring (9 fixes). |
| 5 | Fine-Tuning & Eval | LoRA training, GGUF conversion, Ollama deployment. Honest evaluation showing regression — analysed root causes. |

## Detailed Logs

For full experiment data and debugging notes:

- [Weekly Development Log](docs/weekly_log.md)
- [Embedding Model Debugging](docs/embedding_model_debugging.md)
- [Retrieval Optimisation Experiments](docs/retrieval_optimisation.md)
- [Fine-Tuning Experiment Log](docs/finetuning_experiment.md)

## Known Limitations & Scaling Considerations

- **In-memory BM25**: All chunks loaded into memory. Sufficient for 132 papers (~5K chunks), 
  but would require ElasticSearch/OpenSearch for larger corpora.
- **Synchronous Ollama calls**: Embedding and generation use blocking `httpx.Client`. 
  Adequate for single-user demo; multi-user serving would need `httpx.AsyncClient` with async/await.

## License

MIT
