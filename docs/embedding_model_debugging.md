# Embedding Model Selection: Debugging & Decision Log

## Context

During the development of the arXiv RAG System, I encountered a critical failure in the embedding pipeline that fundamentally broke retrieval quality. This document captures the debugging process, data, and lessons learned.

## Problem Statement

After indexing ~4000 chunks from 132 arXiv papers into ChromaDB, the test query `"What is Retrieval Augmented Generation?"` consistently returned **completely irrelevant results** — Software Engineering papers, Requirements Engineering papers, and Gaussian Process papers instead of the dozens of RAG-specific papers in the corpus.

## Initial Setup

- **Embedding Model**: `nomic-embed-text` (137M params, F16) via Ollama
- **Vector DB**: ChromaDB with cosine distance
- **Chunk Size**: 256 words
- **Corpus**: 132 papers, ~4000 chunks after filtering

## Debugging Process

### Step 1: Check retrieval distances

The top results all had distances around 0.34–0.40, which is suspiciously high for cosine distance when the query terms literally appear in many documents.

```
[1] dist=0.3472 | A Study about the Knowledge and Use of Requirements Engineering...
[2] dist=0.4033 | Software Engineering for Collective Cyber-Physical...
[3] dist=0.4133 | Morescient GAI for Software Engineering...
```

### Step 2: Verify chunks exist

Confirmed that RAG-relevant chunks were present in the index:

```
Found RAG chunk: "Ragas: Automated Evaluation of Retrieval Augmented Generation"
Preview: We introduce Ragas (Retrieval Augmented Generation Assessment), a framework for...
```

### Step 3: Sanity Check — Direct cosine similarity test

This was the critical test. I embedded a query, a relevant document, and an irrelevant document, then computed cosine similarity directly.

#### Test with `nomic-embed-text`:

| Pair | Cosine Similarity |
|------|-------------------|
| Query ↔ RAG paper chunk | **0.4081** |
| Query ↔ Irrelevant chunk | **0.5963** |

**The irrelevant document scored HIGHER than the relevant one.** This means the embedding model was producing a fundamentally broken vector space for this use case.

#### Attempted fix: nomic-embed-text with prefixes

`nomic-embed-text` documentation recommends using `search_query:` and `search_document:` prefixes.

| Pair | Cosine Similarity |
|------|-------------------|
| Query ↔ RAG paper chunk | **0.5358** |
| Query ↔ Irrelevant chunk | **0.6914** |

Prefixes improved absolute scores but the **ranking was still inverted** — irrelevant content still scored higher.

### Step 4: Benchmark alternative model

Tested `mxbai-embed-large` (335M params) as a replacement.

#### Test with `mxbai-embed-large`:

| Pair | Cosine Similarity |
|------|-------------------|
| Query ↔ RAG paper chunk | **0.7649** |
| Query ↔ Irrelevant chunk | **0.4850** |

**Correct ranking restored.** Relevant content now scores significantly higher (0.76 vs 0.49).

### Step 5: Re-index and validate

After switching to `mxbai-embed-large` (with chunk size reduced to 128 words to fit the model's context window), the retrieval results were correct:

```
[1] dist=0.1668 | RAGPart & RAGMask: Retrieval-Stage Defenses Against...
[2] dist=0.1724 | RAG-Gym: Systematic Optimization of Language Agents...
[3] dist=0.1840 | Engineering the RAG Stack: A Comprehensive Review...
[4] dist=0.1853 | MultiHop-RAG: Benchmarking Retrieval-Augmented Gen...
[5] dist=0.1909 | T-RAG: Lessons from the LLM Trenches
```

All top-5 results are RAG-related papers with distances of 0.17–0.19.

## Summary of Results

| Model | RAG Similarity | Irrelevant Similarity | Correct Ranking? | Retrieval Quality |
|-------|---------------|----------------------|-------------------|-------------------|
| nomic-embed-text | 0.41 | 0.60 | No | Broken |
| nomic-embed-text + prefix | 0.54 | 0.69 | No | Broken |
| **mxbai-embed-large** | **0.76** | **0.49** | **Yes** | **Excellent** |

## Root Cause Analysis

The likely cause is that `nomic-embed-text` running via Ollama in GGUF/quantised format does not properly handle the task-specific prefixes it was trained with. The Hugging Face version of nomic-embed may work correctly, but the Ollama-served GGUF variant produces degraded embeddings for retrieval tasks.

## Key Takeaways

1. **Never trust a model blindly.** Always run a sanity check (relevant vs irrelevant similarity comparison) before building on top of an embedding model.
2. **Unit testing applies to ML pipelines.** A simple 3-line cosine similarity test caught a critical failure that would have made the entire RAG system useless.
3. **Quantisation can break task-specific behaviour.** Models optimised for specific tasks (like retrieval with prefixes) may lose that capability when quantised to GGUF format.
4. **Trade-offs are real.** `mxbai-embed-large` has a shorter context window (512 tokens vs 8192), requiring smaller chunks. But correct retrieval with smaller chunks far outperforms broken retrieval with larger ones.

## Interview-Ready Summary

> "During development, I initially deployed `nomic-embed-text` as the embedding model. However, a cosine similarity sanity check revealed a critical failure: irrelevant documents (similarity 0.60) scored higher than relevant ones (0.41), producing a completely inverted vector space.
>
> Rather than trusting the model blindly, I benchmarked `mxbai-embed-large` as an alternative. It produced correct separation (relevant: 0.76, irrelevant: 0.49), and after re-indexing with adjusted chunk sizes, all top-5 retrieval results were accurately matched to the query domain.
>
> This experience reinforced that unit testing is essential for ML pipelines — a simple 3-line similarity test caught a failure that would have rendered the entire system useless."
