# Test Coverage

208 tests across 7 files. All tests run without external services — LLM, ChromaDB, and the pymupdf4llm C extension are mocked at the module boundary.

```
uv run pytest tests/ -v
```

---

## Test Suite Overview

| File | Tests | Type | Module under test |
|------|------:|------|-------------------|
| `test_chunker.py` | 33 | Unit | `src/api/core/chunker.py` |
| `test_llm_client.py` | 22 | Unit | `src/api/core/llm_client.py` |
| `test_rag_chain.py` | 18 | Unit | `src/api/core/rag_chain.py` |
| `test_hybrid_retriever.py` | 18 | Unit | `src/api/core/hybrid_retriever.py` |
| `test_api_integration.py` | 21 | Integration | `src/api/routers/` + `src/api/main.py` |
| `test_qa_dataset_generator.py` | 46 | Unit | `src/finetuning/generate_qa_dataset.py` |
| `test_pdf_parser.py` | 51 | Unit | `src/ingestion/pdf_parser.py` |
| **Total** | **208** | | |

---

## File-by-File Coverage

### `test_chunker.py` — 33 tests

| Class | Function | Tests |
|-------|----------|-------|
| `TestStripReferences` | `strip_references_from_text` | Bold-numbered, markdown heading, standalone bold, bibliography, acknowledgments headers; no-op when absent; earliest match wins |
| `TestIsCitationText` | `is_citation_text` | DOI-heavy, conference-heavy, normal academic text, empty string, mixed high-citation score |
| `TestCleanChunkText` | `clean_chunk_text` | Image markdown, link label preservation, markdown headers, unicode replacement char, table remnants, whitespace collapse, LaTeX preservation |
| `TestIsQualityChunk` | `is_quality_chunk` | Short text, sufficient length, low alpha ratio, citation-heavy, good academic text |
| `TestGenerateChunkId` | `generate_chunk_id` | Deterministic, different inputs → different IDs, length == 12 |
| `TestChunkText` | `chunk_text` | Short text → 1 chunk, long text → multiple chunks, overlap creates more chunks |
| `TestChunkPaper` | `chunk_paper` | References section excluded, full_text fallback, empty paper |

**Mock strategy**: `_get_tokenizer` patched to return a `MagicMock` with controllable `encode`/`decode` return values. No real tokeniser loaded.

---

### `test_llm_client.py` — 22 tests

| Class | Function | Tests |
|-------|----------|-------|
| `TestCleanResponse` | `_clean_response` | Complete `<think>` tags, unclosed tag, multiline think, no tags, multiple blocks, empty tags, whitespace stripping |
| `TestGenerate` | `generate` | `/no_think` in system only, default/custom temperature, `num_predict`, stream=False, empty system omitted, response cleaned, model name in payload, correct endpoint |
| `TestBuildPayload` | `_build_payload` | stream True/False, `/no_think` in system, no system when empty |
| `TestStreamGenerate` | `stream_generate` | Clean token streaming, `<think>` block filtered mid-stream |

**Mock strategy**: `httpx.AsyncClient` mocked; streaming tests use async generator mocks for `aiter_lines`.

---

### `test_rag_chain.py` — 18 tests

| Class | Function | Tests |
|-------|----------|-------|
| `TestPromptInjection` | `RAGChain.__init__` | Default prompts, custom system, custom query template, few-shot injection, None falls back to default |
| `TestFormatContext` | `_format_context` | Numbered format, section included, content included, empty chunks |
| `TestDeduplicateSources` | `_deduplicate_sources` | Different papers kept, same paper different sections kept, same paper same section deduped (first wins), empty input |
| `TestQuery` | `query` | Returns `RAGResponse`, system prompt passed to LLM, custom prompt, top_k forwarded, context in prompt |

**Mock strategy**: Both `Retriever` and `LLMClient` patched at `src.api.core.rag_chain` import site.

---

### `test_hybrid_retriever.py` — 18 tests

| Class | Tests |
|-------|-------|
| BM25 tokenization | Lowercase normalisation, stop-word-like tokens |
| RRF merging | Scores combined correctly, rank ordering |
| Deduplication | `arxiv_id::section` key, first occurrence wins |
| Reranker score conversion | Sigmoid normalisation, score → distance mapping |

**Mock strategy**: `chromadb`, `rank_bm25`, `sentence_transformers` mocked at `sys.modules` level.

---

### `test_api_integration.py` — 21 tests

| Endpoint | Tests |
|----------|-------|
| `POST /query` | 200 success with answer + sources, 422 validation error (empty query), 500 on service failure |
| `GET /health` | Healthy state, degraded state (Ollama/ChromaDB down) |
| `GET /` | Root info endpoint |

**Mock strategy**: `RAGChain.query` mocked as `AsyncMock`; ChromaDB and BM25 mocked at import level. Uses `fastapi.testclient.TestClient`.

---

### `test_qa_dataset_generator.py` — 46 tests

| Class | Function | Tests |
|-------|----------|-------|
| `TestParseJsonResponse` | `parse_json_response` | Valid JSON, JSON embedded in prose, empty string, None, malformed JSON, missing `question`, missing `answer`, question ≤10 chars rejected, answer ≤20 chars rejected, extra keys tolerated, nested braces, Unicode |
| `TestGenerateType1` | `generate_type1` | Schema keys, instruction == RAG_SYSTEM_PROMPT, title in input, question in input, arxiv_id, None on LLM failure, None on invalid JSON, text truncated to 500 chars |
| `TestGenerateType2` | `generate_type2` | Schema keys, both arxiv IDs in source_arxiv_id, both titles in input, instruction, None on LLM failure, text truncated to 400 chars per paper |
| `TestGenerateType3` | `generate_type3` | Schema keys, instruction, None on LLM failure, 7 topic_map keyword cases (rag, qlora, lora, hallucin, prompt, instruct, default), refusal phrase in output |
| `TestLoadChunks` | `load_chunks` | Returns tuple, grouped by arxiv_id, paper entry structure, empty file |
| `TestDatasetSampleContract` | all three generators | All 5 required keys present, type values in `{grounded, synthesis, refusal}` |

**Mock strategy**: `call_llm` patched with `side_effect` returning fixture JSON strings. `CHUNKS_FILE` patched to `tmp_path` fixtures.

**Bug caught**: `generate_type3` topic_map had `lora` before `qlora`; a paper titled "QLoRA: ..." would match `lora` first and receive topic `"parameter-efficient fine-tuning"` instead of `"quantised fine-tuning"`. Fixed by reordering the dict; the parametrised test prevents regression.

---

### `test_pdf_parser.py` — 51 tests

| Class | Function | Tests |
|-------|----------|-------|
| `TestCleanText` | `clean_text` | Image placeholder removal (single, multiple, empty alt), triple/many newlines collapsed, double newline preserved, multiple spaces collapsed, leading/trailing whitespace stripped, empty string, normal text unchanged |
| `TestSplitSectionsBoldNumbered` | `split_sections` | Introduction, Method, Experiments, Conclusion detected; references excluded from content; section content assigned correctly; `**1.**` variant (period after number) |
| `TestSplitSectionsMarkdown` | `split_sections` | `## Introduction`, `## 1. Introduction`, Related Work, Background→related_work, `###`, Evaluation→experiments, Discussion, Ablation→discussion, Summary→conclusion, multiple sections |
| `TestSplitSectionsStandaloneBold` | `split_sections` | `**Abstract**` detected; `**References**`-only → full_text fallback (references is excluded); `**Acknowledgments**` standalone → full_text (not in pattern list — documented limitation) |
| `TestSplitSectionsPlainText` | `split_sections` | `Abstract\n` detected; plain `References\n` → full_text (not supported — documented limitation) |
| `TestSplitSectionsEdgeCases` | `split_sections` | No headers → full_text, empty string → full_text, references-only → full_text, 50-char minimum guard, duplicate headers → first wins, mixed formats, section order preserved, Approach/Framework→method, Results→experiments, Analysis→discussion, bibliography excluded, Acknowledgements (British) → full_text (documented limitation), no-abstract paper |
| `TestExtractTextFromPdf` | `extract_text_from_pdf` | Stripped markdown on success, None on RuntimeError, None on FileNotFoundError, path forwarded to pymupdf4llm, blank PDF → empty string |

**Mock strategy**: `sys.modules["pymupdf4llm"] = MagicMock()` at module import level. Individual tests use `patch("src.ingestion.pdf_parser.pymupdf4llm.to_markdown")`.

---

## Known Gaps

The following areas are not covered by the current test suite:

| Area | Gap | Reason |
|------|-----|--------|
| `src/ingestion/arxiv_crawler.py` | No unit tests | Requires live arXiv API or complex HTTP mocking; excluded from CI scope |
| `src/ingestion/indexer.py` | No unit tests | Tightly coupled to ChromaDB batch operations; covered indirectly by integration tests |
| `src/evaluation/evaluate.py` | No unit tests | Evaluation logic is orchestration-heavy; results validated manually during experiments |
| `src/finetuning/generate_qa_dataset.py` — `call_llm` retry logic | Partially covered | Retry backoff (`time.sleep`) not directly asserted; only final return value tested |
| `src/finetuning/generate_qa_dataset.py` — `main()` | Not covered | Orchestration entry point; would require full filesystem + LLM mocking |
| `pdf_parser.py` — `parse_all_papers()` | Not covered | Filesystem + metadata JSON orchestration; covered by manual smoke tests |
| `split_sections` — `**Acknowledgments**` standalone | Documented as unsupported | Pattern not in `section_markers`; see `test_standalone_acknowledgments_not_supported` |
| `split_sections` — plain `References\n` | Documented as unsupported | No plain-text pattern for references; see `test_plain_references_not_supported` |

---

## Running Tests

```bash
# All tests
uv run pytest tests/ -v

# Specific file
uv run pytest tests/test_qa_dataset_generator.py -v
uv run pytest tests/test_pdf_parser.py -v

# By marker (unit only)
uv run pytest tests/ -v -k "not integration"

# With coverage report
uv run pytest tests/ --cov=src --cov-report=term-missing
```

Tests require no running services. All external dependencies (Ollama, ChromaDB, pymupdf4llm) are mocked.
