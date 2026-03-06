"""
Unit tests for src/api/core/rag_chain.py

Tests cover:
- Prompt dependency injection (default and custom prompts)
- Context formatting from retrieved chunks
- Source deduplication by arxiv_id::section
- Full query pipeline (with mocked retriever and LLM)
"""

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Mock heavy dependencies before importing modules that need them
sys.modules["chromadb"] = MagicMock()
sys.modules["rank_bm25"] = MagicMock()
sys.modules["sentence_transformers"] = MagicMock()

from src.api.core.hybrid_retriever import RetrievedChunk
from src.api.core.prompts import (
    QUERY_TEMPLATE_DEFAULT,
    SYSTEM_PROMPT_FEW_SHOT,
    SYSTEM_PROMPT_ZERO_SHOT,
)
from src.api.core.rag_chain import RAGChain, RAGResponse

# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────

def _make_chunk(arxiv_id="2401.00001", title="Paper A", section="Introduction",
                text="Sample text", distance=0.15, authors="Author A", published="2024-01"):
    return RetrievedChunk(
        chunk_id="test_chunk",
        text=text,
        arxiv_id=arxiv_id,
        title=title,
        section=section,
        authors=authors,
        published=published,
        distance=distance,
    )


@pytest.fixture
def mock_rag_chain():
    """Create a RAGChain with mocked retriever and LLM."""
    with patch("src.api.core.rag_chain.Retriever") as MockRetriever, \
         patch("src.api.core.rag_chain.LLMClient") as MockLLM:

        mock_retriever = MagicMock()
        mock_llm = MagicMock()
        MockRetriever.return_value = mock_retriever
        MockLLM.return_value = mock_llm

        chain = RAGChain()
        chain._mock_retriever = mock_retriever
        chain._mock_llm = mock_llm
        yield chain


@pytest.fixture
def sample_chunks():
    return [
        _make_chunk(arxiv_id="2401.00001", title="Paper A", section="Introduction",
                     text="RAG combines retrieval with generation."),
        _make_chunk(arxiv_id="2401.00002", title="Paper B", section="Method",
                     text="QLoRA uses 4-bit quantisation."),
        _make_chunk(arxiv_id="2401.00001", title="Paper A", section="Results",
                     text="The system achieved 100% hit rate."),
    ]


# ──────────────────────────────────────────────
# Prompt Dependency Injection
# ──────────────────────────────────────────────

class TestPromptInjection:
    def test_default_prompts(self, mock_rag_chain):
        assert mock_rag_chain.system_prompt == SYSTEM_PROMPT_ZERO_SHOT
        assert mock_rag_chain.query_template == QUERY_TEMPLATE_DEFAULT

    def test_custom_system_prompt(self):
        with patch("src.api.core.rag_chain.Retriever"), \
             patch("src.api.core.rag_chain.LLMClient"):
            chain = RAGChain(system_prompt="Custom prompt")
            assert chain.system_prompt == "Custom prompt"
            assert chain.query_template == QUERY_TEMPLATE_DEFAULT

    def test_custom_query_template(self):
        with patch("src.api.core.rag_chain.Retriever"), \
             patch("src.api.core.rag_chain.LLMClient"):
            chain = RAGChain(query_template="Q: {question}\nC: {context}")
            assert chain.query_template == "Q: {question}\nC: {context}"
            assert chain.system_prompt == SYSTEM_PROMPT_ZERO_SHOT

    def test_fewshot_prompt_injection(self):
        with patch("src.api.core.rag_chain.Retriever"), \
             patch("src.api.core.rag_chain.LLMClient"):
            chain = RAGChain(system_prompt=SYSTEM_PROMPT_FEW_SHOT)
            assert "Example 1" in chain.system_prompt
            assert "Example 2" in chain.system_prompt
            assert "Example 3" in chain.system_prompt

    def test_none_falls_back_to_default(self):
        with patch("src.api.core.rag_chain.Retriever"), \
             patch("src.api.core.rag_chain.LLMClient"):
            chain = RAGChain(system_prompt=None, query_template=None)
            assert chain.system_prompt == SYSTEM_PROMPT_ZERO_SHOT
            assert chain.query_template == QUERY_TEMPLATE_DEFAULT


# ──────────────────────────────────────────────
# _format_context
# ──────────────────────────────────────────────

class TestFormatContext:
    def test_numbered_format(self, mock_rag_chain, sample_chunks):
        context = mock_rag_chain._format_context(sample_chunks)
        assert "[1] Paper: Paper A" in context
        assert "[2] Paper: Paper B" in context
        assert "[3] Paper: Paper A" in context

    def test_includes_section(self, mock_rag_chain, sample_chunks):
        context = mock_rag_chain._format_context(sample_chunks)
        assert "Section: Introduction" in context
        assert "Section: Method" in context
        assert "Section: Results" in context

    def test_includes_content(self, mock_rag_chain, sample_chunks):
        context = mock_rag_chain._format_context(sample_chunks)
        assert "RAG combines retrieval with generation." in context
        assert "QLoRA uses 4-bit quantisation." in context

    def test_empty_chunks(self, mock_rag_chain):
        context = mock_rag_chain._format_context([])
        assert context == ""


# ──────────────────────────────────────────────
# _deduplicate_sources
# ──────────────────────────────────────────────

class TestDeduplicateSources:
    def test_different_papers_kept(self, mock_rag_chain, sample_chunks):
        sources = mock_rag_chain._deduplicate_sources(sample_chunks)
        arxiv_ids = [s.arxiv_id for s in sources]
        assert "2401.00001" in arxiv_ids
        assert "2401.00002" in arxiv_ids

    def test_same_paper_different_sections_kept(self, mock_rag_chain, sample_chunks):
        sources = mock_rag_chain._deduplicate_sources(sample_chunks)
        # Paper A has Introduction and Results — both should be kept
        paper_a_sections = [s.section for s in sources if s.arxiv_id == "2401.00001"]
        assert len(paper_a_sections) == 2
        assert "Introduction" in paper_a_sections
        assert "Results" in paper_a_sections

    def test_same_paper_same_section_deduped(self, mock_rag_chain):
        chunks = [
            _make_chunk(arxiv_id="2401.00001", section="Introduction", distance=0.1),
            _make_chunk(arxiv_id="2401.00001", section="Introduction", distance=0.2),
        ]
        sources = mock_rag_chain._deduplicate_sources(chunks)
        assert len(sources) == 1
        assert sources[0].distance == 0.1  # First one wins

    def test_empty_input(self, mock_rag_chain):
        sources = mock_rag_chain._deduplicate_sources([])
        assert sources == []


# ──────────────────────────────────────────────
# query (full pipeline)
# ──────────────────────────────────────────────

class TestQuery:
    @pytest.mark.asyncio
    async def test_returns_rag_response(self, mock_rag_chain, sample_chunks):
        mock_rag_chain._mock_retriever.search = AsyncMock(return_value=(sample_chunks, None))
        mock_rag_chain._mock_llm.generate = AsyncMock(return_value="RAG is a technique.")

        response = await mock_rag_chain.query("What is RAG?")
        assert isinstance(response, RAGResponse)
        assert response.answer == "RAG is a technique."
        assert response.query == "What is RAG?"
        assert len(response.sources) > 0
        # Verify latency breakdown is populated
        assert response.latency.retrieval_ms >= 0
        assert response.latency.generation_ms >= 0
        assert response.latency.total_ms >= 0

    @pytest.mark.asyncio
    async def test_passes_system_prompt_to_llm(self, mock_rag_chain, sample_chunks):
        mock_rag_chain._mock_retriever.search = AsyncMock(return_value=(sample_chunks, None))
        mock_rag_chain._mock_llm.generate = AsyncMock(return_value="Answer")

        await mock_rag_chain.query("test question")
        call_kwargs = mock_rag_chain._mock_llm.generate.call_args
        assert call_kwargs.kwargs["system"] == SYSTEM_PROMPT_ZERO_SHOT

    @pytest.mark.asyncio
    async def test_custom_prompt_passed_to_llm(self, sample_chunks):
        with patch("src.api.core.rag_chain.Retriever") as MockRet, \
             patch("src.api.core.rag_chain.LLMClient") as MockLLM:

            mock_ret = MagicMock()
            mock_llm = MagicMock()
            MockRet.return_value = mock_ret
            MockLLM.return_value = mock_llm

            mock_ret.search = AsyncMock(return_value=(sample_chunks, None))
            mock_llm.generate = AsyncMock(return_value="Answer")

            chain = RAGChain(system_prompt="Custom system prompt")
            await chain.query("test")

            call_kwargs = mock_llm.generate.call_args
            assert call_kwargs.kwargs["system"] == "Custom system prompt"

    @pytest.mark.asyncio
    async def test_top_k_forwarded(self, mock_rag_chain):
        mock_rag_chain._mock_retriever.search = AsyncMock(return_value=([], None))
        mock_rag_chain._mock_llm.generate = AsyncMock(return_value="")

        await mock_rag_chain.query("test", top_k=10)
        mock_rag_chain._mock_retriever.search.assert_called_with("test", top_k=10, get_embeddings=False)

    @pytest.mark.asyncio
    async def test_context_included_in_prompt(self, mock_rag_chain, sample_chunks):
        mock_rag_chain._mock_retriever.search = AsyncMock(return_value=(sample_chunks, None))
        mock_rag_chain._mock_llm.generate = AsyncMock(return_value="Answer")

        await mock_rag_chain.query("What is RAG?")
        call_args = mock_rag_chain._mock_llm.generate.call_args
        prompt = call_args.kwargs["prompt"]
        assert "Paper A" in prompt
        assert "Paper B" in prompt
        assert "What is RAG?" in prompt
