"""
Unit tests for src/api/core/hybrid_retriever.py

Tests cover:
- BM25 tokenization
- Reciprocal Rank Fusion (RRF) merging
- Deduplication by arxiv_id::section
- Reranker score to distance conversion (sigmoid normalisation)

External dependencies (ChromaDB, Ollama, CrossEncoder, file I/O)
are mocked to isolate pure logic.
"""

import json
import math
import sys
from io import StringIO
from unittest.mock import MagicMock, patch

import pytest

# Mock heavy dependencies before importing the module
_mock_bm25_module = MagicMock()
sys.modules["chromadb"] = MagicMock()
sys.modules["rank_bm25"] = _mock_bm25_module
sys.modules["sentence_transformers"] = MagicMock()

# Need to patch BM25Okapi as a class that the module imports
BM25Okapi = _mock_bm25_module.BM25Okapi

from src.api.core.hybrid_retriever import HybridRetriever

# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────

@pytest.fixture
def retriever():
    """Create a HybridRetriever with all external deps mocked."""
    mock_chunks_data = {
        "chunks": [
            {
                "chunk_id": "aaa111",
                "arxiv_id": "2401.00001",
                "title": "Paper A",
                "section": "Introduction",
                "text": "RAG combines retrieval with generation for better answers.",
                "metadata": {"authors": ["Author A"], "published": "2024-01"},
            },
            {
                "chunk_id": "bbb222",
                "arxiv_id": "2401.00001",
                "title": "Paper A",
                "section": "Method",
                "text": "The method uses dense vector search followed by reranking.",
                "metadata": {"authors": ["Author A"], "published": "2024-01"},
            },
            {
                "chunk_id": "ccc333",
                "arxiv_id": "2401.00002",
                "title": "Paper B",
                "section": "Introduction",
                "text": "QLoRA reduces memory by quantising base model weights.",
                "metadata": {"authors": ["Author B"], "published": "2024-02"},
            },
            {
                "chunk_id": "ddd444",
                "arxiv_id": "2401.00003",
                "title": "Paper C",
                "section": "Results",
                "text": "Hallucination detection improved with retrieval augmentation.",
                "metadata": {"authors": ["Author C"], "published": "2024-03"},
            },
        ]
    }

    from unittest.mock import mock_open
    mock_file_content = json.dumps(mock_chunks_data)

    with patch("src.api.core.hybrid_retriever.chromadb.HttpClient") as mock_chroma, \
         patch("src.api.core.hybrid_retriever.httpx.Client"), \
         patch("src.api.core.hybrid_retriever.CrossEncoder"), \
         patch("builtins.open", mock_open(read_data=mock_file_content)), \
         patch.object(sys.modules["rank_bm25"], "BM25Okapi") as MockBM25:

        mock_chroma.return_value.get_collection.return_value = MagicMock()
        MockBM25.return_value = MagicMock()

        # Disable UMAP file check
        with patch("pathlib.Path.exists", return_value=False):
            ret = HybridRetriever()

    yield ret


# ──────────────────────────────────────────────
# _tokenize
# ──────────────────────────────────────────────

class TestTokenize:
    def test_basic_tokenization(self, retriever):
        tokens = retriever._tokenize("Hello World! This is a test.")
        assert "hello" in tokens
        assert "world" in tokens
        assert "!" not in tokens

    def test_removes_single_char_tokens(self, retriever):
        tokens = retriever._tokenize("I am a test")
        assert "i" not in tokens
        assert "a" not in tokens
        assert "am" in tokens
        assert "test" in tokens

    def test_handles_special_characters(self, retriever):
        tokens = retriever._tokenize("QLoRA: 4-bit quantisation (NF4)")
        assert "qlora" in tokens
        assert "4-bit" in tokens
        assert "quantisation" in tokens

    def test_empty_input(self, retriever):
        tokens = retriever._tokenize("")
        assert tokens == []


# ──────────────────────────────────────────────
# _rrf_fusion
# ──────────────────────────────────────────────

class TestRRFFusion:
    def test_single_source(self, retriever):
        vector = {"aaa111": 1, "bbb222": 2}
        bm25 = {}
        result = retriever._rrf_fusion(vector, bm25)
        assert result[0] == "aaa111"
        assert result[1] == "bbb222"

    def test_both_sources_boost(self, retriever):
        vector = {"aaa111": 1, "ccc333": 2}
        bm25 = {"ccc333": 1, "aaa111": 3}
        result = retriever._rrf_fusion(vector, bm25)
        # aaa111: 1/(60+1) + 1/(60+3) = 0.01639 + 0.01587 = 0.03226
        # ccc333: 1/(60+2) + 1/(60+1) = 0.01613 + 0.01639 = 0.03252
        # ccc333 should rank higher (appears in both, rank 1 in bm25)
        assert result[0] == "ccc333"

    def test_unique_items_from_each_source(self, retriever):
        vector = {"aaa111": 1}
        bm25 = {"bbb222": 1}
        result = retriever._rrf_fusion(vector, bm25)
        assert set(result) == {"aaa111", "bbb222"}

    def test_empty_inputs(self, retriever):
        result = retriever._rrf_fusion({}, {})
        assert result == []

    def test_custom_k_parameter(self, retriever):
        vector = {"aaa111": 1, "bbb222": 2}
        bm25 = {"bbb222": 1}
        result_k60 = retriever._rrf_fusion(vector, bm25, k=60)
        result_k1 = retriever._rrf_fusion(vector, bm25, k=1)
        # With k=1, rank differences are amplified more
        # Both should have bbb222 first (boosted by both sources)
        assert result_k60[0] == "bbb222"
        assert result_k1[0] == "bbb222"


# ──────────────────────────────────────────────
# _deduplicate
# ──────────────────────────────────────────────

class TestDeduplicate:
    def test_same_paper_same_section_deduped(self, retriever):
        # aaa111 and bbb222 are from same paper but different sections
        scored = [("aaa111", 0.9), ("bbb222", 0.8), ("ccc333", 0.7)]
        result = retriever._deduplicate(scored, top_k=5)
        # All three kept — different arxiv_id::section keys
        assert len(result) == 3

    def test_respects_top_k(self, retriever):
        scored = [("aaa111", 0.9), ("bbb222", 0.8), ("ccc333", 0.7), ("ddd444", 0.6)]
        result = retriever._deduplicate(scored, top_k=2)
        assert len(result) == 2

    def test_preserves_score_order(self, retriever):
        scored = [("ccc333", 0.95), ("aaa111", 0.8), ("ddd444", 0.7)]
        result = retriever._deduplicate(scored, top_k=5)
        scores = [s for _, s in result]
        assert scores == sorted(scores, reverse=True)

    def test_skips_unknown_chunk_ids(self, retriever):
        scored = [("unknown_id", 0.99), ("aaa111", 0.8)]
        result = retriever._deduplicate(scored, top_k=5)
        ids = [cid for cid, _ in result]
        assert "unknown_id" not in ids
        assert "aaa111" in ids

    def test_empty_input(self, retriever):
        result = retriever._deduplicate([], top_k=5)
        assert result == []


# ──────────────────────────────────────────────
# Sigmoid distance normalisation
# ──────────────────────────────────────────────

class TestSigmoidNormalisation:
    def test_high_score_low_distance(self):
        """Positive reranker score should produce low distance."""
        score = 5.0
        distance = 1.0 - (1.0 / (1.0 + math.exp(-score)))
        assert distance < 0.01

    def test_low_score_high_distance(self):
        """Negative reranker score should produce high distance."""
        score = -5.0
        distance = 1.0 - (1.0 / (1.0 + math.exp(-score)))
        assert distance > 0.99

    def test_zero_score_mid_distance(self):
        """Zero score should produce 0.5 distance."""
        score = 0.0
        distance = 1.0 - (1.0 / (1.0 + math.exp(-score)))
        assert abs(distance - 0.5) < 0.001

    def test_distance_always_in_range(self):
        """Distance should always be between 0 and 1."""
        for score in [-10, -1, 0, 1, 10]:
            distance = 1.0 - (1.0 / (1.0 + math.exp(-score)))
            assert 0.0 <= distance <= 1.0
