"""
Unit tests for src/api/core/chunker.py

Tests cover:
- Reference stripping from full text
- Citation content detection
- Text cleaning for embedding quality
- Quality filtering for meaningful chunks
- Chunk ID generation
- Token-based text splitting
"""

from unittest.mock import MagicMock, patch

from src.api.core.chunker import (
    chunk_paper,
    chunk_text,
    clean_chunk_text,
    generate_chunk_id,
    is_citation_text,
    is_quality_chunk,
    strip_references_from_text,
)

# ──────────────────────────────────────────────
# strip_references_from_text
# ──────────────────────────────────────────────

class TestStripReferences:
    def test_bold_numbered_references(self):
        text = "Some content here.\n**5** **References**\n[1] Author et al."
        result = strip_references_from_text(text)
        assert "Author et al." not in result
        assert "Some content here." in result

    def test_markdown_heading_references(self):
        text = "Content above.\n## References\n[1] Paper title."
        result = strip_references_from_text(text)
        assert "Paper title." not in result
        assert "Content above." in result

    def test_standalone_bold_references(self):
        text = "Main body text.\n**References**\nBibliography entries."
        result = strip_references_from_text(text)
        assert "Bibliography entries." not in result

    def test_bibliography_heading(self):
        text = "Results section.\n## Bibliography\nEntries here."
        result = strip_references_from_text(text)
        assert "Entries here." not in result

    def test_acknowledgments_removed(self):
        text = "Discussion here.\n**Acknowledgments**\nThanks to reviewers."
        result = strip_references_from_text(text)
        assert "Thanks to reviewers." not in result

    def test_no_references_section(self):
        text = "This paper has no references section at the end."
        result = strip_references_from_text(text)
        assert result == text

    def test_earliest_match_wins(self):
        text = "Body.\n**Acknowledgments**\nThanks.\n**References**\nRefs."
        result = strip_references_from_text(text)
        assert "Thanks." not in result
        assert "Refs." not in result
        assert "Body." in result


# ──────────────────────────────────────────────
# is_citation_text
# ──────────────────────────────────────────────

class TestIsCitationText:
    def test_doi_heavy_text(self):
        text = "See https://doi.org/10.1234 and https://doi.org/10.5678 for details."
        assert is_citation_text(text) is True

    def test_conference_heavy_text(self):
        text = "In Proceedings of ACL 2023. In Proceedings of EMNLP 2022."
        assert is_citation_text(text) is True

    def test_normal_academic_text(self):
        text = (
            "Retrieval-Augmented Generation combines a retriever and a generator. "
            "The retriever fetches relevant documents from a corpus, and the generator "
            "produces an answer conditioned on those documents."
        )
        assert is_citation_text(text) is False

    def test_empty_text(self):
        assert is_citation_text("") is True

    def test_mixed_but_high_citation_score(self):
        text = (
            "Smith, 2020. Jones, 2021. pp. 12-15. "
            "Brown, 2019. In _NeurIPS_. https://doi.org/10.1234. "
            "Williams, 2022. pp. 45-50."
        )
        assert is_citation_text(text) is True


# ──────────────────────────────────────────────
# clean_chunk_text
# ──────────────────────────────────────────────

class TestCleanChunkText:
    def test_removes_image_markdown(self):
        text = "Text before ![alt](http://img.png) text after."
        result = clean_chunk_text(text)
        assert "![" not in result
        assert "text after" in result

    def test_removes_link_keeps_label(self):
        text = "See [this paper](http://example.com) for details."
        result = clean_chunk_text(text)
        assert "this paper" in result
        assert "http://example.com" not in result

    def test_removes_markdown_headers(self):
        text = "## Section Title\nContent below."
        result = clean_chunk_text(text)
        assert result.startswith("Section Title")

    def test_removes_unicode_replacement(self):
        text = "Some text with \ufffd broken chars."
        result = clean_chunk_text(text)
        assert "\ufffd" not in result

    def test_removes_markdown_table_remnants(self):
        text = "Normal text.\n|Col1|Col2|Col3|\nMore text."
        result = clean_chunk_text(text)
        assert "|Col1" not in result

    def test_collapses_excessive_whitespace(self):
        text = "Line one.\n\n\n\n\nLine two."
        result = clean_chunk_text(text)
        assert "\n\n\n" not in result

    def test_preserves_latex_commands(self):
        text = r"The loss function $\mathcal{L}$ is minimised."
        result = clean_chunk_text(text)
        assert r"$\mathcal{L}$" in result


# ──────────────────────────────────────────────
# is_quality_chunk
# ──────────────────────────────────────────────

class TestIsQualityChunk:
    def test_short_text_rejected(self):
        text = "Too short to be useful."
        assert is_quality_chunk(text, min_words=30) is False

    def test_sufficient_length_accepted(self):
        text = " ".join(["word"] * 50)
        assert is_quality_chunk(text) is True

    def test_low_alpha_ratio_rejected(self):
        text = " ".join(["123"] * 50)
        assert is_quality_chunk(text) is False

    def test_citation_heavy_rejected(self):
        text = (
            "Smith, 2020. Jones, 2021. pp. 12-15. "
            "Brown, 2019. In _NeurIPS_. https://doi.org/10.1234. "
            "Williams, 2022. pp. 45-50. " * 3
        )
        assert is_quality_chunk(text) is False

    def test_good_academic_text_accepted(self):
        text = (
            "Retrieval-Augmented Generation combines a retriever component with a "
            "language model generator. The retriever fetches relevant documents from "
            "a large corpus based on the input query. The generator then produces "
            "an answer conditioned on both the query and the retrieved context."
        )
        assert is_quality_chunk(text) is True


# ──────────────────────────────────────────────
# generate_chunk_id
# ──────────────────────────────────────────────

class TestGenerateChunkId:
    def test_deterministic(self):
        id1 = generate_chunk_id("2401.12345", "Introduction", 0)
        id2 = generate_chunk_id("2401.12345", "Introduction", 0)
        assert id1 == id2

    def test_different_inputs_different_ids(self):
        id1 = generate_chunk_id("2401.12345", "Introduction", 0)
        id2 = generate_chunk_id("2401.12345", "Method", 0)
        assert id1 != id2

    def test_length_is_12(self):
        chunk_id = generate_chunk_id("2401.12345", "Abstract", 0)
        assert len(chunk_id) == 12


# ──────────────────────────────────────────────
# chunk_text (token-based splitting)
# ──────────────────────────────────────────────

class TestChunkText:
    """Tests use a mock tokenizer to avoid loading the real model."""

    @patch("src.api.core.chunker._get_tokenizer")
    def test_short_text_single_chunk(self, mock_get_tok):
        mock_tok = MagicMock()
        # Simulate 100 tokens for short text
        mock_tok.encode.return_value = list(range(100))
        mock_tok.decode.return_value = "short text content"
        mock_get_tok.return_value = mock_tok

        result = chunk_text("short text content", chunk_size=450, overlap=64)
        assert len(result) == 1

    @patch("src.api.core.chunker._get_tokenizer")
    def test_long_text_multiple_chunks(self, mock_get_tok):
        mock_tok = MagicMock()
        # Simulate 1000 tokens for long text
        mock_tok.encode.return_value = list(range(1000))
        mock_tok.decode.return_value = " ".join(["word"] * 50)
        mock_get_tok.return_value = mock_tok

        result = chunk_text("long " * 500, chunk_size=450, overlap=64)
        assert len(result) > 1

    @patch("src.api.core.chunker._get_tokenizer")
    def test_overlap_creates_more_chunks(self, mock_get_tok):
        mock_tok = MagicMock()
        mock_tok.encode.return_value = list(range(900))
        mock_tok.decode.return_value = " ".join(["word"] * 50)
        mock_get_tok.return_value = mock_tok

        result_no_overlap = chunk_text("text", chunk_size=450, overlap=0)
        result_with_overlap = chunk_text("text", chunk_size=450, overlap=64)
        assert len(result_with_overlap) >= len(result_no_overlap)


# ──────────────────────────────────────────────
# chunk_paper
# ──────────────────────────────────────────────

class TestChunkPaper:
    @patch("src.api.core.chunker._get_tokenizer")
    def test_excludes_references_section(self, mock_get_tok):
        mock_tok = MagicMock()
        mock_tok.encode.return_value = list(range(100))
        mock_tok.decode.return_value = " ".join(["word"] * 50)
        mock_get_tok.return_value = mock_tok

        paper = {
            "arxiv_id": "2401.12345",
            "title": "Test Paper",
            "sections": {
                "Introduction": " ".join(["introduction content"] * 30),
                "References": "[1] Author et al. 2023. " * 20,
            },
        }
        chunks = chunk_paper(paper)
        sections = {c.section for c in chunks}
        assert "References" not in sections

    @patch("src.api.core.chunker._get_tokenizer")
    def test_full_text_fallback(self, mock_get_tok):
        mock_tok = MagicMock()
        mock_tok.encode.return_value = list(range(100))
        mock_tok.decode.return_value = " ".join(["word"] * 50)
        mock_get_tok.return_value = mock_tok

        paper = {
            "arxiv_id": "2401.12345",
            "title": "Test Paper",
            "full_text": " ".join(["content"] * 100),
            "sections": {},
        }
        chunks = chunk_paper(paper)
        assert len(chunks) >= 1
        assert chunks[0].section == "full_text"

    @patch("src.api.core.chunker._get_tokenizer")
    def test_empty_paper_no_chunks(self, mock_get_tok):
        mock_tok = MagicMock()
        mock_tok.encode.return_value = []
        mock_tok.decode.return_value = ""
        mock_get_tok.return_value = mock_tok

        paper = {
            "arxiv_id": "2401.12345",
            "title": "Empty Paper",
            "sections": {},
            "full_text": "",
        }
        chunks = chunk_paper(paper)
        assert len(chunks) == 0
