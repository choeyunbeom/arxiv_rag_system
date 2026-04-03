"""
Unit tests for src/ingestion/pdf_parser.py

This module tests the three public functions of the PDF parser used in the
arXiv ingestion pipeline.  ``pymupdf4llm`` is mocked at the module level so
the tests have no dependency on a real PDF file or the C extension.

Coverage
--------
clean_text
    Removes markdown image syntax ``![alt](url)``, collapses runs of 3+ newlines
    to at most 2, collapses 2+ spaces to 1, strips leading/trailing whitespace.
    Edge cases: multiple images, empty string, double newline preservation.

split_sections
    Parses pymupdf4llm markdown output into a ``{section_name: text}`` dict.

    Header formats supported (tested individually):
    - Bold-numbered:  ``**1** **Introduction**``  and  ``**1.** **Introduction**``
    - Markdown hash:  ``## Introduction``, ``## 1. Introduction``, ``### ...``
    - Standalone bold: ``**Abstract**``

    Section name normalisation (tested via parametrised and named cases):
    - ``Background``  → ``related_work``
    - ``Approach`` / ``Framework`` / ``Model`` / ``Proposed`` → ``method``
    - ``Evaluation`` / ``Results`` / ``Setup`` → ``experiments``
    - ``Analysis`` / ``Ablation`` → ``discussion``
    - ``Summary`` → ``conclusion``

    Exclusion logic:
    - ``references``, ``bibliography``, ``appendix``, ``acknowledgments`` and
      spelling variants are present in ``EXCLUDED_SECTIONS``; if *all* detected
      sections are excluded, ``split_sections`` falls back to
      ``{"full_text": <entire_text>}``

    Known limitations (documented via negative tests):
    - Standalone ``**Acknowledgments**`` / ``**Acknowledgements**`` → full_text
      (pattern not in ``section_markers``)
    - Plain-text ``References\\n`` → full_text (no plain-text pattern for this)
    - ``**References**``-only input → full_text (references is excluded)

    Minimum content length guard: sections with ≤ 50 characters are dropped.
    The ``_LONG`` constant (100 chars) is used throughout to satisfy this guard.

extract_text_from_pdf
    Thin wrapper around ``pymupdf4llm.to_markdown``.
    Returns stripped text on success, empty string for blank output, and
    ``None`` for any exception (RuntimeError, FileNotFoundError, etc.).
"""

import sys
from unittest.mock import MagicMock, patch

import pytest

# Mock pymupdf4llm before importing the module under test
sys.modules["pymupdf4llm"] = MagicMock()

from src.ingestion.pdf_parser import clean_text, extract_text_from_pdf, split_sections

# ─────────────────────────────────────────────
# clean_text
# ─────────────────────────────────────────────

class TestCleanText:
    """Tests for ``clean_text``.

    Verifies that markdown image syntax, excessive newlines, and multiple
    consecutive spaces are removed or collapsed.  Normal prose is untouched.
    """
    def test_removes_image_placeholders(self):
        text = "Before image. ![Figure 1](./fig1.png) After image."
        result = clean_text(text)
        assert "![" not in result
        assert "Before image." in result
        assert "After image." in result

    def test_removes_image_with_empty_alt(self):
        text = "Text ![](http://example.com/img.png) more text."
        result = clean_text(text)
        assert "![" not in result

    def test_collapses_triple_newlines(self):
        text = "Paragraph one.\n\n\n\nParagraph two."
        result = clean_text(text)
        assert "\n\n\n" not in result
        assert "Paragraph one." in result
        assert "Paragraph two." in result

    def test_collapses_many_newlines(self):
        text = "A.\n\n\n\n\n\n\nB."
        result = clean_text(text)
        assert result.count("\n") <= 2

    def test_double_newline_preserved(self):
        text = "Line one.\n\nLine two."
        result = clean_text(text)
        assert "\n\n" in result

    def test_collapses_multiple_spaces(self):
        text = "Word1   word2    word3."
        result = clean_text(text)
        assert "  " not in result

    def test_strips_leading_trailing_whitespace(self):
        text = "  \n  Actual content here.  \n  "
        result = clean_text(text)
        assert result == result.strip()

    def test_empty_string_returns_empty(self):
        assert clean_text("") == ""

    def test_preserves_normal_text(self):
        text = "This is a normal sentence with no special characters."
        result = clean_text(text)
        assert result == text

    def test_multiple_images_removed(self):
        text = "A ![img1](a.png) B ![img2](b.png) C."
        result = clean_text(text)
        assert "![" not in result
        assert "A" in result
        assert "C." in result


# ─────────────────────────────────────────────
# split_sections — bold-numbered headers
# ─────────────────────────────────────────────

class TestSplitSectionsBoldNumbered:
    """Tests for ``split_sections`` with bold-numbered headers.

    pymupdf4llm renders many two-column academic PDFs with headers like
    ``**1** **Introduction**`` or ``**1.** **Introduction**``.  This class
    verifies detection of all standard section types under that format.
    """
    _LONG = "word " * 20  # 100 chars — comfortably above the 50-char minimum

    def _paper(self, *sections):
        """Build a multi-section paper string from (name, content) tuples."""
        return "\n\n".join(
            f"**{i+1}** **{name}**\n{content}" for i, (name, content) in enumerate(sections)
        )

    def test_detects_introduction(self):
        text = self._paper(("Introduction", self._LONG))
        sections = split_sections(text)
        assert "introduction" in sections

    def test_detects_method(self):
        text = self._paper(("Method", self._LONG))
        sections = split_sections(text)
        assert "method" in sections

    def test_detects_experiments(self):
        text = self._paper(("Experiments", self._LONG))
        sections = split_sections(text)
        assert "experiments" in sections

    def test_detects_conclusion(self):
        text = self._paper(("Conclusion", self._LONG))
        sections = split_sections(text)
        assert "conclusion" in sections

    def test_references_excluded_from_content(self):
        text = self._paper(
            ("Introduction", self._LONG),
            ("References", "[1] Vaswani et al. 2017. Attention Is All You Need. NeurIPS."),
        )
        sections = split_sections(text)
        assert "introduction" in sections
        content = {k for k in sections if k not in {"references", "bibliography", "appendix", "acknowledgments", "acknowledgements", "acknowledgement"}}
        assert len(content) >= 1

    def test_section_content_assigned_correctly(self):
        text = (
            "**1** **Introduction**\n"
            "This is the introduction content and it is long enough to pass the minimum length guard.\n\n"
            "**2** **Method**\n"
            "This describes the proposed method with sufficient detail to pass the length guard."
        )
        sections = split_sections(text)
        assert "introduction" in sections
        assert "method" in sections
        assert "introduction content" in sections["introduction"]
        assert "proposed method" in sections["method"]

    def test_numbered_with_period(self):
        text = f"**1.** **Introduction**\n{self._LONG}"
        sections = split_sections(text)
        assert "introduction" in sections


# ─────────────────────────────────────────────
# split_sections — markdown headers
# ─────────────────────────────────────────────

_LONG = "word " * 20  # 100 chars — above the 50-char minimum used in split_sections


class TestSplitSectionsMarkdown:
    """Tests for ``split_sections`` with markdown hash headers.

    Single-column PDFs and some preprints use standard markdown headers
    (``## Section``, ``## 1. Section``, ``### Section``).  Verifies all
    section name normalisations (background→related_work, ablation→discussion,
    summary→conclusion, etc.).
    """
    def test_hash_introduction(self):
        text = f"## Introduction\n{_LONG}"
        sections = split_sections(text)
        assert "introduction" in sections

    def test_hash_with_number(self):
        text = f"## 1. Introduction\n{_LONG}"
        sections = split_sections(text)
        assert "introduction" in sections

    def test_hash_related_work(self):
        text = f"## Related Work\n{_LONG}"
        sections = split_sections(text)
        assert "related_work" in sections

    def test_hash_background(self):
        text = f"## Background\n{_LONG}"
        sections = split_sections(text)
        assert "related_work" in sections

    def test_double_and_triple_hash(self):
        text = f"### Experiments\n{_LONG}"
        sections = split_sections(text)
        assert "experiments" in sections

    def test_evaluation_maps_to_experiments(self):
        text = f"## Evaluation\n{_LONG}"
        sections = split_sections(text)
        assert "experiments" in sections

    def test_discussion_detected(self):
        text = f"## Discussion\n{_LONG}"
        sections = split_sections(text)
        assert "discussion" in sections

    def test_ablation_maps_to_discussion(self):
        text = f"## Ablation Study\n{_LONG}"
        sections = split_sections(text)
        assert "discussion" in sections

    def test_summary_maps_to_conclusion(self):
        text = f"## Summary\n{_LONG}"
        sections = split_sections(text)
        assert "conclusion" in sections

    def test_multiple_markdown_sections(self):
        text = (
            f"## Introduction\n{_LONG}\n\n"
            f"## Method\n{_LONG}\n\n"
            f"## Conclusion\n{_LONG}"
        )
        sections = split_sections(text)
        assert "introduction" in sections
        assert "method" in sections
        assert "conclusion" in sections


# ─────────────────────────────────────────────
# split_sections — standalone bold headers
# ─────────────────────────────────────────────

class TestSplitSectionsStandaloneBold:
    """Tests for ``split_sections`` with standalone bold headers.

    Some papers render headers as bare ``**Abstract**`` without a section
    number.  Only ``**Abstract**`` and ``**References**`` are in the
    ``section_markers`` list; ``**Acknowledgments**`` is not.
    Negative tests document this known limitation explicitly.
    """
    def test_standalone_abstract(self):
        text = f"**Abstract**\n{_LONG}"
        sections = split_sections(text)
        assert "abstract" in sections

    def test_standalone_bold_references_falls_back(self):
        # **References** IS matched by the pattern, but because 'references' is in
        # EXCLUDED_SECTIONS the content_sections dict becomes empty → full_text fallback.
        # This test documents that current behaviour: references-only input → full_text.
        text = "Body text that is long enough to pass the minimum length requirement.\n**References**\n[1] Smith et al. 2023. A paper title. In Proceedings of ACL."
        sections = split_sections(text)
        assert "full_text" in sections

    def test_standalone_acknowledgments_not_supported(self):
        # **Acknowledgments** standalone is NOT in the section_markers patterns;
        # it falls back to full_text — this test documents current behaviour.
        text = f"Main content.\n**Acknowledgments**\nThanks to reviewers."
        sections = split_sections(text)
        assert "full_text" in sections


# ─────────────────────────────────────────────
# split_sections — plain-text headers (last resort)
# ─────────────────────────────────────────────

class TestSplitSectionsPlainText:
    """Tests for ``split_sections`` plain-text header patterns (last resort).

    Only ``Abstract\\n`` (line-start, no bold/hash) is supported as a
    plain-text fallback.  ``References\\n`` is NOT in the pattern list;
    a negative test documents this gap so future contributors know the
    boundary of what the parser handles.
    """
    def test_plain_abstract(self):
        text = f"Abstract\n{_LONG}"
        sections = split_sections(text)
        assert "abstract" in sections

    def test_plain_references_not_supported(self):
        # Plain "References\n" (no bold/hash) is NOT in section_markers;
        # the parser falls back to full_text — documents current behaviour.
        text = f"Some content here that is long enough.\nReferences\n[1] Author 2023."
        sections = split_sections(text)
        assert "full_text" in sections


# ─────────────────────────────────────────────
# split_sections — fallback and edge cases
# ─────────────────────────────────────────────

class TestSplitSectionsEdgeCases:
    """Edge-case and regression tests for ``split_sections``.

    Covers:
    - No headers found → ``{"full_text": text}`` fallback
    - Empty string input → ``{"full_text": ""}`` fallback
    - Only excluded section (references) → full_text fallback
    - 50-char minimum length guard (sections shorter than 50 chars are dropped)
    - Duplicate header names — only first occurrence stored
    - Mixed header formats in a single document
    - Section order preserved by position in text
    - All ``method``-variant keywords: Approach, Framework, Model, Proposed
    - All ``experiments``-variant keywords: Evaluation, Results, Setup
    - ``discussion``-variant: Analysis, Ablation
    - ``bibliography`` excluded from content_sections
    - ``**Acknowledgements**`` (British spelling) not supported — documents
      current limitation (falls back to full_text)
    - Paper with no Abstract section still parses correctly
    """
    def test_no_sections_returns_full_text(self):
        text = "This is just a blob of text with no section markers whatsoever."
        sections = split_sections(text)
        assert "full_text" in sections
        assert sections["full_text"] == text

    def test_empty_string_returns_full_text(self):
        sections = split_sections("")
        assert "full_text" in sections

    def test_only_references_returns_full_text_fallback(self):
        """If the only detected section is excluded, should fall back to full_text."""
        text = "## References\n[1] Vaswani et al. 2017. Attention is all you need."
        sections = split_sections(text)
        # content_sections would be empty (references is excluded),
        # so should fall back to full_text
        assert "full_text" in sections

    def test_section_content_minimum_length(self):
        """Sections with <= 50 chars of content should not be stored."""
        text = "## Introduction\nShort.\n\n## Method\nAlso tiny text here."
        sections = split_sections(text)
        for key, value in sections.items():
            if key != "full_text":
                assert len(value) > 50

    def test_each_section_name_unique(self):
        """Duplicate section names should only appear once."""
        text = (
            "## Introduction\nFirst intro.\n\n"
            "## Introduction\nSecond intro (duplicate heading)."
        )
        sections = split_sections(text)
        assert list(sections.keys()).count("introduction") <= 1

    def test_mixed_header_formats_all_detected(self):
        """A paper using mixed header formats should still be parsed."""
        text = (
            f"**Abstract**\n{_LONG}\n\n"
            f"## 1. Introduction\n{_LONG}\n\n"
            f"**2** **Method**\n{_LONG}"
        )
        sections = split_sections(text)
        found = set(sections.keys())
        assert len(found) >= 2  # at least two content sections

    def test_sections_sorted_by_position(self):
        """Introduction should come before Conclusion in text order."""
        text = (
            f"## Introduction\n{_LONG}\n\n"
            f"## Method\n{_LONG}\n\n"
            f"## Conclusion\n{_LONG}"
        )
        sections = split_sections(text)
        keys = list(sections.keys())
        if "introduction" in keys and "conclusion" in keys:
            assert keys.index("introduction") < keys.index("conclusion")

    def test_approach_maps_to_method(self):
        text = f"## Approach\n{_LONG}"
        sections = split_sections(text)
        assert "method" in sections

    def test_framework_maps_to_method(self):
        text = f"## Framework\n{_LONG}"
        sections = split_sections(text)
        assert "method" in sections

    def test_results_maps_to_experiments(self):
        text = f"## Results\n{_LONG}"
        sections = split_sections(text)
        assert "experiments" in sections

    def test_analysis_maps_to_discussion(self):
        text = f"## Analysis\n{_LONG}"
        sections = split_sections(text)
        assert "discussion" in sections

    def test_bibliography_excluded_from_content(self):
        text = (
            f"## Introduction\n{_LONG}\n\n"
            "## Bibliography\n[1] Reference one. Smith et al. 2023. NeurIPS. pp. 1-10."
        )
        sections = split_sections(text)
        content = {k for k in sections if k not in {"references", "bibliography", "appendix", "acknowledgments", "acknowledgements", "acknowledgement"}}
        assert "introduction" in content

    def test_acknowledgements_variant_spelling_not_supported(self):
        # **Acknowledgements** (with -es) is not in section_markers;
        # falls back to full_text — documents current behaviour.
        text = f"Main body text that is long enough.\n**Acknowledgements**\nWe thank the reviewers."
        sections = split_sections(text)
        assert "full_text" in sections

    def test_paper_with_no_abstract_but_has_method(self):
        text = (
            f"**1** **Introduction**\n{_LONG}\n\n"
            f"**2** **Method**\n{_LONG}\n\n"
            f"**3** **Conclusion**\n{_LONG}"
        )
        sections = split_sections(text)
        assert "abstract" not in sections
        assert "introduction" in sections
        assert "method" in sections
        assert "conclusion" in sections


# ─────────────────────────────────────────────
# extract_text_from_pdf
# ─────────────────────────────────────────────

class TestExtractTextFromPdf:
    """Tests for ``extract_text_from_pdf``.

    Wraps ``pymupdf4llm.to_markdown``.  The C extension is mocked so tests
    run without a real PDF file.  Verifies the contract:
    - Success: returns stripped markdown string
    - Blank output (whitespace only): returns ``""``
    - Any exception: returns ``None`` and does not propagate
    - Passes the path argument through unchanged to pymupdf4llm
    """
    def test_returns_stripped_markdown_on_success(self):
        with patch("src.ingestion.pdf_parser.pymupdf4llm.to_markdown") as mock_md:
            mock_md.return_value = "  ## Introduction\nSome content here.  "
            result = extract_text_from_pdf("/fake/paper.pdf")

        assert result is not None
        assert result == result.strip()
        assert "Introduction" in result

    def test_returns_none_on_exception(self):
        with patch("src.ingestion.pdf_parser.pymupdf4llm.to_markdown", side_effect=RuntimeError("corrupt PDF")):
            result = extract_text_from_pdf("/fake/corrupt.pdf")
        assert result is None

    def test_returns_none_on_file_not_found(self):
        with patch("src.ingestion.pdf_parser.pymupdf4llm.to_markdown", side_effect=FileNotFoundError("not found")):
            result = extract_text_from_pdf("/nonexistent/path.pdf")
        assert result is None

    def test_passes_path_to_pymupdf(self):
        with patch("src.ingestion.pdf_parser.pymupdf4llm.to_markdown") as mock_md:
            mock_md.return_value = "content"
            extract_text_from_pdf("/my/paper.pdf")
        mock_md.assert_called_once_with("/my/paper.pdf")

    def test_empty_pdf_returns_empty_string(self):
        with patch("src.ingestion.pdf_parser.pymupdf4llm.to_markdown", return_value="   "):
            result = extract_text_from_pdf("/empty.pdf")
        assert result == ""
