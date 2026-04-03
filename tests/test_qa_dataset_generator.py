"""
Unit tests for src/finetuning/generate_qa_dataset.py

This module validates the synthetic Q&A dataset generator used to produce
training data for QLoRA fine-tuning of Llama3/Qwen3.  All tests are pure unit
tests — no real LLM calls are made.  The Ollama HTTP API is mocked via
``unittest.mock.patch``.

Coverage
--------
parse_json_response
    Validates JSON extraction from raw LLM output strings.
    Edge cases: embedded JSON, None input, empty string, malformed JSON,
    missing ``question``/``answer`` keys, minimum length guards (>10 chars
    for question, >20 chars for answer), extra keys, nested braces, Unicode.

load_chunks
    Validates chunk loading and paper-grouping logic.
    Edge cases: multiple chunks per paper, single chunk, empty file.

generate_type1  (context-grounded Q&A)
    Schema contract: required keys, correct ``type`` value, instruction equals
    ``RAG_SYSTEM_PROMPT``, title and question present in ``input``,
    ``source_arxiv_id`` matches chunk, None on LLM failure or invalid JSON,
    text truncated to 500 chars before sending to LLM.

generate_type2  (multi-paper synthesis Q&A)
    Schema contract: required keys, ``type == "synthesis"``, both arxiv IDs in
    ``source_arxiv_id``, both paper titles in ``input``, None on LLM failure,
    text truncated to 400 chars per paper.

generate_type3  (refusal Q&A)
    Schema contract: required keys, ``type == "refusal"``, topic-map keyword
    detection for 7 title patterns (rag, qlora, lora, hallucin, prompt,
    instruct, default), refusal phrase in output, None on LLM failure.

DatasetSampleContract  (integration-style)
    All three type generators produce samples with the exact 5-key schema
    expected by the fine-tuning pipeline: ``{type, instruction, input,
    output, source_arxiv_id}``.  Type values are restricted to
    ``{grounded, synthesis, refusal}``.

Bug regression
--------------
``generate_type3`` topic_map previously listed ``lora`` before ``qlora``,
causing "QLoRA: ..." titles to resolve to "parameter-efficient fine-tuning"
instead of "quantised fine-tuning".  The ``test_topic_map_keyword_detection``
parametrised test catches this regression.
"""

import json
from unittest.mock import patch

import pytest

from src.finetuning.generate_qa_dataset import (
    RAG_SYSTEM_PROMPT,
    generate_type1,
    generate_type2,
    generate_type3,
    parse_json_response,
)

# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

def _make_chunk(
    arxiv_id="2401.00001",
    title="Attention Is All You Need",
    section="Introduction",
    text=None,
    word_count=120,
):
    if text is None:
        text = (
            "The Transformer model relies entirely on an attention mechanism to draw global "
            "dependencies between input and output. This architecture achieves superior results "
            "on machine translation tasks compared to recurrent and convolutional models."
        )
    return {
        "chunk_id": f"{arxiv_id}_{section}_0",
        "arxiv_id": arxiv_id,
        "title": title,
        "section": section,
        "text": text,
        "word_count": word_count,
    }


# ─────────────────────────────────────────────
# parse_json_response
# ─────────────────────────────────────────────

class TestParseJsonResponse:
    """Tests for ``parse_json_response``.

    The function must extract a ``{"question": ..., "answer": ...}`` object
    from a raw LLM output string that may contain surrounding prose, whitespace,
    or thinking-mode artefacts.  It returns ``None`` for any input that does not
    yield a usable Q&A pair.
    """
    def test_valid_json_with_question_and_answer(self):
        raw = '{"question": "What is the Transformer?", "answer": "The Transformer is a seq2seq model based solely on attention."}'
        result = parse_json_response(raw)
        assert result is not None
        assert result["question"] == "What is the Transformer?"
        assert "attention" in result["answer"]

    def test_json_embedded_in_surrounding_text(self):
        raw = 'Sure! Here is the pair: {"question": "What does RAG stand for?", "answer": "Retrieval-Augmented Generation, a technique that combines retrieval with generation."} Hope that helps.'
        result = parse_json_response(raw)
        assert result is not None
        assert "question" in result
        assert "answer" in result

    def test_returns_none_for_empty_string(self):
        assert parse_json_response("") is None

    def test_returns_none_for_none_input(self):
        assert parse_json_response(None) is None

    def test_returns_none_for_malformed_json(self):
        raw = '{"question": "What is X?", "answer": incomplete'
        assert parse_json_response(raw) is None

    def test_returns_none_when_question_key_missing(self):
        raw = '{"answer": "A long enough answer to pass the length guard."}'
        assert parse_json_response(raw) is None

    def test_returns_none_when_answer_key_missing(self):
        raw = '{"question": "What is this paper about?"}'
        assert parse_json_response(raw) is None

    def test_returns_none_when_question_too_short(self):
        # question <= 10 chars should be rejected
        raw = '{"question": "Short?", "answer": "A sufficiently long answer to pass the length check."}'
        assert parse_json_response(raw) is None

    def test_returns_none_when_answer_too_short(self):
        # answer <= 20 chars should be rejected
        raw = '{"question": "What is Retrieval-Augmented Generation?", "answer": "Short ans."}'
        assert parse_json_response(raw) is None

    def test_extra_keys_are_tolerated(self):
        raw = '{"question": "How does attention work?", "answer": "Attention computes a weighted sum of value vectors based on query-key similarity scores.", "confidence": 0.9}'
        result = parse_json_response(raw)
        assert result is not None

    def test_nested_braces_parsed_correctly(self):
        raw = '{"question": "What does the formula represent?", "answer": "The formula {x | x > 0} defines positive reals in set notation."}'
        result = parse_json_response(raw)
        assert result is not None

    def test_unicode_content_preserved(self):
        raw = '{"question": "What language was the paper written in?", "answer": "The paper was written in English and contains mathematical notation such as Σ and μ."}'
        result = parse_json_response(raw)
        assert result is not None
        assert "Σ" in result["answer"]


# ─────────────────────────────────────────────
# generate_type1
# ─────────────────────────────────────────────

class TestGenerateType1:
    """Tests for ``generate_type1`` — context-grounded single-paper Q&A.

    Each call should produce a dict with keys
    ``{type, instruction, input, output, source_arxiv_id}`` where:
    - ``type`` is ``"grounded"``
    - ``instruction`` is the RAG system prompt verbatim
    - ``input`` contains the paper title and the generated question
    - ``output`` is the generated answer
    - Only the first 500 chars of ``chunk["text"]`` are sent to the LLM
    """
    def _valid_llm_response(self):
        return json.dumps({
            "question": "What mechanism does the Transformer use?",
            "answer": "According to 'Attention Is All You Need', the Transformer uses self-attention to capture global dependencies without recurrence."
        })

    def test_returns_correct_schema(self):
        chunk = _make_chunk()
        with patch("src.finetuning.generate_qa_dataset.call_llm", return_value=self._valid_llm_response()):
            result = generate_type1(chunk)

        assert result is not None
        assert result["type"] == "grounded"
        assert "instruction" in result
        assert "input" in result
        assert "output" in result
        assert "source_arxiv_id" in result

    def test_instruction_is_rag_system_prompt(self):
        chunk = _make_chunk()
        with patch("src.finetuning.generate_qa_dataset.call_llm", return_value=self._valid_llm_response()):
            result = generate_type1(chunk)
        assert result["instruction"] == RAG_SYSTEM_PROMPT

    def test_input_contains_chunk_title(self):
        chunk = _make_chunk(title="BERT: Pre-training of Deep Bidirectional Transformers")
        with patch("src.finetuning.generate_qa_dataset.call_llm", return_value=self._valid_llm_response()):
            result = generate_type1(chunk)
        assert "BERT: Pre-training of Deep Bidirectional Transformers" in result["input"]

    def test_input_contains_question(self):
        chunk = _make_chunk()
        with patch("src.finetuning.generate_qa_dataset.call_llm", return_value=self._valid_llm_response()):
            result = generate_type1(chunk)
        assert "What mechanism does the Transformer use?" in result["input"]

    def test_source_arxiv_id_matches_chunk(self):
        chunk = _make_chunk(arxiv_id="2301.99999")
        with patch("src.finetuning.generate_qa_dataset.call_llm", return_value=self._valid_llm_response()):
            result = generate_type1(chunk)
        assert result["source_arxiv_id"] == "2301.99999"

    def test_returns_none_when_llm_fails(self):
        chunk = _make_chunk()
        with patch("src.finetuning.generate_qa_dataset.call_llm", return_value=None):
            result = generate_type1(chunk)
        assert result is None

    def test_returns_none_when_llm_returns_invalid_json(self):
        chunk = _make_chunk()
        with patch("src.finetuning.generate_qa_dataset.call_llm", return_value="Not valid JSON at all"):
            result = generate_type1(chunk)
        assert result is None

    def test_text_truncated_to_500_chars_in_prompt(self):
        long_text = "word " * 300  # well over 500 chars
        chunk = _make_chunk(text=long_text)
        captured_prompts = []

        def mock_llm(prompt, max_retries=2):
            captured_prompts.append(prompt)
            return json.dumps({
                "question": "What is described in this excerpt?",
                "answer": "The excerpt describes repeated words as placeholder content for testing purposes."
            })

        with patch("src.finetuning.generate_qa_dataset.call_llm", side_effect=mock_llm):
            generate_type1(chunk)

        assert len(captured_prompts) == 1
        # The prompt should contain the truncated text (first 500 chars) but not beyond
        prompt = captured_prompts[0]
        assert long_text[:500] in prompt
        assert long_text[500:] not in prompt


# ─────────────────────────────────────────────
# generate_type2
# ─────────────────────────────────────────────

class TestGenerateType2:
    """Tests for ``generate_type2`` — multi-paper synthesis Q&A.

    Takes two chunks from *different* papers and asks the LLM to synthesise a
    comparison question.  ``source_arxiv_id`` must contain both paper IDs.
    Only the first 400 chars of each chunk's text are forwarded to the LLM.
    """
    def _valid_llm_response(self):
        return json.dumps({
            "question": "How do the two approaches differ in their use of attention?",
            "answer": (
                "'Attention Is All You Need' uses multi-head self-attention exclusively, "
                "while 'BERT' adds bidirectional pre-training on top of the same mechanism "
                "to capture richer contextual representations."
            )
        })

    def test_returns_correct_schema(self):
        chunk1 = _make_chunk(arxiv_id="2401.00001", title="Attention Is All You Need")
        chunk2 = _make_chunk(arxiv_id="2401.00002", title="BERT", section="Method")
        with patch("src.finetuning.generate_qa_dataset.call_llm", return_value=self._valid_llm_response()):
            result = generate_type2(chunk1, chunk2)

        assert result is not None
        assert result["type"] == "synthesis"
        assert "instruction" in result
        assert "input" in result
        assert "output" in result
        assert "source_arxiv_id" in result

    def test_source_arxiv_id_contains_both_papers(self):
        chunk1 = _make_chunk(arxiv_id="2401.00001")
        chunk2 = _make_chunk(arxiv_id="2401.00002", section="Method")
        with patch("src.finetuning.generate_qa_dataset.call_llm", return_value=self._valid_llm_response()):
            result = generate_type2(chunk1, chunk2)

        assert "2401.00001" in result["source_arxiv_id"]
        assert "2401.00002" in result["source_arxiv_id"]

    def test_input_contains_both_titles(self):
        chunk1 = _make_chunk(arxiv_id="2401.00001", title="Paper Alpha")
        chunk2 = _make_chunk(arxiv_id="2401.00002", title="Paper Beta", section="Method")
        with patch("src.finetuning.generate_qa_dataset.call_llm", return_value=self._valid_llm_response()):
            result = generate_type2(chunk1, chunk2)

        assert "Paper Alpha" in result["input"]
        assert "Paper Beta" in result["input"]

    def test_instruction_is_rag_system_prompt(self):
        chunk1 = _make_chunk()
        chunk2 = _make_chunk(arxiv_id="2401.00002", section="Conclusion")
        with patch("src.finetuning.generate_qa_dataset.call_llm", return_value=self._valid_llm_response()):
            result = generate_type2(chunk1, chunk2)
        assert result["instruction"] == RAG_SYSTEM_PROMPT

    def test_returns_none_when_llm_fails(self):
        chunk1 = _make_chunk()
        chunk2 = _make_chunk(arxiv_id="2401.00002", section="Conclusion")
        with patch("src.finetuning.generate_qa_dataset.call_llm", return_value=None):
            result = generate_type2(chunk1, chunk2)
        assert result is None

    def test_text_truncated_to_400_chars(self):
        long_text = "token " * 200  # well over 400 chars
        chunk1 = _make_chunk(text=long_text)
        chunk2 = _make_chunk(arxiv_id="2401.00002", section="Method", text=long_text)
        captured = []

        def mock_llm(prompt, max_retries=2):
            captured.append(prompt)
            return json.dumps({
                "question": "How do the two papers compare in methodology?",
                "answer": "Both papers present similar methodologies with subtle differences in scope and evaluation."
            })

        with patch("src.finetuning.generate_qa_dataset.call_llm", side_effect=mock_llm):
            generate_type2(chunk1, chunk2)

        prompt = captured[0]
        assert long_text[:400] in prompt
        assert long_text[400:] not in prompt


# ─────────────────────────────────────────────
# generate_type3
# ─────────────────────────────────────────────

class TestGenerateType3:
    """Tests for ``generate_type3`` — refusal Q&A (context-insufficient).

    Generates a question that *cannot* be answered from the chunk, plus a
    well-formed refusal answer that names the gap.  The topic injected into
    the prompt is derived from a keyword lookup on the paper title (lower-cased).
    The lookup order must ensure ``qlora`` is checked before ``lora``.
    """
    def _valid_llm_response(self):
        return json.dumps({
            "question": "What hardware was used to train the Transformer?",
            "answer": (
                "The provided context does not contain sufficient information to answer this question. "
                "The papers discuss the Transformer architecture and attention mechanisms, but hardware "
                "specifications are not addressed."
            )
        })

    def test_returns_correct_schema(self):
        chunk = _make_chunk()
        with patch("src.finetuning.generate_qa_dataset.call_llm", return_value=self._valid_llm_response()):
            result = generate_type3(chunk)

        assert result is not None
        assert result["type"] == "refusal"
        assert "instruction" in result
        assert "input" in result
        assert "output" in result
        assert "source_arxiv_id" in result

    def test_instruction_is_rag_system_prompt(self):
        chunk = _make_chunk()
        with patch("src.finetuning.generate_qa_dataset.call_llm", return_value=self._valid_llm_response()):
            result = generate_type3(chunk)
        assert result["instruction"] == RAG_SYSTEM_PROMPT

    def test_returns_none_when_llm_fails(self):
        chunk = _make_chunk()
        with patch("src.finetuning.generate_qa_dataset.call_llm", return_value=None):
            result = generate_type3(chunk)
        assert result is None

    @pytest.mark.parametrize("title,expected_topic", [
        ("Advances in RAG Systems", "retrieval-augmented generation"),
        ("QLoRA: Efficient Fine-tuning", "quantised fine-tuning"),
        ("LoRA: Low-Rank Adaptation", "parameter-efficient fine-tuning"),
        ("Hallucination in LLMs", "LLM hallucination"),
        ("Prompt Engineering for GPT", "prompt engineering"),
        ("Instruction Tuning Methods", "instruction tuning"),
        ("Unrelated Title", "machine learning"),  # default fallback
    ])
    def test_topic_map_keyword_detection(self, title, expected_topic):
        chunk = _make_chunk(title=title)
        captured_prompts = []

        def mock_llm(prompt, max_retries=2):
            captured_prompts.append(prompt)
            return self._valid_llm_response()

        with patch("src.finetuning.generate_qa_dataset.call_llm", side_effect=mock_llm):
            generate_type3(chunk)

        assert expected_topic in captured_prompts[0]

    def test_refusal_output_contains_expected_phrase(self):
        chunk = _make_chunk()
        with patch("src.finetuning.generate_qa_dataset.call_llm", return_value=self._valid_llm_response()):
            result = generate_type3(chunk)
        assert "does not contain sufficient information" in result["output"]


# ─────────────────────────────────────────────
# load_chunks
# ─────────────────────────────────────────────

class TestLoadChunks:
    """Tests for ``load_chunks``.

    Reads ``chunks.json`` from disk and returns ``(chunks_list, papers_dict)``.
    ``papers_dict`` groups chunks by ``arxiv_id`` with structure
    ``{arxiv_id: {"title": str, "chunks": list}}``.
    Uses ``tmp_path`` to avoid touching real data files.
    """
    def _make_chunks_json(self, chunks: list) -> str:
        return json.dumps({"chunks": chunks})

    def test_returns_chunks_list_and_papers_dict(self, tmp_path):
        raw = [
            _make_chunk(arxiv_id="2401.00001", title="Paper A"),
            _make_chunk(arxiv_id="2401.00002", title="Paper B", section="Method"),
        ]
        chunks_file = tmp_path / "chunks.json"
        chunks_file.write_text(self._make_chunks_json(raw), encoding="utf-8")

        with patch("src.finetuning.generate_qa_dataset.CHUNKS_FILE", chunks_file):
            from src.finetuning.generate_qa_dataset import load_chunks
            chunks, papers = load_chunks()

        assert len(chunks) == 2
        assert isinstance(papers, dict)

    def test_papers_grouped_by_arxiv_id(self, tmp_path):
        raw = [
            _make_chunk(arxiv_id="2401.00001", title="Paper A", section="Intro"),
            _make_chunk(arxiv_id="2401.00001", title="Paper A", section="Method"),
            _make_chunk(arxiv_id="2401.00002", title="Paper B", section="Intro"),
        ]
        chunks_file = tmp_path / "chunks.json"
        chunks_file.write_text(self._make_chunks_json(raw), encoding="utf-8")

        with patch("src.finetuning.generate_qa_dataset.CHUNKS_FILE", chunks_file):
            from src.finetuning.generate_qa_dataset import load_chunks
            _, papers = load_chunks()

        assert len(papers) == 2
        assert len(papers["2401.00001"]["chunks"]) == 2
        assert len(papers["2401.00002"]["chunks"]) == 1

    def test_paper_entry_has_title_and_chunks(self, tmp_path):
        raw = [_make_chunk(arxiv_id="2401.00001", title="Test Paper")]
        chunks_file = tmp_path / "chunks.json"
        chunks_file.write_text(self._make_chunks_json(raw), encoding="utf-8")

        with patch("src.finetuning.generate_qa_dataset.CHUNKS_FILE", chunks_file):
            from src.finetuning.generate_qa_dataset import load_chunks
            _, papers = load_chunks()

        paper = papers["2401.00001"]
        assert paper["title"] == "Test Paper"
        assert isinstance(paper["chunks"], list)

    def test_empty_chunks_file(self, tmp_path):
        chunks_file = tmp_path / "chunks.json"
        chunks_file.write_text(json.dumps({"chunks": []}), encoding="utf-8")

        with patch("src.finetuning.generate_qa_dataset.CHUNKS_FILE", chunks_file):
            from src.finetuning.generate_qa_dataset import load_chunks
            chunks, papers = load_chunks()

        assert chunks == []
        assert papers == {}


# ─────────────────────────────────────────────
# Dataset sample contract (integration-style, no LLM)
# ─────────────────────────────────────────────

class TestDatasetSampleContract:
    """Integration-style contract tests for the fine-tuning dataset schema.

    Every sample written to ``qa_dataset.json`` must satisfy the schema
    expected by the SFTTrainer training loop:
    - Required keys: ``type``, ``instruction``, ``input``, ``output``,
      ``source_arxiv_id``
    - ``type`` must be one of ``{"grounded", "synthesis", "refusal"}``

    These tests verify the contract without touching the filesystem or LLM.
    """
    """Verify every generated sample has the fields expected by fine-tuning."""

    REQUIRED_KEYS = {"type", "instruction", "input", "output", "source_arxiv_id"}
    VALID_TYPES = {"grounded", "synthesis", "refusal"}

    def _grounded_response(self):
        return json.dumps({
            "question": "What does the model learn during pre-training?",
            "answer": "According to the paper, the model learns contextual representations by predicting masked tokens bidirectionally."
        })

    def test_type1_sample_has_all_required_keys(self):
        chunk = _make_chunk()
        with patch("src.finetuning.generate_qa_dataset.call_llm", return_value=self._grounded_response()):
            result = generate_type1(chunk)
        assert self.REQUIRED_KEYS.issubset(result.keys())

    def test_type2_sample_has_all_required_keys(self):
        chunk1 = _make_chunk(arxiv_id="2401.00001")
        chunk2 = _make_chunk(arxiv_id="2401.00002", section="Method")
        resp = json.dumps({
            "question": "How do the two papers differ in training objectives?",
            "answer": "The first paper uses masked language modelling, while the second uses contrastive learning as its primary training objective."
        })
        with patch("src.finetuning.generate_qa_dataset.call_llm", return_value=resp):
            result = generate_type2(chunk1, chunk2)
        assert self.REQUIRED_KEYS.issubset(result.keys())

    def test_type3_sample_has_all_required_keys(self):
        chunk = _make_chunk()
        resp = json.dumps({
            "question": "What dataset size was used for pre-training?",
            "answer": "The provided context does not contain sufficient information to answer this question. The papers discuss model architecture but dataset size is not addressed."
        })
        with patch("src.finetuning.generate_qa_dataset.call_llm", return_value=resp):
            result = generate_type3(chunk)
        assert self.REQUIRED_KEYS.issubset(result.keys())

    def test_type_values_are_valid(self):
        chunk = _make_chunk()
        responses = [
            json.dumps({"question": "What is attention?", "answer": "Attention is a mechanism that computes weighted sums over value vectors guided by query-key similarity."}),
            json.dumps({"question": "What is attention?", "answer": "Attention is a mechanism that computes weighted sums over value vectors guided by query-key similarity."}),
            json.dumps({"question": "What dataset was used?", "answer": "The provided context does not contain sufficient information to answer this question. The papers discuss architecture but not datasets."}),
        ]
        with patch("src.finetuning.generate_qa_dataset.call_llm", side_effect=responses):
            r1 = generate_type1(chunk)
            r2 = generate_type2(chunk, _make_chunk(arxiv_id="2401.00002", section="Conclusion"))
            r3 = generate_type3(chunk)

        assert r1["type"] in self.VALID_TYPES
        assert r2["type"] in self.VALID_TYPES
        assert r3["type"] in self.VALID_TYPES
