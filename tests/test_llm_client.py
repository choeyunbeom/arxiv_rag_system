"""
Unit tests for src/api/core/llm_client.py

Tests cover:
- Qwen3 thinking mode tag cleaning
- Request payload construction (no_think injection, temperature, num_predict)
- Response parsing
- Error handling

External dependency (Ollama HTTP API) is mocked.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.api.core.llm_client import LLMClient

# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────

@pytest.fixture
def client():
    """Create an LLMClient with mocked async HTTP client."""
    with patch("src.api.core.llm_client.httpx.AsyncClient") as MockClient:
        mock_http = MagicMock()
        MockClient.return_value = mock_http
        llm = LLMClient(model="qwen3:4b")
        llm._mock_http = mock_http
        yield llm


# ──────────────────────────────────────────────
# _clean_response
# ──────────────────────────────────────────────

class TestCleanResponse:
    def test_removes_complete_think_tags(self, client):
        text = "<think>Some reasoning here.</think>The actual answer."
        result = client._clean_response(text)
        assert result == "The actual answer."

    def test_removes_unclosed_think_tag(self, client):
        text = "<think>Reasoning that got cut off because token limit"
        result = client._clean_response(text)
        assert result == ""

    def test_removes_multiline_think(self, client):
        text = "<think>\nStep 1: Consider X\nStep 2: Analyse Y\n</think>\nFinal answer here."
        result = client._clean_response(text)
        assert result == "Final answer here."

    def test_preserves_text_without_think(self, client):
        text = "A clean answer without any thinking tags."
        result = client._clean_response(text)
        assert result == text

    def test_multiple_think_blocks(self, client):
        text = "<think>First thought.</think>Part one. <think>Second thought.</think>Part two."
        result = client._clean_response(text)
        assert "First thought." not in result
        assert "Second thought." not in result
        assert "Part one." in result
        assert "Part two." in result

    def test_empty_think_tags(self, client):
        text = "<think></think>Answer."
        result = client._clean_response(text)
        assert result == "Answer."

    def test_strips_whitespace(self, client):
        text = "  <think>reasoning</think>  Answer with spaces.  "
        result = client._clean_response(text)
        assert result == "Answer with spaces."


# ──────────────────────────────────────────────
# generate — payload construction
# ──────────────────────────────────────────────

def _make_mock_response(text="Answer"):
    mock_response = MagicMock()
    mock_response.json.return_value = {"response": text}
    mock_response.raise_for_status = MagicMock()
    return mock_response


class TestGenerate:
    @pytest.mark.asyncio
    async def test_no_think_injected_in_system_only(self, client):
        client._mock_http.post = AsyncMock(return_value=_make_mock_response("Test answer"))

        await client.generate(prompt="What is RAG?", system="Be helpful.")
        call_args = client._mock_http.post.call_args
        payload = call_args.kwargs["json"]

        # prompt should NOT have /no_think (removed from prompt)
        assert not payload["prompt"].startswith("/no_think")
        # system SHOULD have /no_think
        assert payload["system"].startswith("/no_think")

    @pytest.mark.asyncio
    async def test_default_temperature(self, client):
        client._mock_http.post = AsyncMock(return_value=_make_mock_response())

        await client.generate(prompt="test")
        payload = client._mock_http.post.call_args.kwargs["json"]
        assert payload["options"]["temperature"] == 0.3

    @pytest.mark.asyncio
    async def test_custom_temperature(self, client):
        client._mock_http.post = AsyncMock(return_value=_make_mock_response())

        await client.generate(prompt="test", temperature=0.7)
        payload = client._mock_http.post.call_args.kwargs["json"]
        assert payload["options"]["temperature"] == 0.7

    @pytest.mark.asyncio
    async def test_num_predict_set(self, client):
        client._mock_http.post = AsyncMock(return_value=_make_mock_response())

        await client.generate(prompt="test")
        payload = client._mock_http.post.call_args.kwargs["json"]
        assert payload["options"]["num_predict"] == 2048

    @pytest.mark.asyncio
    async def test_stream_disabled(self, client):
        client._mock_http.post = AsyncMock(return_value=_make_mock_response())

        await client.generate(prompt="test")
        payload = client._mock_http.post.call_args.kwargs["json"]
        assert payload["stream"] is False

    @pytest.mark.asyncio
    async def test_system_prompt_omitted_when_empty(self, client):
        client._mock_http.post = AsyncMock(return_value=_make_mock_response())

        await client.generate(prompt="test", system="")
        payload = client._mock_http.post.call_args.kwargs["json"]
        assert "system" not in payload

    @pytest.mark.asyncio
    async def test_response_cleaned(self, client):
        client._mock_http.post = AsyncMock(
            return_value=_make_mock_response("<think>Reasoning.</think>Clean answer.")
        )

        result = await client.generate(prompt="test", system="Be helpful.")
        assert result == "Clean answer."

    @pytest.mark.asyncio
    async def test_model_name_in_payload(self, client):
        client._mock_http.post = AsyncMock(return_value=_make_mock_response())

        await client.generate(prompt="test")
        payload = client._mock_http.post.call_args.kwargs["json"]
        assert payload["model"] == "qwen3:4b"

    @pytest.mark.asyncio
    async def test_correct_endpoint(self, client):
        client._mock_http.post = AsyncMock(return_value=_make_mock_response())

        await client.generate(prompt="test")
        call_args = client._mock_http.post.call_args
        url = call_args.args[0]
        assert url.endswith("/api/generate")


# ──────────────────────────────────────────────
# _build_payload
# ──────────────────────────────────────────────

class TestBuildPayload:
    def test_stream_false(self, client):
        payload = client._build_payload("test", "system", 0.3, stream=False)
        assert payload["stream"] is False

    def test_stream_true(self, client):
        payload = client._build_payload("test", "system", 0.3, stream=True)
        assert payload["stream"] is True

    def test_no_think_in_system(self, client):
        payload = client._build_payload("test", "Be helpful.", 0.3, stream=False)
        assert payload["system"].startswith("/no_think")

    def test_no_system_when_empty(self, client):
        payload = client._build_payload("test", "", 0.3, stream=False)
        assert "system" not in payload


# ──────────────────────────────────────────────
# stream_generate
# ──────────────────────────────────────────────

class TestStreamGenerate:
    @pytest.mark.asyncio
    async def test_streams_clean_tokens(self, client):
        """stream_generate should yield tokens with <think> blocks filtered out."""
        import json

        # Simulate Ollama streaming response: each line is a JSON object
        lines = [
            json.dumps({"response": "Hello", "done": False}),
            json.dumps({"response": " world", "done": False}),
            json.dumps({"response": "", "done": True}),
        ]

        mock_response = AsyncMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        async def _aiter_lines():
            for line in lines:
                yield line

        mock_response.aiter_lines = _aiter_lines

        client._mock_http.stream = MagicMock(return_value=mock_response)

        tokens = []
        async for token in client.stream_generate(prompt="test", system="Be helpful."):
            tokens.append(token)

        full_text = "".join(tokens)
        assert "Hello" in full_text
        assert "world" in full_text

    @pytest.mark.asyncio
    async def test_filters_think_tags(self, client):
        """stream_generate should filter out <think>...</think> content."""
        import json

        lines = [
            json.dumps({"response": "<think>", "done": False}),
            json.dumps({"response": "reasoning here", "done": False}),
            json.dumps({"response": "</think>", "done": False}),
            json.dumps({"response": "Actual answer", "done": False}),
            json.dumps({"response": "", "done": True}),
        ]

        mock_response = AsyncMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        async def _aiter_lines():
            for line in lines:
                yield line

        mock_response.aiter_lines = _aiter_lines
        client._mock_http.stream = MagicMock(return_value=mock_response)

        tokens = []
        async for token in client.stream_generate(prompt="test", system="Be helpful."):
            tokens.append(token)

        full_text = "".join(tokens)
        assert "reasoning here" not in full_text
        assert "Actual answer" in full_text
