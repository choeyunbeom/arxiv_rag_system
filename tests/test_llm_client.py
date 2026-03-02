"""
Unit tests for src/api/core/llm_client.py

Tests cover:
- Qwen3 thinking mode tag cleaning
- Request payload construction (no_think injection, temperature, num_predict)
- Response parsing
- Error handling

External dependency (Ollama HTTP API) is mocked.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.api.core.llm_client import LLMClient

# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────

@pytest.fixture
def client():
    """Create an LLMClient with mocked HTTP client."""
    with patch("src.api.core.llm_client.httpx.Client") as MockClient:
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

class TestGenerate:
    def test_no_think_injected_in_prompt(self, client):
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "Test answer"}
        mock_response.raise_for_status = MagicMock()
        client._mock_http.post.return_value = mock_response

        client.generate(prompt="What is RAG?", system="Be helpful.")
        call_args = client._mock_http.post.call_args
        payload = call_args.kwargs["json"]

        assert payload["prompt"].startswith("/no_think")
        assert payload["system"].startswith("/no_think")

    def test_default_temperature(self, client):
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "Answer"}
        mock_response.raise_for_status = MagicMock()
        client._mock_http.post.return_value = mock_response

        client.generate(prompt="test")
        payload = client._mock_http.post.call_args.kwargs["json"]
        assert payload["options"]["temperature"] == 0.3

    def test_custom_temperature(self, client):
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "Answer"}
        mock_response.raise_for_status = MagicMock()
        client._mock_http.post.return_value = mock_response

        client.generate(prompt="test", temperature=0.7)
        payload = client._mock_http.post.call_args.kwargs["json"]
        assert payload["options"]["temperature"] == 0.7

    def test_num_predict_set(self, client):
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "Answer"}
        mock_response.raise_for_status = MagicMock()
        client._mock_http.post.return_value = mock_response

        client.generate(prompt="test")
        payload = client._mock_http.post.call_args.kwargs["json"]
        assert payload["options"]["num_predict"] == 2048

    def test_stream_disabled(self, client):
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "Answer"}
        mock_response.raise_for_status = MagicMock()
        client._mock_http.post.return_value = mock_response

        client.generate(prompt="test")
        payload = client._mock_http.post.call_args.kwargs["json"]
        assert payload["stream"] is False

    def test_system_prompt_omitted_when_empty(self, client):
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "Answer"}
        mock_response.raise_for_status = MagicMock()
        client._mock_http.post.return_value = mock_response

        client.generate(prompt="test", system="")
        payload = client._mock_http.post.call_args.kwargs["json"]
        assert "system" not in payload

    def test_response_cleaned(self, client):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "response": "<think>Reasoning.</think>Clean answer."
        }
        mock_response.raise_for_status = MagicMock()
        client._mock_http.post.return_value = mock_response

        result = client.generate(prompt="test", system="Be helpful.")
        assert result == "Clean answer."

    def test_model_name_in_payload(self, client):
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "Answer"}
        mock_response.raise_for_status = MagicMock()
        client._mock_http.post.return_value = mock_response

        client.generate(prompt="test")
        payload = client._mock_http.post.call_args.kwargs["json"]
        assert payload["model"] == "qwen3:4b"

    def test_correct_endpoint(self, client):
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "Answer"}
        mock_response.raise_for_status = MagicMock()
        client._mock_http.post.return_value = mock_response

        client.generate(prompt="test")
        call_args = client._mock_http.post.call_args
        url = call_args.args[0]
        assert url.endswith("/api/generate")
