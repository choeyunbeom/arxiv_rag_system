"""
Integration tests for FastAPI API endpoints.

Tests the full HTTP request/response cycle through FastAPI's TestClient,
with external services (Ollama, ChromaDB) mocked at the boundary.

Covers:
- POST /query — success, validation errors, service errors
- GET /health — healthy, degraded states
- GET / — root info endpoint
"""

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Mock heavy dependencies before any src imports
sys.modules["chromadb"] = MagicMock()
sys.modules["rank_bm25"] = MagicMock()
sys.modules["sentence_transformers"] = MagicMock()

import httpx
from fastapi.testclient import TestClient

from src.api.core.rag_chain import LatencyInfo, RAGChain, RAGResponse, Source

# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────

@pytest.fixture
def mock_rag_chain():
    """Create a mock RAGChain that returns predictable responses."""
    chain = MagicMock(spec=RAGChain)
    from src.api.models.schemas import VisData, VisPoint
    rag_response = RAGResponse(
        answer="RAG combines retrieval with generation to produce grounded answers.",
        sources=[
            Source(
                title="A Survey on RAG",
                arxiv_id="2401.00001",
                section="Introduction",
                authors="Author A",
                distance=0.15,
            ),
            Source(
                title="Engineering the RAG Stack",
                arxiv_id="2401.00002",
                section="Method",
                authors="Author B",
                distance=0.22,
            ),
        ],
        query="What is RAG?",
        latency=LatencyInfo(
            retrieval_ms=1340.5,
            generation_ms=18200.3,
            total_ms=19540.8,
        ),
    )
    rag_response.vis_data = VisData(points=[VisPoint(x=0.1, y=0.5, z=0.5, is_query=True)])
    chain.query = AsyncMock(return_value=rag_response)
    return chain


@pytest.fixture
def client(mock_rag_chain):
    """Create a TestClient with mocked RAGChain (overriding lifespan)."""
    from contextlib import asynccontextmanager

    from src.api.main import app

    # Override lifespan to inject our mock instead of creating a real RAGChain
    @asynccontextmanager
    async def mock_lifespan(app):
        app.state.rag_chain = mock_rag_chain
        yield

    original_lifespan = app.router.lifespan_context
    app.router.lifespan_context = mock_lifespan

    with TestClient(app, raise_server_exceptions=False) as tc:
        yield tc

    app.router.lifespan_context = original_lifespan


# ──────────────────────────────────────────────
# GET / — Root endpoint
# ──────────────────────────────────────────────

class TestRootEndpoint:
    def test_returns_info(self, client):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "arXiv RAG System"
        assert "version" in data
        assert data["docs"] == "/docs"


# ──────────────────────────────────────────────
# POST /query — Query endpoint
# ──────────────────────────────────────────────

class TestQueryEndpoint:
    def test_successful_query(self, client, mock_rag_chain):
        response = client.post("/query", json={
            "question": "What is RAG?",
            "top_k": 5,
        })
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert data["query"] == "What is RAG?"
        assert len(data["sources"]) == 2
        assert data["sources"][0]["title"] == "A Survey on RAG"
        assert data["sources"][0]["arxiv_id"] == "2401.00001"

    def test_default_top_k(self, client, mock_rag_chain):
        response = client.post("/query", json={"question": "What is QLoRA?"})
        assert response.status_code == 200
        mock_rag_chain.query.assert_called_with(question="What is QLoRA?", top_k=5, include_vis=False)

    def test_custom_top_k(self, client, mock_rag_chain):
        response = client.post("/query", json={"question": "What is QLoRA?", "top_k": 10})
        assert response.status_code == 200
        mock_rag_chain.query.assert_called_with(question="What is QLoRA?", top_k=10, include_vis=False)

    def test_empty_question_rejected(self, client):
        response = client.post("/query", json={"question": ""})
        assert response.status_code == 422

    def test_short_question_rejected(self, client):
        response = client.post("/query", json={"question": "Hi"})
        assert response.status_code == 422

    def test_missing_question_rejected(self, client):
        response = client.post("/query", json={})
        assert response.status_code == 422

    def test_top_k_too_high_rejected(self, client):
        response = client.post("/query", json={"question": "What is RAG?", "top_k": 100})
        assert response.status_code == 422

    def test_top_k_zero_rejected(self, client):
        response = client.post("/query", json={"question": "What is RAG?", "top_k": 0})
        assert response.status_code == 422

    def test_ollama_unavailable_returns_503(self, client, mock_rag_chain):
        mock_rag_chain.query.side_effect = httpx.ConnectError("Connection refused")
        response = client.post("/query", json={"question": "What is RAG?"})
        assert response.status_code == 503
        assert "unavailable" in response.json()["detail"].lower()

    def test_llm_timeout_returns_504(self, client, mock_rag_chain):
        mock_rag_chain.query.side_effect = httpx.TimeoutException("Timed out")
        response = client.post("/query", json={"question": "What is RAG?"})
        assert response.status_code == 504
        assert "timed out" in response.json()["detail"].lower()

    def test_value_error_returns_400(self, client, mock_rag_chain):
        mock_rag_chain.query.side_effect = ValueError("Invalid input")
        response = client.post("/query", json={"question": "What is RAG?"})
        assert response.status_code == 400

    def test_unexpected_error_returns_500(self, client, mock_rag_chain):
        mock_rag_chain.query.side_effect = RuntimeError("Something broke")
        response = client.post("/query", json={"question": "What is RAG?"})
        assert response.status_code == 500

    def test_response_schema_matches(self, client):
        response = client.post("/query", json={"question": "What is RAG?"})
        data = response.json()
        # Verify all required fields exist with correct types
        assert isinstance(data["answer"], str)
        assert isinstance(data["sources"], list)
        assert isinstance(data["query"], str)
        for source in data["sources"]:
            assert "title" in source
            assert "arxiv_id" in source
            assert "section" in source
            assert "authors" in source
            assert isinstance(source["distance"], float)
        # Verify latency breakdown
        assert "latency" in data
        latency = data["latency"]
        assert isinstance(latency["retrieval_ms"], float)
        assert isinstance(latency["generation_ms"], float)
        assert isinstance(latency["total_ms"], float)
        assert latency["retrieval_ms"] > 0
        assert latency["generation_ms"] > 0
        assert latency["total_ms"] >= latency["retrieval_ms"]


# ──────────────────────────────────────────────
# GET /health — Health endpoint
# ──────────────────────────────────────────────

class TestHealthEndpoint:
    @patch("src.api.routers.health.chromadb")
    @patch("src.api.routers.health.httpx.AsyncClient")
    def test_all_healthy(self, mock_async_client_cls, mock_chromadb, client):
        # Mock Ollama healthy via AsyncClient context manager
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_client_instance = AsyncMock()
        mock_client_instance.get = AsyncMock(return_value=mock_response)
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=False)
        mock_async_client_cls.return_value = mock_client_instance

        # Mock ChromaDB healthy
        mock_collection = MagicMock()
        mock_collection.count.return_value = 3500
        mock_chromadb.HttpClient.return_value.get_collection.return_value = mock_collection

        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["ollama"] is True
        assert data["chromadb"] is True
        assert data["collection_count"] == 3500

    @patch("src.api.routers.health.chromadb")
    @patch("src.api.routers.health.httpx.AsyncClient")
    def test_ollama_down_degraded(self, mock_async_client_cls, mock_chromadb, client):
        mock_client_instance = AsyncMock()
        mock_client_instance.get = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=False)
        mock_async_client_cls.return_value = mock_client_instance

        mock_collection = MagicMock()
        mock_collection.count.return_value = 3500
        mock_chromadb.HttpClient.return_value.get_collection.return_value = mock_collection

        response = client.get("/health")
        data = response.json()
        assert data["status"] == "degraded"
        assert data["ollama"] is False
        assert data["chromadb"] is True

    @patch("src.api.routers.health.chromadb")
    @patch("src.api.routers.health.httpx.AsyncClient")
    def test_chromadb_down_degraded(self, mock_async_client_cls, mock_chromadb, client):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_client_instance = AsyncMock()
        mock_client_instance.get = AsyncMock(return_value=mock_response)
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=False)
        mock_async_client_cls.return_value = mock_client_instance

        mock_chromadb.HttpClient.return_value.get_collection.side_effect = Exception("Connection refused")

        response = client.get("/health")
        data = response.json()
        assert data["status"] == "degraded"
        assert data["ollama"] is True
        assert data["chromadb"] is False
        assert data["collection_count"] == 0

    @patch("src.api.routers.health.chromadb")
    @patch("src.api.routers.health.httpx.AsyncClient")
    def test_both_down_degraded(self, mock_async_client_cls, mock_chromadb, client):
        mock_client_instance = AsyncMock()
        mock_client_instance.get = AsyncMock(side_effect=httpx.ConnectError("refused"))
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=False)
        mock_async_client_cls.return_value = mock_client_instance

        mock_chromadb.HttpClient.return_value.get_collection.side_effect = Exception("refused")

        response = client.get("/health")
        data = response.json()
        assert data["status"] == "degraded"
        assert data["ollama"] is False
        assert data["chromadb"] is False

    @patch("src.api.routers.health.chromadb")
    @patch("src.api.routers.health.httpx.AsyncClient")
    def test_health_response_schema(self, mock_async_client_cls, mock_chromadb, client):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_client_instance = AsyncMock()
        mock_client_instance.get = AsyncMock(return_value=mock_response)
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=False)
        mock_async_client_cls.return_value = mock_client_instance
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        mock_chromadb.HttpClient.return_value.get_collection.return_value = mock_collection

        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "ollama" in data
        assert "chromadb" in data
        assert "collection_count" in data
        assert isinstance(data["collection_count"], int)


# ──────────────────────────────────────────────
# POST /query/stream — Streaming endpoint
# ──────────────────────────────────────────────

class TestQueryStreamEndpoint:
    def test_stream_returns_ndjson(self, client, mock_rag_chain):
        import json

        # Mock stream_query to yield NDJSON lines
        async def _mock_stream(question, top_k):
            yield json.dumps({"event": "sources", "data": [{"title": "Paper A", "arxiv_id": "2401.00001", "section": "Intro", "authors": "Auth A", "distance": 0.1}]}) + "\n"
            yield json.dumps({"event": "token", "data": "Hello"}) + "\n"
            yield json.dumps({"event": "token", "data": " world"}) + "\n"
            yield json.dumps({"event": "done", "data": {"latency": {"retrieval_ms": 100.0, "generation_ms": 200.0, "total_ms": 300.0}}}) + "\n"

        mock_rag_chain.stream_query = _mock_stream

        response = client.post("/query/stream", json={"question": "What is RAG?"})
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("application/x-ndjson")

        lines = [line for line in response.text.strip().split("\n") if line]
        events = [json.loads(line) for line in lines]

        assert events[0]["event"] == "sources"
        assert events[1]["event"] == "token"
        assert events[1]["data"] == "Hello"
        assert events[-1]["event"] == "done"
        assert "latency" in events[-1]["data"]

    def test_stream_rag_not_initialized(self, client, mock_rag_chain):
        # Temporarily set rag_chain to None
        original = client.app.state.rag_chain
        client.app.state.rag_chain = None

        response = client.post("/query/stream", json={"question": "What is RAG?"})
        assert response.status_code == 503

        client.app.state.rag_chain = original
