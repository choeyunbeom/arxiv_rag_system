"""
RAG Chain
- Combines retriever and LLM into a question-answering pipeline
- Formats retrieved context into a prompt
- Returns answer with source citations and latency breakdown
- Accepts external system_prompt / query_template for experiment injection
"""

import asyncio
import json
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from src.api.core.hybrid_retriever import HybridRetriever as Retriever
from src.api.core.hybrid_retriever import RetrievedChunk
from src.api.core.llm_client import LLMClient
from src.api.core.prompts import QUERY_TEMPLATE_DEFAULT, SYSTEM_PROMPT_ZERO_SHOT


@dataclass
class Source:
    title: str
    arxiv_id: str
    section: str
    authors: str
    distance: float


@dataclass
class LatencyInfo:
    retrieval_ms: float = 0.0
    generation_ms: float = 0.0
    total_ms: float = 0.0


@dataclass
class RAGResponse:
    answer: str
    sources: list[Source]
    query: str
    latency: LatencyInfo = field(default_factory=LatencyInfo)
    vis_data: Any | None = None


class RAGChain:
    def __init__(
        self,
        system_prompt: str | None = None,
        query_template: str | None = None,
    ):
        self.retriever = Retriever()
        self.llm = LLMClient()
        self.system_prompt = system_prompt or SYSTEM_PROMPT_ZERO_SHOT
        self.query_template = query_template or QUERY_TEMPLATE_DEFAULT

    def _format_context(self, chunks: list[RetrievedChunk]) -> str:
        """Format retrieved chunks into a context string."""
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            part = f"[{i}] Paper: {chunk.title}\n"
            part += f"    Section: {chunk.section}\n"
            part += f"    Content: {chunk.text}\n"
            context_parts.append(part)
        return "\n".join(context_parts)

    def _deduplicate_sources(self, chunks: list[RetrievedChunk]) -> list[Source]:
        """Deduplicate sources by arxiv_id::section, consistent with retriever dedup."""
        seen = {}
        for chunk in chunks:
            key = f"{chunk.arxiv_id}::{chunk.section}"
            if key not in seen:
                seen[key] = Source(
                    title=chunk.title,
                    arxiv_id=chunk.arxiv_id,
                    section=chunk.section,
                    authors=chunk.authors,
                    distance=chunk.distance,
                )
        return list(seen.values())

    async def query(self, question: str, top_k: int = 5, include_vis: bool = False) -> RAGResponse:
        """Run the full RAG pipeline: retrieve -> format -> generate."""
        total_start = time.perf_counter()

        # 1. Retrieve relevant chunks
        retrieval_start = time.perf_counter()
        chunks, query_emb = await self.retriever.search(question, top_k=top_k, get_embeddings=include_vis)
        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000

        # 2. Format context
        context = self._format_context(chunks)

        # 3. Build prompt
        prompt = self.query_template.format(context=context, question=question)

        # 4. Generate answer
        generation_start = time.perf_counter()
        answer = await self.llm.generate(prompt=prompt, system=self.system_prompt)
        generation_ms = (time.perf_counter() - generation_start) * 1000

        # 5. Collect sources
        sources = self._deduplicate_sources(chunks)

        total_ms = (time.perf_counter() - total_start) * 1000

        # 6. Build UMAP Visualization Data if requested
        vis_data = None
        if include_vis and self.retriever.umap_reducer and self.retriever.umap_bg_data and query_emb:
            import numpy as np

            from src.api.models.schemas import VisData, VisPoint

            # transform query (CPU-intensive, offload to threadpool)
            query_3d = (await asyncio.to_thread(self.retriever.umap_reducer.transform, np.array([query_emb])))[0]

            points = []

            # Add background points
            source_cids = {c.chunk_id for c in chunks}
            for bg in self.retriever.umap_bg_data:
                is_source = bg["chunk_id"] in source_cids
                points.append(
                    VisPoint(
                        chunk_id=bg["chunk_id"],
                        x=bg["x"],
                        y=bg["y"],
                        z=bg.get("z", 0.0),
                        title=bg["title"],
                        section=bg["section"],
                        text_preview=bg["text_preview"],
                        is_source=is_source,
                        is_query=False
                    )
                )

            # Add Query point
            points.append(
                VisPoint(
                    chunk_id=None,
                    x=float(query_3d[0]),
                    y=float(query_3d[1]),
                    z=float(query_3d[2]),
                    title="User Query",
                    section=None,
                    text_preview=question,
                    is_source=False,
                    is_query=True
                )
            )

            vis_data = VisData(points=points)

        return RAGResponse(
            answer=answer,
            sources=sources,
            query=question,
            latency=LatencyInfo(
                retrieval_ms=round(retrieval_ms, 1),
                generation_ms=round(generation_ms, 1),
                total_ms=round(total_ms, 1),
            ),
            vis_data=vis_data
        )

    async def stream_query(self, question: str, top_k: int = 5) -> AsyncIterator[str]:
        """Stream the RAG pipeline as Server-Sent Events (SSE).

        Yields newline-delimited JSON events:
          {"event": "sources", "data": [...]}      — retrieved sources
          {"event": "token", "data": "..."}        — each answer token
          {"event": "done", "data": {"latency": ...}} — final latency info
          {"event": "error", "data": "..."}        — on failure
        """
        total_start = time.perf_counter()

        # 1. Retrieve (non-streaming — must complete before generation)
        retrieval_start = time.perf_counter()
        chunks, _ = await self.retriever.search(question, top_k=top_k, get_embeddings=False)
        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000

        # 2. Emit sources immediately so the client can render them while LLM generates
        sources = self._deduplicate_sources(chunks)
        sources_payload = [
            {
                "title": s.title,
                "arxiv_id": s.arxiv_id,
                "section": s.section,
                "authors": s.authors,
                "distance": s.distance,
            }
            for s in sources
        ]
        yield json.dumps({"event": "sources", "data": sources_payload}) + "\n"

        # 3. Stream LLM generation
        context = self._format_context(chunks)
        prompt = self.query_template.format(context=context, question=question)

        generation_start = time.perf_counter()
        async for token in self.llm.stream_generate(prompt=prompt, system=self.system_prompt):
            yield json.dumps({"event": "token", "data": token}) + "\n"
        generation_ms = (time.perf_counter() - generation_start) * 1000

        total_ms = (time.perf_counter() - total_start) * 1000

        # 4. Final event with latency
        yield json.dumps({
            "event": "done",
            "data": {
                "latency": {
                    "retrieval_ms": round(retrieval_ms, 1),
                    "generation_ms": round(generation_ms, 1),
                    "total_ms": round(total_ms, 1),
                }
            },
        }) + "\n"
