"""
RAG Chain
- Combines retriever and LLM into a question-answering pipeline
- Formats retrieved context into a prompt
- Returns answer with source citations
- Accepts external system_prompt / query_template for experiment injection
"""

from dataclasses import dataclass

from src.api.core.hybrid_retriever import HybridRetriever as Retriever, RetrievedChunk
from src.api.core.llm_client import LLMClient
from src.api.core.prompts import SYSTEM_PROMPT_ZERO_SHOT, QUERY_TEMPLATE_DEFAULT


@dataclass
class Source:
    title: str
    arxiv_id: str
    section: str
    authors: str
    distance: float


@dataclass
class RAGResponse:
    answer: str
    sources: list[Source]
    query: str


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

    def query(self, question: str, top_k: int = 5) -> RAGResponse:
        """Run the full RAG pipeline: retrieve -> format -> generate."""
        # 1. Retrieve relevant chunks
        chunks = self.retriever.search(question, top_k=top_k)

        # 2. Format context
        context = self._format_context(chunks)

        # 3. Build prompt
        prompt = self.query_template.format(context=context, question=question)

        # 4. Generate answer
        answer = self.llm.generate(prompt=prompt, system=self.system_prompt)

        # 5. Collect sources
        sources = self._deduplicate_sources(chunks)

        return RAGResponse(
            answer=answer,
            sources=sources,
            query=question,
        )
