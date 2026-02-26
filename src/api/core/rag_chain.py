"""
RAG Chain
- Combines retriever and LLM into a question-answering pipeline
- Formats retrieved context into a prompt
- Returns answer with source citations
"""

from dataclasses import dataclass

from src.api.core.retriever import Retriever, RetrievedChunk
from src.api.core.llm_client import LLMClient


SYSTEM_PROMPT = """You are a helpful research assistant specialising in AI and machine learning.
Answer questions based ONLY on the provided context from academic papers.
If the context does not contain enough information to answer, say so honestly.
Always cite the paper title when referencing specific information.
Be concise and precise."""

QUERY_TEMPLATE = """Context from relevant papers:

{context}

---

Question: {question}

Answer based on the context above. Cite paper titles where appropriate."""


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
    def __init__(self):
        self.retriever = Retriever()
        self.llm = LLMClient()

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
        """Deduplicate sources by arxiv_id, keep best distance."""
        seen = {}
        for chunk in chunks:
            if chunk.arxiv_id not in seen or chunk.distance < seen[chunk.arxiv_id].distance:
                seen[chunk.arxiv_id] = Source(
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
        prompt = QUERY_TEMPLATE.format(context=context, question=question)

        # 4. Generate answer
        answer = self.llm.generate(prompt=prompt, system=SYSTEM_PROMPT)

        # 5. Collect sources
        sources = self._deduplicate_sources(chunks)

        return RAGResponse(
            answer=answer,
            sources=sources,
            query=question,
        )
