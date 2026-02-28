"""
RAG-Specialised Q&A Dataset Generator for QLoRA Fine-Tuning
- Type 1 (60%): Context-grounded answering with source attribution
- Type 2 (20%): Multi-chunk synthesis across papers
- Type 3 (20%): Refusal when context is insufficient
- Uses Qwen3 for generation with strict output format
"""

import json
import random
import re
import time
from pathlib import Path

import httpx

from src.api.core.config import settings, DATA_DIR


CHUNKS_FILE = DATA_DIR / "processed" / "chunks.json"
OUTPUT_FILE = DATA_DIR / "processed" / "qa_dataset.json"

# System prompt that mirrors actual RAG usage
RAG_SYSTEM_PROMPT = """You are a helpful academic research assistant. Answer questions based ONLY on the provided context from academic papers. Follow these rules strictly:
1. Only use information from the provided context
2. Cite which paper the information comes from
3. If the context does not contain enough information, say so clearly
4. Answer in concise prose paragraphs without markdown headers or bullet points
5. Do not generalise findings from one paper as universal recommendations"""

# === PROMPT TEMPLATES ===

TYPE1_PROMPT = """

Generate a question-answer pair from this academic paper excerpt.

Rules:
- The question should ask about a specific claim, method, or finding in the text
- The answer MUST cite the paper title and only use information from the text
- The answer should be 2-4 sentences in plain prose (no markdown headers, no bullet points)
- Do NOT make general recommendations — attribute findings to the specific paper

Paper title: {title}
Text:
{text}

Respond in this exact JSON format only:
{{"question": "...", "answer": "..."}}"""

TYPE2_PROMPT = """

Given excerpts from two different academic papers, generate a comparison question and answer.

Rules:
- The question should ask about comparing or contrasting the two approaches
- The answer MUST reference both papers by title
- The answer should synthesise information from both, in 3-5 sentences of plain prose
- No markdown headers or bullet points

Paper 1 - {title1}:
{text1}

Paper 2 - {title2}:
{text2}

Respond in this exact JSON format only:
{{"question": "...", "answer": "..."}}"""

TYPE3_PROMPT = """

Generate a question that CANNOT be answered from the given context, along with a proper refusal answer.

The question should be related to the paper's topic but ask about something NOT mentioned in the text.

Paper title: {title}
Topic area: {topic}
Text:
{text}

The answer must follow this pattern: "The provided context does not contain sufficient information to answer this question. The papers discuss [brief summary of what IS covered], but [specific gap] is not addressed."

Respond in this exact JSON format only:
{{"question": "...", "answer": "..."}}"""


def load_chunks():
    """Load and return chunks grouped by paper."""
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)["chunks"]

    papers = {}
    for c in chunks:
        aid = c["arxiv_id"]
        if aid not in papers:
            papers[aid] = {"title": c["title"], "chunks": []}
        papers[aid]["chunks"].append(c)

    return chunks, papers


def call_llm(prompt: str, max_retries: int = 2) -> str | None:
    """Call Qwen3 with format=json. Extracts JSON from thinking or response field."""
    for attempt in range(max_retries):
        try:
            response = httpx.post(
                f"http://{settings.OLLAMA_HOST}/api/generate",
                json={
                    "model": settings.LLM_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                    "options": {"temperature": 0.7, "num_predict": 4096},
                },
                timeout=120.0,
            )
            response.raise_for_status()
            data = response.json()
            raw = data.get("response", "").strip()
            thinking = data.get("thinking", "").strip()
            return raw if raw else thinking
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(2)
    return None

def parse_json_response(raw: str) -> dict | None:
    """Extract JSON object from LLM response."""
    if not raw:
        return None
    try:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            obj = json.loads(raw[start:end])
            if "question" in obj and "answer" in obj:
                if len(obj["question"]) > 10 and len(obj["answer"]) > 20:
                    return obj
    except json.JSONDecodeError:
        pass
    return None


def generate_type1(chunk: dict) -> dict | None:
    """Generate context-grounded Q&A from a single chunk."""
    prompt = TYPE1_PROMPT.format(
        title=chunk["title"],
        text=chunk["text"][:500],
    )
    raw = call_llm(prompt)
    result = parse_json_response(raw)
    if result:
        context = f"Context from '{chunk['title']}' ({chunk['section']}):\n{chunk['text'][:500]}"
        return {
            "type": "grounded",
            "instruction": RAG_SYSTEM_PROMPT,
            "input": f"{context}\n\nQuestion: {result['question']}",
            "output": result["answer"],
            "source_arxiv_id": chunk["arxiv_id"],
        }
    return None


def generate_type2(chunk1: dict, chunk2: dict) -> dict | None:
    """Generate multi-paper synthesis Q&A."""
    prompt = TYPE2_PROMPT.format(
        title1=chunk1["title"],
        text1=chunk1["text"][:400],
        title2=chunk2["title"],
        text2=chunk2["text"][:400],
    )
    raw = call_llm(prompt)
    result = parse_json_response(raw)
    if result:
        context = (
            f"Context from '{chunk1['title']}' ({chunk1['section']}):\n{chunk1['text'][:400]}\n\n"
            f"Context from '{chunk2['title']}' ({chunk2['section']}):\n{chunk2['text'][:400]}"
        )
        return {
            "type": "synthesis",
            "instruction": RAG_SYSTEM_PROMPT,
            "input": f"{context}\n\nQuestion: {result['question']}",
            "output": result["answer"],
            "source_arxiv_id": f"{chunk1['arxiv_id']},{chunk2['arxiv_id']}",
        }
    return None


def generate_type3(chunk: dict) -> dict | None:
    """Generate refusal Q&A where context is insufficient."""
    title_lower = chunk["title"].lower()
    topic_map = {
        "rag": "retrieval-augmented generation",
        "lora": "parameter-efficient fine-tuning",
        "qlora": "quantised fine-tuning",
        "hallucin": "LLM hallucination",
        "prompt": "prompt engineering",
        "instruct": "instruction tuning",
    }
    topic = "machine learning"
    for key, val in topic_map.items():
        if key in title_lower:
            topic = val
            break

    prompt = TYPE3_PROMPT.format(
        title=chunk["title"],
        topic=topic,
        text=chunk["text"][:500],
    )
    raw = call_llm(prompt)
    result = parse_json_response(raw)
    if result:
        context = f"Context from '{chunk['title']}' ({chunk['section']}):\n{chunk['text'][:500]}"
        return {
            "type": "refusal",
            "instruction": RAG_SYSTEM_PROMPT,
            "input": f"{context}\n\nQuestion: {result['question']}",
            "output": result["answer"],
            "source_arxiv_id": chunk["arxiv_id"],
        }
    return None


def main():
    print("Loading chunks...")
    chunks, papers = load_chunks()
    print(f"Total: {len(chunks)} chunks from {len(papers)} papers")

    good_chunks = [c for c in chunks if c["word_count"] >= 80]
    random.seed(42)
    random.shuffle(good_chunks)

    # Calculate target counts
    total_target = min(len(good_chunks), 2000)
    n_type1 = int(total_target * 0.6)
    n_type2 = int(total_target * 0.2)
    n_type3 = total_target - n_type1 - n_type2

    type1_chunks = good_chunks[:n_type1]
    type3_chunks = good_chunks[n_type1:n_type1 + n_type3]

    # For type 2, pair chunks from different papers
    type2_pairs = []
    chunk_list = good_chunks[n_type1 + n_type3:]
    for i in range(0, min(len(chunk_list) - 1, n_type2 * 2), 2):
        if chunk_list[i]["arxiv_id"] != chunk_list[i + 1]["arxiv_id"]:
            type2_pairs.append((chunk_list[i], chunk_list[i + 1]))
        if len(type2_pairs) >= n_type2:
            break

    print(f"\nTargets: Type1={n_type1}, Type2={len(type2_pairs)}, Type3={n_type3}")
    print(f"Estimated time: {(n_type1 + len(type2_pairs) + n_type3) * 15 / 3600:.1f} hours\n")

    dataset = []
    failed = 0
    start_time = time.time()

    # Type 1: Context-grounded
    print("--- Generating Type 1: Context-grounded Q&A ---")
    for i, chunk in enumerate(type1_chunks):
        result = generate_type1(chunk)
        if result:
            dataset.append(result)
        else:
            failed += 1

        if (i + 1) % 20 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed * 60
            print(f"  Type1 [{i+1}/{n_type1}] total={len(dataset)} ({rate:.0f}/min, {failed} failed)")

    # Type 2: Multi-paper synthesis
    print("\n--- Generating Type 2: Multi-paper synthesis ---")
    for i, (c1, c2) in enumerate(type2_pairs):
        result = generate_type2(c1, c2)
        if result:
            dataset.append(result)
        else:
            failed += 1

        if (i + 1) % 10 == 0:
            print(f"  Type2 [{i+1}/{len(type2_pairs)}] total={len(dataset)}")

    # Type 3: Refusal
    print("\n--- Generating Type 3: Refusal Q&A ---")
    for i, chunk in enumerate(type3_chunks):
        result = generate_type3(chunk)
        if result:
            dataset.append(result)
        else:
            failed += 1

        if (i + 1) % 20 == 0:
            print(f"  Type3 [{i+1}/{n_type3}] total={len(dataset)}")

    # Shuffle and save
    random.shuffle(dataset)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "data": dataset,
            "total": len(dataset),
            "stats": {
                "type1_grounded": sum(1 for d in dataset if d["type"] == "grounded"),
                "type2_synthesis": sum(1 for d in dataset if d["type"] == "synthesis"),
                "type3_refusal": sum(1 for d in dataset if d["type"] == "refusal"),
            },
        }, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - start_time
    print(f"\n{'='*50}")
    print(f"Done in {elapsed/60:.1f} min ({elapsed/3600:.1f} hours)")
    print(f"  Total Q&A pairs: {len(dataset)}")
    print(f"  - Grounded:  {sum(1 for d in dataset if d['type'] == 'grounded')}")
    print(f"  - Synthesis: {sum(1 for d in dataset if d['type'] == 'synthesis')}")
    print(f"  - Refusal:   {sum(1 for d in dataset if d['type'] == 'refusal')}")
    print(f"  Failed: {failed}")
    print(f"  Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
