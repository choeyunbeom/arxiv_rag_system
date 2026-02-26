"""
Evaluation Pipeline
- Measures RAG system performance across multiple metrics
- Retrieval metrics: Hit Rate, MRR, Source Precision
- Answer metrics: Keyword Coverage, Faithfulness (basic)
- Outputs a JSON report for baseline vs fine-tuned comparison
"""

import json
import time
from datetime import datetime
from pathlib import Path

from src.api.core.rag_chain import RAGChain
from src.api.core.retriever import Retriever
from src.evaluation.eval_dataset import EVAL_DATASET


RESULTS_DIR = Path("data/processed")


def evaluate_retrieval(retriever: Retriever, dataset: list[dict], top_k: int = 5) -> dict:
    """Evaluate retrieval quality."""
    hits = 0
    reciprocal_ranks = []
    precision_scores = []

    for item in dataset:
        results = retriever.search(item["question"], top_k=top_k)
        retrieved_ids = [r.arxiv_id for r in results]

        # Hit Rate: is any expected paper in top-k?
        hit = any(eid in retrieved_ids for eid in item["expected_arxiv_ids"])
        hits += int(hit)

        # MRR: reciprocal rank of first relevant result
        rr = 0.0
        for rank, rid in enumerate(retrieved_ids, 1):
            if rid in item["expected_arxiv_ids"]:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)

        # Source Precision: how many retrieved are relevant?
        relevant_count = sum(1 for rid in retrieved_ids if rid in item["expected_arxiv_ids"])
        precision_scores.append(relevant_count / len(retrieved_ids))

    n = len(dataset)
    return {
        "hit_rate": hits / n,
        "mrr": sum(reciprocal_ranks) / n,
        "avg_precision": sum(precision_scores) / n,
        "total_questions": n,
        "top_k": top_k,
    }


def evaluate_answers(rag_chain: RAGChain, dataset: list[dict], top_k: int = 5) -> dict:
    """Evaluate answer quality."""
    results = []
    total_time = 0

    for i, item in enumerate(dataset):
        print(f"  [{i+1}/{len(dataset)}] {item['question'][:50]}...")

        start = time.time()
        response = rag_chain.query(item["question"], top_k=top_k)
        elapsed = time.time() - start
        total_time += elapsed

        answer_lower = response.answer.lower()

        # Keyword Coverage: what fraction of expected keywords appear in answer?
        keyword_hits = sum(
            1 for kw in item["expected_keywords"]
            if kw.lower() in answer_lower
        )
        keyword_coverage = keyword_hits / len(item["expected_keywords"])

        # Answer Length
        word_count = len(response.answer.split())

        # Source Hit: did the answer use the expected papers?
        source_ids = [s.arxiv_id for s in response.sources]
        source_hit = any(eid in source_ids for eid in item["expected_arxiv_ids"])

        # Faithfulness (basic): does the answer avoid "I don't know" / empty responses?
        is_substantive = word_count > 20 and "i don't" not in answer_lower

        results.append({
            "question": item["question"],
            "topic": item["topic"],
            "answer": response.answer,
            "answer_word_count": word_count,
            "keyword_coverage": keyword_coverage,
            "keywords_found": keyword_hits,
            "keywords_total": len(item["expected_keywords"]),
            "source_hit": source_hit,
            "source_ids": source_ids,
            "expected_ids": item["expected_arxiv_ids"],
            "is_substantive": is_substantive,
            "latency_seconds": round(elapsed, 2),
        })

    n = len(results)
    return {
        "summary": {
            "avg_keyword_coverage": sum(r["keyword_coverage"] for r in results) / n,
            "source_hit_rate": sum(r["source_hit"] for r in results) / n,
            "substantive_rate": sum(r["is_substantive"] for r in results) / n,
            "avg_word_count": sum(r["answer_word_count"] for r in results) / n,
            "avg_latency": total_time / n,
            "total_time": round(total_time, 2),
            "total_questions": n,
        },
        "details": results,
    }


def run_evaluation(top_k: int = 5, label: str = "baseline"):
    """Run full evaluation and save report."""
    print(f"\n{'='*60}")
    print(f"  Running evaluation: {label}")
    print(f"  Dataset: {len(EVAL_DATASET)} questions, top_k={top_k}")
    print(f"{'='*60}\n")

    retriever = Retriever()
    rag_chain = RAGChain()

    # 1. Retrieval evaluation
    print("Evaluating retrieval...")
    retrieval_metrics = evaluate_retrieval(retriever, EVAL_DATASET, top_k=top_k)
    print(f"  Hit Rate:  {retrieval_metrics['hit_rate']:.2%}")
    print(f"  MRR:       {retrieval_metrics['mrr']:.4f}")
    print(f"  Precision: {retrieval_metrics['avg_precision']:.2%}")

    # 2. Answer evaluation
    print("\nEvaluating answers...")
    answer_metrics = evaluate_answers(rag_chain, EVAL_DATASET, top_k=top_k)
    summary = answer_metrics["summary"]
    print(f"\n  Keyword Coverage:  {summary['avg_keyword_coverage']:.2%}")
    print(f"  Source Hit Rate:   {summary['source_hit_rate']:.2%}")
    print(f"  Substantive Rate:  {summary['substantive_rate']:.2%}")
    print(f"  Avg Word Count:    {summary['avg_word_count']:.0f}")
    print(f"  Avg Latency:       {summary['avg_latency']:.1f}s")
    print(f"  Total Time:        {summary['total_time']:.1f}s")

    # 3. Save report
    report = {
        "label": label,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "top_k": top_k,
            "dataset_size": len(EVAL_DATASET),
        },
        "retrieval": retrieval_metrics,
        "answer": answer_metrics,
    }

    output_path = RESULTS_DIR / f"eval_{label}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n  Report saved to {output_path}")
    return report


if __name__ == "__main__":
    run_evaluation(top_k=5, label="baseline")
