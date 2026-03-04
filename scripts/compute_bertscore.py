"""
BERTScore Post-Hoc Evaluation
- Computes BERTScore (precision, recall, F1) for existing eval results
- Uses reference answers from eval_dataset.py
- Model: distilbert-base-uncased (fast, avoids tokenizer overflow on long answers)
- Answers truncated to 400 words to stay within model token limits

Usage:
    python -m scripts.compute_bertscore
"""

import json
from pathlib import Path

import numpy as np
from bert_score import score as bert_score

from src.evaluation.eval_dataset import EVAL_DATASET

PROCESSED_DIR = Path("data/processed")
MAX_WORDS = 400  # Truncation limit — prevents tokenizer overflow on long answers


def truncate(text: str, max_words: int = MAX_WORDS) -> str:
    """Truncate text to max_words to avoid tokenizer overflow."""
    words = text.split()
    if len(words) > max_words:
        return " ".join(words[:max_words])
    return text


def compute_bertscore():
    """Compute BERTScore for all eval result files."""
    reference_answers = [item["expected_answer"] for item in EVAL_DATASET]

    labels = ["baseline_zeroshot", "baseline_fewshot", "finetuned_lora"]
    summary_table = []

    for label in labels:
        file_path = PROCESSED_DIR / f"eval_{label}.json"
        if not file_path.exists():
            print(f"  Skipping {label} — file not found")
            continue

        with open(file_path) as f:
            data = json.load(f)

        details = data["answer"]["details"]
        if len(details) != len(reference_answers):
            print(f"  Skipping {label} — {len(details)} answers vs {len(reference_answers)} references")
            continue

        print(f"\n  Computing BERTScore for {label}...")

        predictions = [truncate(d["answer"]) for d in details]
        references = reference_answers

        P, R, F1 = bert_score(
            predictions,
            references,
            lang="en",
            model_type="distilbert-base-uncased",
            verbose=False,
        )

        mean_p = float(np.mean(P.numpy()))
        mean_r = float(np.mean(R.numpy()))
        mean_f1 = float(np.mean(F1.numpy()))

        print(f"    Precision: {mean_p:.4f}")
        print(f"    Recall:    {mean_r:.4f}")
        print(f"    F1:        {mean_f1:.4f}")

        # Store per-question scores
        for i, detail in enumerate(details):
            detail["bertscore_f1"] = round(float(F1[i]), 4)

        # Store aggregate scores
        data["answer"]["summary"]["bertscore_precision"] = round(mean_p, 4)
        data["answer"]["summary"]["bertscore_recall"] = round(mean_r, 4)
        data["answer"]["summary"]["bertscore_f1"] = round(mean_f1, 4)

        # Save updated results
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        summary_table.append({
            "label": label,
            "precision": mean_p,
            "recall": mean_r,
            "f1": mean_f1,
        })

    # Print comparison table
    if summary_table:
        print("\n" + "=" * 60)
        print("  BERTScore Comparison (distilbert-base-uncased)")
        print("=" * 60)
        print(f"  {'Config':<22} {'Precision':>10} {'Recall':>10} {'F1':>10}")
        print(f"  {'-'*22} {'-'*10} {'-'*10} {'-'*10}")
        for row in summary_table:
            print(f"  {row['label']:<22} {row['precision']:>10.4f} {row['recall']:>10.4f} {row['f1']:>10.4f}")
        print("=" * 60)

        # Save comparison summary
        summary_path = PROCESSED_DIR / "bertscore_comparison.json"
        with open(summary_path, "w") as f:
            json.dump(summary_table, f, indent=2)
        print(f"\n  Summary saved to {summary_path}")


if __name__ == "__main__":
    compute_bertscore()
