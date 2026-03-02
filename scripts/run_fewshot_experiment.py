"""
3-Way Prompt Comparison Experiment
Runs the evaluation pipeline three times with different configurations:
  1. Zero-shot (base model, default prompt)
  2. Few-shot  (base model, few-shot prompt with style examples)
  3. Fine-tuned (LoRA model, default prompt)

Usage:
    python -m scripts.run_fewshot_experiment
"""

from src.api.core.prompts import SYSTEM_PROMPT_FEW_SHOT
from src.evaluation.evaluate import run_evaluation


def main():
    # ------------------------------------------------------------------
    # Experiment 1: Zero-Shot Baseline (base model + default prompt)
    # ------------------------------------------------------------------
    # run_evaluation(
    #     top_k=5,
    #     label="baseline_zeroshot",
    # )

    # ------------------------------------------------------------------
    # Experiment 2: Few-Shot Baseline (base model + few-shot prompt)
    # ------------------------------------------------------------------
    # run_evaluation(
    #     top_k=5,
    #     label="baseline_fewshot",
    #     system_prompt=SYSTEM_PROMPT_FEW_SHOT,
    # )

    # ------------------------------------------------------------------
    # Experiment 3: Fine-Tuned (requires LLM_MODEL=qwen3-4b-rag in .env
    # or OLLAMA model swap before running)
    # ------------------------------------------------------------------
    # To run this experiment:
    #   1. Change LLM_MODEL in .env to "qwen3-4b-rag"
    #   2. Uncomment the block below
    #   3. Run this script again
    
    run_evaluation(
        top_k=5,
        label="finetuned_lora",
    )

    print("\n" + "=" * 60)
    print("  All experiments complete.")
    print("  Results saved to data/processed/eval_*.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
