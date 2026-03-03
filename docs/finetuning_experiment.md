# Fine-Tuning Experiment Log

## Objective

Fine-tune Qwen3 4B with LoRA to improve RAG-specific behaviours:
1. **Context grounding** — answer only from provided context, cite paper titles
2. **Prose output** — no markdown headers or bullet points
3. **Proper refusal** — decline when context is insufficient

## Training Data

Generated 1,997 synthetic Q&A pairs from the 132-paper corpus using Qwen3 4B itself via Ollama's `format: json` parameter.

| Type | Count | Purpose |
|------|-------|---------|
| Grounded (60%) | 1,200 | Single-paper context → cited prose answer |
| Synthesis (20%) | 397 | Two-paper context → comparative prose answer |
| Refusal (20%) | 400 | Irrelevant context → polite refusal with explanation |

**Data format**: Each sample follows the Qwen3 chat template:
- `system` → instruction (RAG behaviour rules)
- `user` → context chunks + question
- `assistant` → expected answer

**Token statistics**: min 257, max 841, mean 377 (all within 2048 max_length, 0 truncated)

**Generation speed**: ~33 pairs/min (1,997 pairs in 67 minutes)

### Qwen3 Thinking Mode Discovery

Qwen3's `<think>` feature consumes output tokens for internal reasoning before producing visible output. With `num_predict: 512`, the model exhausted all tokens on thinking and returned empty responses. Key discovery: combining Ollama's `format: json` with `num_predict: 4096` causes the model to produce structured JSON within its thinking field, which can be extracted programmatically. This reduced generation time from ~60s/pair to ~2s/pair.

## Training Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Base model | Qwen3-4B (bf16) | bf16 instead of 4-bit because bitsandbytes is unstable on MPS |
| LoRA rank (r) | 16 | Balance between expressiveness and parameter count |
| LoRA alpha (α) | 32 | Standard 2× rank ratio |
| LoRA dropout | 0.05 | Light regularisation |
| Target modules | q/k/v/o_proj, gate/up/down_proj | All attention + MLP projections for maximum adaptation |
| Epochs | 3 | Sufficient for 2K samples; overfitting monitored via eval loss |
| Batch size | 2 | Conservative for MPS stability |
| Gradient accumulation | 8 | Effective batch size = 16 |
| Learning rate | 2e-4 | Standard for LoRA fine-tuning |
| LR scheduler | Cosine | Smooth decay prevents late-stage instability |
| Warmup ratio | 5% | Brief warmup for stable early training |
| Precision | bf16 | Native Apple Silicon support |
| Max sequence length | 2048 | Covers all training samples |
| Train/Eval split | 1,897 / 100 | 5% held out for validation |

**Trainable parameters**: 33,030,144 / 4,055,498,240 (0.81%)

**Hardware**: Apple M4 Pro, 48GB unified memory, MPS backend

**Framework**: trl 0.29.0 (SFTTrainer with SFTConfig), PEFT 0.15.1

## Training Results

| Epoch | Train Loss | Validation Loss | Notes |
|-------|-----------|-----------------|-------|
| 1 | 1.1056 | 1.1180 | Baseline convergence |
| 2 | 1.0227 | **1.0602** | Best checkpoint ← |
| 3 | 0.8818 | 1.0640 | Slight overfitting (+0.004) |

- **Total training time**: 24,626 seconds (410 minutes, ~6.8 hours)
- **Throughput**: 0.231 samples/sec (~50s/step)
- **Best model**: Epoch 2 (auto-selected via `load_best_model_at_end=True`)
- **Final train loss**: 1.1279

The training loss continued decreasing at epoch 3, but validation loss plateaued, indicating the model began memorising training examples rather than generalising.

## Model Conversion Pipeline

### 1. LoRA Merge

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("data/base_model")
model = PeftModel.from_pretrained(base, "data/finetuned_lora/final")
merged = model.merge_and_unload()
merged.save_pretrained("data/merged_model")
```

**Issue**: `save_pretrained()` only saves model weights, not tokenizer files. Required manual copy of `tokenizer.json`, `tokenizer_config.json`, `vocab.json`, `merges.txt` from base model.

### 2. GGUF Conversion

```bash
python llama.cpp/convert_hf_to_gguf.py data/merged_model \
    --outfile data/qwen3-4b-rag.gguf --outtype q8_0
```

- Output: 4.27 GB (Q8_0 quantisation)
- `q4_K_M` not supported by `convert_hf_to_gguf.py` — requires separate `llama-quantize` step
- BPE pre-tokenizer warning resolved by copying base model tokenizer files

### 3. Ollama Registration

```bash
echo 'FROM data/qwen3-4b-rag.gguf' > Modelfile
ollama create qwen3-4b-rag -f Modelfile
```

### Sanity Test

```
Query: "What is QLoRA?" (with context from QLoRA paper)

Response: "QLoRA is a method that reduces memory usage enough to fine-tune
a 65B parameter model on a single 48GB GPU while preserving full 16-bit
fine-tuning task performance."
```

- Grounded to context ✓
- Concise prose ✓  
- No markdown formatting ✓
- `<think>` tags present but empty (minimal reasoning needed) ✓

## Evaluation: 3-Way Comparison

Ran the same 15-question benchmark on all three configurations under identical conditions (same retrieval pipeline, same hardware).

### Aggregate Results

| Metric | Zero-Shot | Few-Shot | Fine-Tuned |
|--------|-----------|----------|------------|
| Keyword Coverage | 76.4% | **78.0%** | 48.0% |
| Source Hit Rate | 100% | 100% | 100% |
| Substantive Rate | 100% | 100% | 100% |
| Avg Word Count | 175 | 177 | 1,614 |
| Avg Latency | 20.0s | 20.8s | 47.7s |

### Per-Question Keyword Coverage

| Topic | Zero-Shot | Few-Shot | Fine-Tuned |
|-------|-----------|----------|------------|
| qlora | 83% | 83% | 83% |
| rag | 100% | 100% | 80% |
| rag_eval | 100% | 100% | 60% |
| peft | 100% | 100% | 80% |
| prompt_engineering | 100% | 100% | 60% |
| vector_db | 83% | 100% | 50% |
| rag_security | 83% | 83% | 50% |
| instruction_tuning | 60% | 60% | 40% |
| multihop_rag | 40% | 80% | 20% |
| small_llm | 60% | 60% | 40% |
| double_quant | 60% | 60% | 0% |
| hallucination | 60% | 60% | 20% |
| lora | 60% | 40% | 40% |
| lora_plus | 60% | 60% | 20% |
| ragas (topic) | 100% | 80% | 0% |

> Note: the "ragas" row refers to a benchmark question about the Ragas evaluation framework — it is **not** a Ragas metric score. The fine-tuned model scored 0% keyword coverage on this question because every response began by repeating system prompt instructions verbatim, displacing actual answer content. Word counts averaged ~1,600 tokens — nearly 10× the base model — with instruction parroting consuming the majority of the output.

## Root Cause Analysis

### 1. Training Data Contamination (Primary Cause)

The synthetic data generation pipeline used Qwen3's `format: json` with thinking mode enabled. The model's `thinking` field contained system prompt fragments mixed with reasoning. When these were extracted as training answers, the model learned to reproduce instruction text as part of its response — a form of **training data contamination** where the model memorised prompt scaffolding rather than learning the intended answering behaviour.

Inspection of fine-tuned responses confirmed this: **every response begins by repeating the system prompt instructions verbatim** ("Answer in concise prose paragraphs without markdown headers or bullet points..."), inflating word counts to ~1,600 and displacing actual answer content. This instruction parroting is the dominant failure mode, causing keyword coverage to collapse to 0% on 6 of 15 questions.

### 2. Evaluation Metric Limitation

Keyword Coverage measures whether specific terms appear in the answer. The fine-tuned model's verbose, instruction-padded responses fail exact string matching even when the semantic content may be partially correct. Without semantic evaluation metrics like **BERTScore** or LLM-as-a-judge frameworks (e.g., Ragas), it is impossible to conclusively separate "poor answering capability" from "different vocabulary/verbosity".

### 3. Catastrophic Forgetting in Small Models

At 4B parameters, the model's capacity is limited. LoRA fine-tuning on 2,000 examples shifted response style but degraded topic coverage. Larger models (7B+) have more capacity to absorb new behaviours without losing existing capabilities.

### 4. Quantisation Gap

The base model runs as Ollama's default `qwen3:4b` (Q4_K_M quantisation), while the fine-tuned model was converted to Q8_0 GGUF. Different quantisation methods affect token probability distributions, making comparison imprecise.

## What I Would Do Differently

1. **Validate training data for instruction leakage** — add automated checks that reject any training answer containing system prompt fragments
2. **Use a separate model for data generation** — avoid the thinking mode contamination issue by generating data with a different (typically larger) model
3. **Implement semantic evaluation metrics** — BERTScore and Ragas are critical; without them, we cannot distinguish "poor answering" from "different vocabulary"
4. **Start with few-shot baseline before fine-tuning** — establish prompt engineering ceiling first, then fine-tune only if there is a clear gap
5. **Use a larger base model (7B+)** — more capacity reduces catastrophic forgetting risk
6. **Use 1 epoch with lower LR (5e-5)** — minimise forgetting while still imparting style changes
7. **Quantise both models identically** — Q8_0 for both base and fine-tuned for fair comparison

## Files

| Artifact | Path | Note |
|----------|------|------|
| Training notebook | `src/finetuning/finetune_lora.ipynb` | Full training code with outputs |
| Training data | `data/processed/qa_dataset.json` | 1,997 Q&A pairs |
| Data generator | `src/finetuning/generate_qa_dataset.py` | Synthetic data pipeline |
| LoRA adapter | `data/finetuned_lora/final/` | Best checkpoint (git-ignored) |
| Training metrics | `data/finetuned_lora/training_metrics.json` | Loss curves |
| GGUF model | `data/qwen3-4b-rag.gguf` | Ollama-ready model (git-ignored) |
| Evaluation results | `data/processed/eval_baseline.json` | Base model benchmark |
