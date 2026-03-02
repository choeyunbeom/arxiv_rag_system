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

## Evaluation: Base vs Fine-Tuned

Ran the same 15-question benchmark on both models under identical conditions (same retrieval pipeline, same prompts, same hardware).

### Aggregate Results

| Metric | Base Model | Fine-Tuned | Change |
|--------|-----------|------------|--------|
| Keyword Coverage | 80.2% | 71.1% | -9.1%p ❌ |
| Substantive Rate | 100% | 93.3% | -6.7%p ❌ |
| Source Hit Rate | 100% | 100% | — |
| MRR | 0.82 | 0.82 | — |
| Avg Latency | 20.3s | 22.0s | +1.7s |

### Per-Question Keyword Coverage

| Topic | Base | Fine-Tuned | Δ |
|-------|------|------------|---|
| qlora | 83% | 100% | +17%p ✅ |
| rag | 100% | 100% | — |
| rag_eval | 100% | 100% | — |
| peft | 100% | 100% | — |
| prompt_engineering | 100% | 100% | — |
| vector_db | 100% | 83% | -17%p |
| rag_security | 100% | 83% | -17%p |
| instruction_tuning | 80% | 80% | — |
| multihop_rag | 80% | 60% | -20%p |
| small_llm | 80% | 80% | — |
| double_quant | 60% | 60% | — |
| hallucination | 40% | 40% | — |
| lora | 40% | 20% | -20%p |
| lora_plus | 40% | 40% | — |
| **ragas** | **100%** | **20%** | **-80%p ❌** |

The ragas question is the single biggest contributor to the regression — the fine-tuned model returned only 3 words ("Ragas evaluates R") before truncating, consuming 33.6 seconds (longest latency in the set).

## Root Cause Analysis

### 1. Catastrophic Forgetting in Small Models

At 4B parameters, the model's capacity is limited. LoRA fine-tuning on 2,000 examples successfully shifted response style (more concise, less markdown), but this came at the cost of comprehensive topic coverage. Larger models (7B+) have more capacity to absorb new behaviours without degrading existing capabilities.

### 2. Evaluation Metric Mismatch

Keyword Coverage measures whether specific terms appear in the answer. The fine-tuned model was trained to produce concise, prose-style answers — exactly the behaviour that reduces keyword recall. For example, a base model answer that uses both "parameter-efficient fine-tuning" and "PEFT" scores higher than a fine-tuned answer that uses only "PEFT", even though both are correct.

A semantic similarity metric (e.g., BERTScore, GPT-as-judge) would better capture whether the answer conveys the same meaning in fewer words.

### 3. Quantisation Gap

The base model runs as Ollama's default `qwen3:4b` (Q4_K_M quantisation), while the fine-tuned model was converted to Q8_0 GGUF. Different quantisation methods affect token probability distributions and thus generation behaviour, independent of the fine-tuning itself. A fair comparison would require both models at the same quantisation level.

### 4. Thinking Mode Interaction

Qwen3's `<think>` reasoning mode behaves differently after fine-tuning. The `/no_think` prompt injection was calibrated for base model weights and is less effective after LoRA modification. The ragas question failure (3-word output after 33.6s) strongly suggests the model spent its token budget on internal reasoning.

## What I Would Do Differently

1. **Use a larger base model (7B+)** — more capacity reduces catastrophic forgetting risk
2. **Add semantic evaluation metrics** — BERTScore or GPT-as-judge alongside keyword matching
3. **Include `/no_think` in training data** — train the model with thinking mode explicitly disabled
4. **Use 1 epoch with lower LR (5e-5)** — minimise forgetting while still imparting style changes
5. **Quantise both models identically** — Q8_0 for both base and fine-tuned for fair comparison
6. **A/B test with human evaluation** — capture qualitative improvements that automated metrics miss
7. **Increase training data to 10K+** — more diverse examples reduce overfitting to specific patterns

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
