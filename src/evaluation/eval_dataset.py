"""
Evaluation Dataset
- Curated Q&A pairs with expected source papers
- Used to benchmark RAG pipeline performance (baseline and post-fine-tuning)
"""

EVAL_DATASET = [
    {
        "question": "What is QLoRA and how does it reduce memory usage?",
        "expected_arxiv_ids": ["2305.14314v1"],
        "expected_keywords": ["4-bit", "quantization", "Low-Rank Adapter", "LoRA", "memory", "NF4"],
        "topic": "qlora",
    },
    {
        "question": "How does LoRA work for fine-tuning large language models?",
        "expected_arxiv_ids": ["2106.09685v2"],
        "expected_keywords": ["low-rank", "adapter", "weight matrices", "trainable parameters", "frozen"],
        "topic": "lora",
    },
    {
        "question": "What is Retrieval Augmented Generation?",
        "expected_arxiv_ids": ["2601.05264v1", "2502.13957v2", "2401.15391v1"],
        "expected_keywords": ["retrieval", "generation", "external knowledge", "context", "LLM"],
        "topic": "rag",
    },
    {
        "question": "How can RAG systems be evaluated?",
        "expected_arxiv_ids": ["2309.15217v2", "2507.21753v1"],
        "expected_keywords": ["evaluation", "metrics", "faithfulness", "relevance", "context"],
        "topic": "rag_eval",
    },
    {
        "question": "What techniques exist to mitigate hallucinations in LLMs?",
        "expected_arxiv_ids": ["2510.19507v2", "2502.11306v1", "2508.14314v2"],
        "expected_keywords": ["hallucination", "mitigation", "grounding", "factual", "verification"],
        "topic": "hallucination",
    },
    {
        "question": "What is instruction tuning and why is it important?",
        "expected_arxiv_ids": ["2304.03277v1", "2304.12244v3"],
        "expected_keywords": ["instruction", "tuning", "fine-tuning", "alignment", "task"],
        "topic": "instruction_tuning",
    },
    {
        "question": "How does LoRA+ improve upon standard LoRA?",
        "expected_arxiv_ids": ["2402.12354v2"],
        "expected_keywords": ["learning rate", "adapter", "matrix B", "matrix A", "efficiency"],
        "topic": "lora_plus",
    },
    {
        "question": "What is the difference between full fine-tuning and parameter-efficient fine-tuning?",
        "expected_arxiv_ids": ["2106.09685v2", "2305.14314v1"],
        "expected_keywords": ["parameter-efficient", "PEFT", "LoRA", "full fine-tuning", "trainable"],
        "topic": "peft",
    },
    {
        "question": "How do multi-hop questions challenge RAG systems?",
        "expected_arxiv_ids": ["2401.15391v1"],
        "expected_keywords": ["multi-hop", "reasoning", "multiple documents", "complex", "retrieval"],
        "topic": "multihop_rag",
    },
    {
        "question": "What is the role of vector databases in RAG pipelines?",
        "expected_arxiv_ids": ["2601.05264v1", "2505.12524v1"],
        "expected_keywords": ["vector", "database", "embedding", "similarity", "retrieval", "index"],
        "topic": "vector_db",
    },
    {
        "question": "How can prompt engineering improve LLM outputs?",
        "expected_arxiv_ids": ["2401.14043v3", "2507.03405v1"],
        "expected_keywords": ["prompt", "engineering", "technique", "few-shot", "chain-of-thought"],
        "topic": "prompt_engineering",
    },
    {
        "question": "What is Double Quantization in QLoRA?",
        "expected_arxiv_ids": ["2305.14314v1"],
        "expected_keywords": ["double quantization", "quantization constants", "memory", "FP32", "8-bit"],
        "topic": "double_quant",
    },
    {
        "question": "How does the Ragas framework evaluate RAG pipelines?",
        "expected_arxiv_ids": ["2309.15217v2"],
        "expected_keywords": ["Ragas", "faithfulness", "answer relevance", "context", "reference-free"],
        "topic": "ragas",
    },
    {
        "question": "What are corpus poisoning attacks on RAG systems?",
        "expected_arxiv_ids": ["2512.24268v1"],
        "expected_keywords": ["poisoning", "adversarial", "attack", "corpus", "defense", "retrieval"],
        "topic": "rag_security",
    },
    {
        "question": "How can small language models be made more efficient through fine-tuning?",
        "expected_arxiv_ids": ["2504.16584v1", "2305.14314v1"],
        "expected_keywords": ["small", "efficient", "fine-tuning", "quantization", "parameter"],
        "topic": "small_llm",
    },
]
