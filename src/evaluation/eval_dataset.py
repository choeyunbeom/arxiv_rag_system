"""
Evaluation Dataset (v2)
- Curated Q&A pairs with expected source papers
- expected_arxiv_ids updated to reflect papers actually retrievable from our corpus
- expected_answer provides reference answers for semantic evaluation (BERTScore)
- Used to benchmark RAG pipeline performance (baseline and post-fine-tuning)
"""

EVAL_DATASET = [
    {
        "question": "What is QLoRA and how does it reduce memory usage?",
        "expected_arxiv_ids": ["2305.14314v1", "2505.03406v1"],
        "expected_keywords": ["4-bit", "quantization", "Low-Rank Adapter", "LoRA", "memory", "NF4"],
        "expected_answer": (
            "QLoRA is an efficient fine-tuning approach that reduces memory usage by quantizing "
            "the base model to 4-bit NormalFloat (NF4) precision while training low-rank adapters (LoRA). "
            "It introduces three key innovations: 4-bit NormalFloat quantization for normally distributed weights, "
            "double quantization that quantizes the quantization constants themselves, and paged optimizers "
            "using NVIDIA unified memory to handle memory spikes during training."
        ),
        "topic": "qlora",
    },
    {
        "question": "How does LoRA work for fine-tuning large language models?",
        "expected_arxiv_ids": ["2106.09685v2", "2512.17983v1", "2602.05988v1"],
        "expected_keywords": ["low-rank", "adapter", "weight matrices", "trainable parameters", "frozen"],
        "expected_answer": (
            "LoRA works by keeping the pre-trained model weights frozen and introducing small, "
            "trainable low-rank decomposition matrices into each Transformer layer. Instead of updating "
            "the full weight matrix W, LoRA adds a low-rank update W + BA where B and A are small matrices. "
            "This significantly reduces the number of trainable parameters while maintaining model performance."
        ),
        "topic": "lora",
    },
    {
        "question": "What is Retrieval Augmented Generation?",
        "expected_arxiv_ids": ["2601.05264v1", "2502.13957v2", "2401.15391v1", "2502.00306v2"],
        "expected_keywords": ["retrieval", "generation", "external knowledge", "context", "LLM"],
        "expected_answer": (
            "Retrieval Augmented Generation (RAG) is a technique that enhances language models by "
            "retrieving relevant documents or passages from an external knowledge base before generating "
            "an answer. This grounds the LLM's response in factual, up-to-date information, reducing "
            "hallucinations and enabling the model to answer questions beyond its training data."
        ),
        "topic": "rag",
    },
    {
        "question": "How can RAG systems be evaluated?",
        "expected_arxiv_ids": ["2309.15217v2", "2502.00306v2", "2601.05264v1"],
        "expected_keywords": ["evaluation", "metrics", "faithfulness", "relevance", "context"],
        "expected_answer": (
            "RAG systems can be evaluated using metrics across two dimensions: retrieval quality "
            "(context precision, context recall, hit rate, MRR) and generation quality (faithfulness "
            "to retrieved context, answer relevance to the question). Frameworks like Ragas provide "
            "automated, reference-free evaluation using LLM-as-a-judge approaches."
        ),
        "topic": "rag_eval",
    },
    {
        "question": "What techniques exist to mitigate hallucinations in LLMs?",
        "expected_arxiv_ids": ["2311.08117v1", "2409.20550v2", "2507.15903v1", "2510.19507v2", "2502.11306v1"],
        "expected_keywords": ["hallucination", "mitigation", "grounding", "factual", "verification"],
        "expected_answer": (
            "Techniques to mitigate hallucinations include: Retrieval-Augmented Generation (RAG) for "
            "grounding responses in factual sources, self-consistency checking where the model verifies "
            "its own outputs, fine-tuning with human feedback (RLHF) to penalise unsupported claims, "
            "and constrained decoding that restricts generation to verifiable statements."
        ),
        "topic": "hallucination",
    },
    {
        "question": "What is instruction tuning and why is it important?",
        "expected_arxiv_ids": ["2304.07995v1", "2409.14254v1", "2304.03277v1", "2304.12244v3"],
        "expected_keywords": ["instruction", "tuning", "fine-tuning", "alignment", "task"],
        "expected_answer": (
            "Instruction tuning involves fine-tuning a pre-trained language model on a dataset of "
            "tasks described via natural language instructions. It is important because it aligns "
            "the model to follow user intent, improving zero-shot generalisation across diverse tasks "
            "without requiring task-specific training examples."
        ),
        "topic": "instruction_tuning",
    },
    {
        "question": "How does LoRA+ improve upon standard LoRA?",
        "expected_arxiv_ids": ["2402.12354v2"],
        "expected_keywords": ["learning rate", "adapter", "matrix B", "matrix A", "efficiency"],
        "expected_answer": (
            "LoRA+ improves standard LoRA by using different learning rates for the adapter matrices "
            "A and B. Specifically, it sets a much higher learning rate for matrix B than for matrix A, "
            "which accelerates convergence and improves final task performance without increasing "
            "parameter count or computational cost."
        ),
        "topic": "lora_plus",
    },
    {
        "question": "What is the difference between full fine-tuning and parameter-efficient fine-tuning?",
        "expected_arxiv_ids": ["2106.09685v2", "2305.14314v1", "2512.17983v1", "2412.09827v1"],
        "expected_keywords": ["parameter-efficient", "PEFT", "LoRA", "full fine-tuning", "trainable"],
        "expected_answer": (
            "Full fine-tuning updates all parameters of a model, requiring massive memory and compute "
            "proportional to model size. Parameter-efficient fine-tuning (PEFT) methods like LoRA "
            "freeze the base model and only train a small subset of additional parameters, reducing "
            "memory and compute requirements by orders of magnitude while maintaining comparable performance."
        ),
        "topic": "peft",
    },
    {
        "question": "How do multi-hop questions challenge RAG systems?",
        "expected_arxiv_ids": ["2401.15391v1", "2502.13957v2"],
        "expected_keywords": ["multi-hop", "reasoning", "multiple documents", "complex", "retrieval"],
        "expected_answer": (
            "Multi-hop questions require reasoning across multiple documents to form a complete answer. "
            "Standard RAG systems struggle because a single retrieval query often captures only part of "
            "the needed information. Effective multi-hop RAG requires iterative retrieval, query "
            "decomposition, or chain-of-thought reasoning to connect information across documents."
        ),
        "topic": "multihop_rag",
    },
    {
        "question": "What is the role of vector databases in RAG pipelines?",
        "expected_arxiv_ids": ["2601.05264v1", "2505.12524v1", "2502.00306v2"],
        "expected_keywords": ["vector", "database", "embedding", "similarity", "retrieval", "index"],
        "expected_answer": (
            "Vector databases store document chunk embeddings and enable fast similarity search. "
            "When a user submits a query, it is embedded into the same vector space, and the database "
            "finds the most semantically similar chunks using approximate nearest neighbour search. "
            "This forms the retrieval stage of the RAG pipeline."
        ),
        "topic": "vector_db",
    },
    {
        "question": "How can prompt engineering improve LLM outputs?",
        "expected_arxiv_ids": ["2507.03405v1", "2309.13734v2", "2503.02400v2", "2401.14043v3"],
        "expected_keywords": ["prompt", "engineering", "technique", "few-shot", "chain-of-thought"],
        "expected_answer": (
            "Prompt engineering improves LLM outputs by structuring input queries with techniques "
            "like few-shot examples, chain-of-thought reasoning instructions, and role-based system "
            "prompts. These guide the model to adopt specific reasoning paths, output formats, or "
            "domain-specific behaviours without modifying model weights."
        ),
        "topic": "prompt_engineering",
    },
    {
        "question": "What is Double Quantization in QLoRA?",
        "expected_arxiv_ids": ["2305.14314v1", "2404.00862v1"],
        "expected_keywords": ["double quantization", "quantization constants", "memory", "FP32", "8-bit"],
        "expected_answer": (
            "Double Quantization is a memory-saving technique in QLoRA that quantizes the quantization "
            "constants themselves. Standard quantization stores scaling factors as 32-bit floats; "
            "double quantization reduces these to 8-bit, saving approximately 0.37 bits per parameter "
            "and about 3 GB for a 65B parameter model."
        ),
        "topic": "double_quant",
    },
    {
        "question": "How does the Ragas framework evaluate RAG pipelines?",
        "expected_arxiv_ids": ["2309.15217v2", "2601.05264v1"],
        "expected_keywords": ["Ragas", "faithfulness", "answer relevance", "context", "reference-free"],
        "expected_answer": (
            "Ragas evaluates RAG pipelines using reference-free, LLM-powered metrics. It measures "
            "faithfulness (whether the answer is supported by retrieved context), answer relevance "
            "(whether the answer addresses the question), context precision (proportion of relevant "
            "chunks retrieved), and context recall (coverage of ground truth information)."
        ),
        "topic": "ragas",
    },
    {
        "question": "What are corpus poisoning attacks on RAG systems?",
        "expected_arxiv_ids": ["2512.24268v1"],
        "expected_keywords": ["poisoning", "adversarial", "attack", "corpus", "defense", "retrieval"],
        "expected_answer": (
            "Corpus poisoning attacks inject adversarial or fabricated documents into the retrieval "
            "corpus. When a user submits a query, the RAG system retrieves these poisoned documents "
            "as context, causing the LLM to generate incorrect or misleading answers. Defences include "
            "document provenance verification and adversarial training of the retriever."
        ),
        "topic": "rag_security",
    },
    {
        "question": "How can small language models be made more efficient through fine-tuning?",
        "expected_arxiv_ids": ["2408.00690v2", "2504.16584v1", "2305.14314v1"],
        "expected_keywords": ["small", "efficient", "fine-tuning", "quantization", "parameter"],
        "expected_answer": (
            "Small language models can be made more efficient through parameter-efficient fine-tuning "
            "methods like LoRA and QLoRA, which train only a fraction of parameters. Combined with "
            "quantization during inference, these techniques allow small models to achieve specialised "
            "task performance comparable to larger models while running on limited hardware."
        ),
        "topic": "small_llm",
    },
]
