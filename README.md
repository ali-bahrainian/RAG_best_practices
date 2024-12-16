# RAG Best Practices

## Overview
This repository implements a **Retrieval-Augmented Generation (RAG)** pipeline. The framework expands user queries, retrieves relevant contexts, and generates responces using a Large Language Model (LLM).

The RAG framework combines:
1. **Query Expansion Module**: Expands the query using a language model (LM).
2. **Retrieval Module**: Retrieves similar documents or sentences.
3. **Generative LLM**: Generates the final answer based on the retrieved contexts.

![RAG Framework Overview](rag-diagram.png)

---
## Configuration
This project provides a flexible configuration system to customize the RAG system. Key settings include:
```bash
base_config = {
    # Language Model Settings
    "generation_model_name": "mistralai/Mistral-7B-Instruct-v0.2",  # 7B-parameter instruction-tuned LLM
    "embedding_model_name": "sentence-transformers/all-MiniLM-L6-v2",  # Model for document embeddings
    "seq2seq_model_name": "google/flan-t5-small",  # Small T5 model for query expansion
    "is_chat_model": True,  # Indicates if the model follows chat-based input/output

    # Prompt Design
    "instruct_tokens": ("[INST]", "[/INST]"),  # Instruction tokens to guide the LLM

    # Document Indexing and Chunking
    "index_builder": {
        "tokenizer_model_name": None,  # Defaults to the embedding model tokenizer
        "chunk_size": 64,              # Number of tokens per document chunk
        "overlap": 8,                  # Overlap of tokens between chunks for context continuity
        "passes": 10,                  # Number of document passes for indexing
        "icl_kb": False,               # Contrastive In-Context Learning knowledge base (disabled)
        "multi_lingo": False           # Multilingual knowledge base support (disabled)
    },

    # Retrieval-Augmented Language Model (RALM) Settings
    "ralm": {
        "expand_query": False,         # Query expansion techniques (disabled)
        "top_k_docs": 2,               # Top-2 documents retrieved for relevance
        "top_k_titles": 7,             # Top-7 titles retrieved for Step 1 retrieval
        "system_prompt": ......,       # System prompt for generating responses
        "repeat_system_prompt": True,  # Repeat system prompt to guide generation
        "stride": -1,                  # Retrieval stride: -1 means no fixed stride
        "query_len": 200,              # Maximum query length in tokens
        "do_sample": False,            # Disable sampling for deterministic outputs
        "temperature": 1.0,            # Control randomness in generation
        "top_p": 0.1,                  # Nucleus sampling: considers tokens in top-10% probability mass
        "num_beams": 2,                # Number of beams for beam search
        "max_new_tokens": 25,          # Limit the number of generated tokens
        "batch_size": 8,               # Batch size for processing
        "kb_10K": False,               # 10K knowledge base support (disabled)
        "icl_kb": False,               # ICL knowledge base support (disabled)
        "icl_kb_incorrect": False,     # Incorrect ICL knowledge base (disabled)
        "focus": False                 # Focus mode for sentence-level retrieval (disabled)
    }
}

```

---
## Installation
1. **Clone mixtral-offloading repository**:
   ```bash
   git clone https://github.com/dvmazur/mixtral-offloading.git
   ```
2. **Download the Mixtral-8x7B Mode**:
    ```bash
    huggingface-cli download lavawolfiee/Mixtral-8x7B-Instruct-v0.1-offloading-demo --quiet --local-dir Mixtral-8x7B-Instruct-v0.1-offloading-demo
    ```
3. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
---

## Project Structure
Your final directory structure should look like this:
```
RAG_best_practices/ 
│── model/                       # Core RAG implementation 
│ ├── index_builder.py           # Builds document index 
│ ├── language_model.py          # Query expansion logic 
│ ├── model_loader.py            # Loads Mixtral LLM 
│ ├── rag.py                     # Main RAG pipeline 
│ ├── retriever.py               # Retrieves documents 
│ ├── config.py                  # Configuration setup 
│ ├── evaluation.py              # Runs the full RAG pipeline 
│ ├── overview.png               # Image explaining the RAG pipeline 
│ ├── requirements.txt           # Python dependencies 
└── README.md 
```
---

## Run RAG System 
To evaluate our RAG system with different configurations, simply run:

```bash
python resources/evaluation.py
```
---

## Citation
If you find our paper or code helpful, please cite our paper:
```
@inproceedings{title={Enhancing Retrieval-Augmented Generation: A Study of Best Practices}}
```