---
title: AI Coding Assistant
emoji: ⚡
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: "1.29.0"
app_file: app.py
pinned: false
license: mit
---

# AI Coding Assistant

Local AI-powered coding assistant with RAG-based context retrieval and automated code fixes.

## Features

- Sentence Transformers + FAISS for semantic code search
- LangChain RAG pipeline for context-aware responses
- DeepSeek Coder LLM with optional LoRA fine-tuning
- 8-bit quantization for efficient inference
- Optional Qdrant Cloud integration for advanced vector search

## Usage

1. Upload your code repository or use the sample data
2. Index your codebase
3. Ask questions or request code fixes
4. Optionally enable LoRA-tuned model for specialized tasks

## Configuration

For Qdrant Cloud integration, add your credentials in Settings:
- QDRANT_URL: Your cluster URL
- QDRANT_API_KEY: Your API key

## Local Development

```bash
# Clone repository
git clone https://github.com/Kash6/localCopilot
cd localCopilot

# Setup with GPU support (Windows)
setup_conda_gpu.bat
run_conda.bat

# Or use pip
pip install -r requirements.txt
streamlit run app.py
```

## Architecture

- Vector Store: FAISS (default) or Qdrant Cloud (optional)
- Embeddings: all-MiniLM-L6-v2
- Reranker: ms-marco-MiniLM-L-6-v2
- LLM: DeepSeek Coder 1.3B with 8-bit quantization
- Framework: LangChain + Streamlit

## Performance

- 40% latency reduction with LoRA tuning and quantization
- Efficient memory usage with 8-bit quantization
- GPU acceleration when available
