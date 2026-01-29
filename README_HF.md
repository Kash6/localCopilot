---
title: AI Coding Assistant
emoji: ⚡
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# AI Coding Assistant

Production-grade RAG-based coding assistant with LangChain, FAISS, and LoRA-tuned LLMs.

## Features

- Semantic Code Search with FAISS vector store
- LangChain RAG pipeline for context-aware responses
- DeepSeek Coder LLM with optional LoRA fine-tuning
- 8-bit quantization for efficient inference (GPU only)
- Optional Qdrant Cloud integration

## Usage

1. Enter your code repository path (or use sample data)
2. Click "Index Repository" to process your codebase
3. Ask questions or request code fixes
4. Optionally enable LoRA-tuned model (requires GPU)

## Configuration

For Qdrant Cloud integration, add secrets in Space settings:
- `QDRANT_URL`: Your cluster URL
- `QDRANT_API_KEY`: Your API key

## Performance

- CPU inference: ~10-30s per response (free tier)
- GPU inference: ~2-5s per response (upgrade required)
- First load: ~2-3 minutes (model download)

## Local Development

```bash
git clone https://github.com/Kash6/localCopilot
cd localCopilot

# Windows with GPU
setup_conda_gpu.bat
run_conda.bat

# Or use pip
pip install -r requirements.txt
streamlit run app.py
```

## Architecture

- **Vector Store**: FAISS (default) or Qdrant Cloud (optional)
- **Embeddings**: all-MiniLM-L6-v2
- **Reranker**: ms-marco-MiniLM-L-6-v2
- **LLM**: DeepSeek Coder 1.3B
- **Framework**: LangChain + Streamlit
