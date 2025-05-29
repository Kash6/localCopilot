# localCopilot
🚀 Codebase Fix Assistant with RAG + LLM
Context-aware code editing using FAISS, reranking, and quantized LLM inference

🔍 Overview
This project is a retrieval-augmented generation (RAG) pipeline that assists developers in generating code fixes across large repositories. It combines:

Semantic code search using FAISS and SentenceTransformers
Code-aware reranking with CrossEncoder
LLM-based fix generation using a quantized transformer model (e.g., Deepseek, Mistral, or Meta's LLaMA)
Streamlit UI for interactive prompt construction and live editing
Built for efficient inference on consumer GPUs (RTX 3060) using 8-bit quantization.

🧱 Architecture


![image](https://github.com/user-attachments/assets/fa3a3c98-3a65-4454-9677-873d271c8002)

  
⚙️ Features
File-aware retrieval (e.g., query: “Modify testb.py…” limits scope)

Automatic reranking via cross-encoder/ms-marco-MiniLM-L-6-v2

Support for 8-bit quantized models using BitsAndBytesConfig

On-device inference (tested on RTX 3060, 6GB VRAM)

Token-aware trimming and output parsing

Streamlit UI with live input, output, and cache clearing

![Screenshot 2025-05-29 032036](https://github.com/user-attachments/assets/43b5745c-c4a4-4e9e-94cf-03c3e15bc46c)

📦 Setup Instructions

Requirements
Python 3.9+
CUDA-enabled GPU (recommended: RTX 3060+)
Packages:
pip install torch torchvision
pip install sentence-transformers faiss-cpu transformers accelerate bitsandbytes
pip install streamlit scikit-learn

▶ Run the App

streamlit run assistant.py

Models Used
Component	Model Name
Embedding Model	all-MiniLM-L6-v2 (SentenceTransformer)
Reranker	cross-encoder/ms-marco-MiniLM-L-6-v2
LLM	deepseek-ai/deepseek-coder-1.3b-instruct (or LLaMA/Mistral/etc)


🧾 License
MIT License — feel free to fork, modify, and extend.
