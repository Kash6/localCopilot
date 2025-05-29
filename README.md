# localCopilot
🚀 Codebase Fix Assistant with RAG + LLM
Context-aware code editing using FAISS, reranking, and quantized LLM inference

🔍 Overview
This project is a retrieval-augmented generation (RAG) pipeline that assists developers in generating code fixes across large repositories. It combines:

🔎 Semantic code search using FAISS and SentenceTransformers

🧠 Code-aware reranking with CrossEncoder

🤖 LLM-based fix generation using a quantized transformer model (e.g., Deepseek, Mistral, or Meta's LLaMA)

🖥️ Streamlit UI for interactive prompt construction and live editing

Built for efficient inference on consumer GPUs (RTX 3060) using 8-bit quantization.

🎯 Use Case
Give it:

A task like:
Modify classify_data function to rename variable unique_classes to different_classes

A directory of code files

And it will:

Retrieve the most relevant file(s)

Rerank results with a cross-encoder

Feed context + task into an LLM

Return and save the generated fix

🧱 Architecture
mathematica
Copy
Edit
           ┌─────────────┐
           │  Query Input│◄────────────┐
           └─────┬───────┘             │
                 ▼                     │
        ┌──────────────────┐          ▼
        │  FAISS Vector DB │ <── Code Embeddings
        └────────┬─────────┘
                 ▼
        ┌───────────────────────┐
        │ CrossEncoder Reranker│
        └────────┬──────────────┘
                 ▼
           Context + Task
                 ▼
          ┌────────────┐
          │ Quantized  │
          │  LLM (Q8)  │
          └─────┬──────┘
                ▼
          Suggested Fix
⚙️ Features
✅ File-aware retrieval (e.g., query: “Modify testb.py…” limits scope)

✅ Automatic reranking via cross-encoder/ms-marco-MiniLM-L-6-v2

✅ Support for 8-bit quantized models using BitsAndBytesConfig

✅ On-device inference (tested on RTX 3060, 6GB VRAM)

✅ Token-aware trimming and output parsing

✅ Streamlit UI with live input, output, and cache clearing

🖥️ UI Snapshot

🧪 Example Prompt
Input:

Modify testb.py so it uses Meta’s LLaMA LLM instead of Deepseek

Generated Fix:

python
Copy
Edit
model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
📦 Setup Instructions
🧰 Requirements
Python 3.9+

CUDA-enabled GPU (recommended: RTX 3060+)

Packages:

bash
Copy
Edit
pip install torch torchvision
pip install sentence-transformers faiss-cpu transformers accelerate bitsandbytes
pip install streamlit scikit-learn

▶️ Run the App
bash
Copy
Edit
streamlit run try.py

🧠 Models Used
Component	Model Name
Embedding Model	all-MiniLM-L6-v2 (SentenceTransformer)
Reranker	cross-encoder/ms-marco-MiniLM-L-6-v2
LLM	deepseek-ai/deepseek-coder-1.3b-instruct (or LLaMA/Mistral/etc)


🧾 License
MIT License — feel free to fork, modify, and extend.
