"""Configuration management for the AI coding assistant."""
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Qdrant Cloud (Optional - for advanced users)
    # Sign up at https://cloud.qdrant.io for free tier
    QDRANT_URL = os.getenv("QDRANT_URL", "")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
    QDRANT_COLLECTION = "code_embeddings"
    
    # Models
    MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-ai/deepseek-coder-1.3b-instruct")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    
    # LoRA Configuration
    LORA_R = 8
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.05
    LORA_TARGET_MODULES = ["q_proj", "v_proj"]
    
    # Performance
    USE_GPU_OFFLOAD = os.getenv("USE_GPU_OFFLOAD", "true").lower() == "true"
    ENABLE_QUANTIZATION = os.getenv("ENABLE_QUANTIZATION", "true").lower() == "true"
    MAX_CONTEXT_LENGTH = int(os.getenv("MAX_CONTEXT_LENGTH", "2048"))
    
    # Vector Store Selection
    # Default: FAISS (no setup required)
    # Optional: Qdrant Cloud (requires credentials)
    USE_QDRANT = bool(QDRANT_URL and QDRANT_API_KEY)
    FAISS_INDEX_PATH = "code_index"
