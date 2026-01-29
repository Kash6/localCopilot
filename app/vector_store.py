"""Unified vector store interface: FAISS (default) + optional Qdrant Cloud."""
import faiss
import numpy as np
from typing import List, Tuple
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from sentence_transformers import SentenceTransformer
from app.config import Config


class VectorStore:
    """Vector storage with FAISS (default) and optional Qdrant Cloud upgrade."""
    
    def __init__(self, model_name: str = Config.EMBEDDING_MODEL):
        self.model = SentenceTransformer(model_name, device='cpu')
        self.use_qdrant = Config.USE_QDRANT
        
        if self.use_qdrant:
            try:
                from qdrant_client import QdrantClient
                from qdrant_client.models import Distance, VectorParams
                
                # Connect to Qdrant Cloud
                self.client = QdrantClient(
                    url=Config.QDRANT_URL,
                    api_key=Config.QDRANT_API_KEY
                )
                self._init_qdrant_collection()
                print(f"✅ Connected to Qdrant Cloud")
            except Exception as e:
                print(f"⚠️  Qdrant connection failed: {e}")
                print("   Falling back to FAISS...")
                self.use_qdrant = False
                self.index = None
                self.index_path = Config.FAISS_INDEX_PATH
        else:
            self.index = None
            self.index_path = Config.FAISS_INDEX_PATH
    
    def _init_qdrant_collection(self):
        """Initialize Qdrant collection if it doesn't exist."""
        from qdrant_client.models import Distance, VectorParams
        
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if Config.QDRANT_COLLECTION not in collection_names:
            self.client.create_collection(
                collection_name=Config.QDRANT_COLLECTION,
                vectors_config=VectorParams(
                    size=self.model.get_sentence_embedding_dimension(),
                    distance=Distance.COSINE
                )
            )
    
    def add_documents(self, files: List[Tuple[str, str]]):
        """Add documents to vector store."""
        texts = [content for _, content in files]
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        if self.use_qdrant:
            from qdrant_client.models import PointStruct
            
            points = [
                PointStruct(
                    id=idx,
                    vector=embedding.tolist(),
                    payload={"filename": filename, "content": content}
                )
                for idx, (embedding, (filename, content)) in enumerate(zip(embeddings, files))
            ]
            self.client.upsert(
                collection_name=Config.QDRANT_COLLECTION,
                points=points
            )
            print(f"✅ Indexed {len(files)} files to Qdrant Cloud")
        else:
            # FAISS (default)
            if embeddings.ndim != 2:
                raise ValueError(f"Expected 2D embeddings, got shape {embeddings.shape}")
            self.index = faiss.IndexFlatL2(embeddings.shape[1])
            self.index.add(embeddings.astype('float32'))
            faiss.write_index(self.index, self.index_path)
            print(f"✅ Indexed {len(files)} files to FAISS")
        
        return embeddings, files
    
    def search(self, query: str, k: int = 5) -> List[Tuple[str, str]]:
        """Search for similar documents."""
        query_embedding = self.model.encode([query])
        
        if self.use_qdrant:
            results = self.client.search(
                collection_name=Config.QDRANT_COLLECTION,
                query_vector=query_embedding[0].tolist(),
                limit=k
            )
            return [(r.payload["filename"], r.payload["content"]) for r in results]
        else:
            # FAISS (default)
            if self.index is None:
                self.index = faiss.read_index(self.index_path)
            _, indices = self.index.search(query_embedding.astype('float32'), k)
            # Note: This requires files to be stored separately
            return indices[0].tolist()
