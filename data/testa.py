from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
print("Model loaded")
texts = ["def foo(): return 1", "class A: pass"]
embeddings = model.encode(texts)
print("Embedding done")
