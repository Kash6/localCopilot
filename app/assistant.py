# Load code files
import gc
import torch
import re
gc.collect()
torch.cuda.empty_cache()

import streamlit as st
import os
import faiss                      # For FAISS indexing
from sentence_transformers import SentenceTransformer  # For creating text/code embeddings
from sentence_transformers import CrossEncoder

# Code-aware model from Hugging Face:

def load_code_files(directory):
    files = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith(('.py', '.java', '.js')):
                file_path = os.path.join(root, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        files.append((filename, f.read()))
                except Exception as e:
                    print(f"Error reading file {file_path}: {e}")
    print(f"Number of files loaded: {len(files)}")
    return files

# Create embeddings
def create_embeddings(files, model_name='all-MiniLM-L6-v2'):
    if not files:
        raise ValueError("No files to create embeddings for. Check your input directory.")
    model = SentenceTransformer(model_name, device='cpu')
    texts = [content for _, content in files]
    print(f"Number of texts to encode: {len(texts)}")
    embeddings = model.encode(texts)
    print(f"Generated embeddings shape: {embeddings.shape}")
    return embeddings, files

# Save to FAISS
def save_to_faiss(embeddings, files, index_path='code_index'):
    if embeddings.ndim != 2:
        raise ValueError(f"Expected embeddings to be a 2D array, but got shape" + embeddings.shape)
    index = faiss.IndexFlatL2(embeddings.shape[1])  # Ensure dimension matches
    index.add(embeddings)
    faiss.write_index(index, index_path)
    print(f"FAISS index saved at {index_path}")
    return index



def retrieve_code_snippets(query, files, index_path='code_index', k=5, model_name='all-MiniLM-L6-v2'):
    # Try to extract specific file mentions
    mentioned_files = re.findall(r'\b[\w\-]+\.py\b', query)

    if mentioned_files:
        matches = [f for f in files if f[0] in mentioned_files]
        if matches:
            print(f"Retrieved by filename match: {mentioned_files}")
            return matches

    # Otherwise, use FAISS + reranker
    model = SentenceTransformer(model_name)
    query_embedding = model.encode([query])
    index = faiss.read_index(index_path)
    distances, indices = index.search(query_embedding, k)
    retrieved = [files[i] for i in indices[0]]

    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    pairs = [(query, content) for _, content in retrieved]
    scores = reranker.predict(pairs)
    reranked = sorted(zip(scores, retrieved), key=lambda x: x[0], reverse=True)
    return [item[1] for item in reranked[:k]]




from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Load the Llama model and tokenizer
model_name = "deepseek-ai/deepseek-coder-1.3b-instruct"  # Replace with the desired Llama model name
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_skip_modules=None,
    llm_int8_enable_fp32_cpu_offload=True
)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=bnb_config
)

from accelerate import infer_auto_device_map
device_map = infer_auto_device_map(model)
print("Model device map:", device_map)

# Generate fixes
def suggest_fix(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    outputs = model.generate(
    **inputs,
    max_length=2048,
    temperature=0.3,
    num_return_sequences=1,
    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    eos_token_id=tokenizer.eos_token_id
)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    fix_only = response.split("--- Modified Code ---")[-1].strip()
    return fix_only



# RAG pipeline
def rag_pipeline(query, files, index_path, model, tokenizer):
    snippets = retrieve_code_snippets(query, files, index_path)

    # Only include the content of the matched file(s)
    context = "\n".join([f"{name}:\n{code}" for name, code in snippets])

    prompt = f"""You are an expert software engineer. Modify the given code as requested below.

--- Task ---
{query}

--- Original Code ---
{context}

--- Modified Code ---
```python
"""

    return suggest_fix(prompt)



# Main Streamlit app
st.title("Coding assistant for Repository in RAG")

st.sidebar.header("Input Parameters")
repo_path = st.sidebar.text_input("Code Repository Path", value="../data")
query = st.sidebar.text_area("Enter Query", value="Modify decision_tree_functions.py file in section 1.2 in the classify_data class and change the variable unique_classes to different_classes")
output_path = st.sidebar.text_input("Output File Path", value="decision_tree_functions2.py")
jira_ticket= st.sidebar.text_input("Jira ticket ID:", value="25341")
if st.sidebar.button("Generate Fix"):
    st.write("Loading files...")
    code_files = load_code_files(repo_path)
    embeddings, files = create_embeddings(code_files)
    save_to_faiss(embeddings, files)

   

    st.write("Generating fix suggestions...")
    fix_suggestions = rag_pipeline(query, files, 'code_index', model, tokenizer)

    st.write("Generated Fix Suggestions:")
    st.code(fix_suggestions)

    # Save suggestions to the output file
    with open(output_path, 'w') as f:
        f.write(fix_suggestions)

    st.success(f"Fix suggestions saved to {output_path}")

if os.path.exists("code_index"):
    last_modified = os.path.getmtime("code_index")
    from datetime import datetime
    st.sidebar.caption(f"Last indexed: {datetime.fromtimestamp(last_modified).strftime('%Y-%m-%d %H:%M:%S')}")

if st.sidebar.button("Clear RAG Cache"):
    if os.path.exists("code_index"):
        os.remove("code_index")
        st.sidebar.success("FAISS cache cleared.")
    else:
        st.sidebar.info("No cache to clear.")


