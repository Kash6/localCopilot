"""Enhanced AI Coding Assistant with LangChain, Qdrant, and LoRA support."""
import gc
import torch
import streamlit as st
import os
from datetime import datetime

gc.collect()
torch.cuda.empty_cache()

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from app.config import Config
from app.vector_store import VectorStore
from app.langchain_rag import LangChainRAGPipeline
from app.lora_model import OptimizedModelLoader
from app.performance_tracker import PerformanceTracker


# Initialize performance tracker
if 'perf_tracker' not in st.session_state:
    st.session_state.perf_tracker = PerformanceTracker()


def load_code_files(directory):
    """Load code files from directory."""
    files = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith(('.py', '.java', '.js', '.ts', '.jsx', '.tsx')):
                file_path = os.path.join(root, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        files.append((filename, f.read()))
                except Exception as e:
                    st.warning(f"Error reading {file_path}: {e}")
    return files


@st.cache_resource
def initialize_models(use_lora=False):
    """Initialize and cache models."""
    with st.spinner("Loading optimized models..."):
        model, tokenizer = OptimizedModelLoader.load_model(use_lora=use_lora)
        
        # Display actual device mapping
        if hasattr(model, 'hf_device_map'):
            device_info = list(set(model.hf_device_map.values()))
        else:
            # Fallback: check first parameter device
            device_info = [str(next(model.parameters()).device)]
        st.sidebar.success(f"Model loaded on: {device_info}")
        
        return model, tokenizer


@st.cache_resource
def initialize_vector_store():
    """Initialize vector store (FAISS by default, Qdrant Cloud optional)."""
    vector_store = VectorStore()
    if Config.USE_QDRANT:
        store_type = "Qdrant Cloud"
    else:
        store_type = "FAISS (Local)"
    st.sidebar.info(f"Vector Store: {store_type}")
    return vector_store


def main():
    st.set_page_config(
        page_title="AI Coding Assistant",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for professional dark theme
    st.markdown("""
        <style>
        /* Main background */
        .main {
            background-color: #0E1117;
        }
        
        /* Input fields */
        .stTextInput > div > div > input {
            background-color: #1E1E1E;
            color: #E6EDF3;
            border: 1px solid #30363D;
            border-radius: 6px;
        }
        
        .stTextArea > div > div > textarea {
            background-color: #1E1E1E;
            color: #E6EDF3;
            border: 1px solid #30363D;
            border-radius: 6px;
            font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
        }
        
        /* Buttons */
        .stButton > button {
            background-color: #238636;
            color: white;
            border: none;
            border-radius: 6px;
            padding: 0.5rem 1rem;
            font-weight: 600;
            transition: background-color 0.2s;
        }
        
        .stButton > button:hover {
            background-color: #2EA043;
        }
        
        /* Headers */
        h1 {
            color: #E6EDF3;
            font-weight: 600;
            letter-spacing: -0.5px;
        }
        
        h2, h3 {
            color: #E6EDF3;
            font-weight: 500;
        }
        
        /* Sidebar */
        .css-1d391kg {
            background-color: #161B22;
        }
        
        /* Code blocks */
        .stCodeBlock {
            background-color: #161B22;
            border: 1px solid #30363D;
            border-radius: 6px;
        }
        
        /* Metrics */
        .stMetric {
            background-color: #161B22;
            border: 1px solid #30363D;
            border-radius: 6px;
            padding: 1rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("AI-Powered Coding Assistant")
    st.caption("RAG-based code generation with LangChain, Qdrant Cloud, and LoRA-tuned LLMs")
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    repo_path = st.sidebar.text_input("Code Repository Path", value="./data")
    use_lora = st.sidebar.checkbox("Use LoRA-tuned Model", value=False)
    
    # Initialize models
    model, tokenizer = initialize_models(use_lora=use_lora)
    vector_store = initialize_vector_store()
    
    # Performance metrics display
    st.sidebar.header("Performance Metrics")
    stats = st.session_state.perf_tracker.get_summary_stats()
    if stats:
        improvement = stats.get("improvement", {})
        if improvement.get("improvement_percentage"):
            st.sidebar.metric(
                "Latency Improvement",
                f"{improvement['improvement_percentage']:.1f}%",
                delta=f"{improvement['improvement_percentage']:.1f}%"
            )
        st.sidebar.metric("Avg Latency", f"{stats.get('avg_latency', 0):.2f}s")
        st.sidebar.metric("Total Queries", stats.get('total_queries', 0))
    
    # Main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Input")
        query = st.text_area(
            "Enter your code modification request:",
            value="Modify decision_tree_functions.py in section 1.2 classify_data and change variable unique_classes to different_classes",
            height=150
        )
        
        jira_ticket = st.text_input("JIRA Ticket ID (optional):", value="")
        output_path = st.text_input("Output File Path:", value="output_fix.py")
        
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            generate_btn = st.button("Generate Fix", type="primary", use_container_width=True)
        with col_btn2:
            clear_cache_btn = st.button("Clear Cache", use_container_width=True)
    
    with col2:
        st.subheader("Generated Output")
        output_container = st.empty()
    
    # Generate fix
    if generate_btn:
        if not query.strip():
            st.error("Please enter a query")
            return
        
        # Start performance tracking
        start_time = st.session_state.perf_tracker.start_query()
        
        with st.spinner("Loading and indexing code files..."):
            code_files = load_code_files(repo_path)
            if not code_files:
                st.error(f"No code files found in {repo_path}")
                return
            
            st.info(f"Loaded {len(code_files)} files")
            
            # Index files
            embeddings, files = vector_store.add_documents(code_files)
        
        with st.spinner("Generating fix with RAG pipeline..."):
            # Initialize RAG pipeline
            rag_pipeline = LangChainRAGPipeline(
                model=model,
                tokenizer=tokenizer,
                vector_store=vector_store,
                files_cache=files
            )
            
            # Generate fix
            fix_suggestions = rag_pipeline.generate_fix(query)
        
        # End performance tracking
        tokens_generated = len(tokenizer.encode(fix_suggestions))
        metric = st.session_state.perf_tracker.end_query(start_time, query, tokens_generated)
        
        # Display results
        with output_container.container():
            st.code(fix_suggestions, language="python")
            
            # Performance info
            st.caption(f"Generated in {metric['latency_seconds']:.2f}s | "
                      f"{metric['tokens_per_second']:.1f} tokens/sec")
        
        # Save to file
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(fix_suggestions)
            st.success(f"Fix saved to {output_path}")
            
            # Add metadata comment
            if jira_ticket:
                st.info(f"Associated with JIRA ticket: {jira_ticket}")
        except Exception as e:
            st.error(f"Error saving file: {e}")
        
        # Save metrics
        st.session_state.perf_tracker.save_metrics()
    
    # Clear cache
    if clear_cache_btn:
        if Config.USE_QDRANT:
            try:
                vector_store = initialize_vector_store()
                vector_store.client.delete_collection(Config.QDRANT_COLLECTION)
                st.success("Qdrant Cloud collection cleared")
            except Exception as e:
                st.warning(f"Could not clear Qdrant: {e}")
        else:
            if os.path.exists(Config.FAISS_INDEX_PATH):
                os.remove(Config.FAISS_INDEX_PATH)
                st.success("FAISS cache cleared")
        
        st.cache_resource.clear()
        st.rerun()
    
    # Footer with tech stack
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Tech Stack")
    st.sidebar.markdown(f"""
    - **LLM**: {Config.MODEL_NAME.split('/')[-1]}
    - **Embeddings**: {Config.EMBEDDING_MODEL}
    - **Vector DB**: {'Qdrant Cloud' if Config.USE_QDRANT else 'FAISS'}
    - **Framework**: LangChain
    - **Optimization**: 8-bit Quantization + {'LoRA' if use_lora else 'Base'}
    """)


if __name__ == "__main__":
    main()
