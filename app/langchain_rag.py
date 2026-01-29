"""LangChain-based RAG pipeline for code generation."""
from typing import List, Tuple, Any, Optional
import re
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from langchain_core.prompts import PromptTemplate
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from sentence_transformers import CrossEncoder
from app.config import Config


class HuggingFaceLLM(LLM):
    """Custom LangChain LLM wrapper for HuggingFace models."""
    
    model: Any = None
    tokenizer: Any = None
    
    @property
    def _llm_type(self) -> str:
        return "huggingface"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any
    ) -> str:
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=Config.MAX_CONTEXT_LENGTH
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=512,  # Generate up to 512 new tokens (not total length)
            temperature=0.7,
            do_sample=True,
            top_p=0.95,
            num_return_sequences=1,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response


class LangChainRAGPipeline:
    """RAG pipeline using LangChain for orchestration."""
    
    def __init__(self, model, tokenizer, vector_store, files_cache):
        self.vector_store = vector_store
        self.files_cache = files_cache
        self.reranker = CrossEncoder(Config.RERANKER_MODEL)
        
        # Create LangChain LLM wrapper
        self.llm = HuggingFaceLLM(model=model, tokenizer=tokenizer)
        
        # Define prompt template
        self.prompt_template = PromptTemplate(
            input_variables=["task", "context"],
            template="""You are an expert software engineer. Modify the given code as requested below.

--- Task ---
{task}

--- Original Code ---
{context}

--- Modified Code ---
```python
"""
        )
    
    def retrieve_with_reranking(self, query: str, k: int = 5) -> List[Tuple[str, str]]:
        """Retrieve and rerank code snippets."""
        # Check for explicit file mentions
        mentioned_files = re.findall(r'\b[\w\-]+\.py\b', query)
        
        if mentioned_files:
            matches = [f for f in self.files_cache if f[0] in mentioned_files]
            if matches:
                return matches
        
        # Vector search
        if Config.USE_QDRANT:
            retrieved = self.vector_store.search(query, k=k)
        else:
            indices = self.vector_store.search(query, k=k)
            retrieved = [self.files_cache[i] for i in indices if i < len(self.files_cache)]
        
        # Rerank results
        pairs = [(query, content) for _, content in retrieved]
        scores = self.reranker.predict(pairs)
        reranked = sorted(zip(scores, retrieved), key=lambda x: x[0], reverse=True)
        
        return [item[1] for item in reranked[:k]]
    
    def generate_fix(self, query: str) -> str:
        """Generate code fix using RAG pipeline."""
        snippets = self.retrieve_with_reranking(query)
        context = "\n".join([f"{name}:\n{code}" for name, code in snippets])
        
        # Format prompt using template
        prompt = self.prompt_template.format(task=query, context=context)
        
        # Use LLM to generate response
        response = self.llm.invoke(prompt)
        
        # Extract code from response
        fix_only = response.split("--- Modified Code ---")[-1].strip()
        if "```python" in fix_only:
            fix_only = fix_only.split("```python")[1].split("```")[0].strip()
        
        return fix_only
