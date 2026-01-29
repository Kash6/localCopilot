"""LoRA-tuned model loading and inference optimization."""
import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer
from app.config import Config

# Optional imports for GPU features
try:
    from transformers import BitsAndBytesConfig
    QUANTIZATION_AVAILABLE = True
except ImportError:
    QUANTIZATION_AVAILABLE = False

try:
    from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training
    LORA_AVAILABLE = True
except ImportError:
    LORA_AVAILABLE = False


class OptimizedModelLoader:
    """Load and optimize LLM with quantization, GPU offloading, and LoRA."""
    
    @staticmethod
    def load_model(model_name: str = Config.MODEL_NAME, use_lora: bool = False):
        """Load model with optimizations."""
        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Configure 8-bit quantization (only if GPU available and bitsandbytes installed)
        bnb_config = None
        if Config.ENABLE_QUANTIZATION and device == "cuda" and QUANTIZATION_AVAILABLE:
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_skip_modules=None,
                llm_int8_enable_fp32_cpu_offload=True
            )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto" if device == "cuda" else None,
            quantization_config=bnb_config,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        # Move to device if not using device_map
        if device == "cpu":
            model = model.to(device)
        
        # Apply LoRA if requested (only on GPU with LORA available)
        if use_lora and device == "cuda" and LORA_AVAILABLE:
            model = OptimizedModelLoader._apply_lora(model)
        
        return model, tokenizer
    
    @staticmethod
    def _apply_lora(model):
        """Apply LoRA configuration to model."""
        if not LORA_AVAILABLE:
            return model
            
        # Prepare model for k-bit training
        model = prepare_model_for_kbit_training(model)
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=Config.LORA_R,
            lora_alpha=Config.LORA_ALPHA,
            target_modules=Config.LORA_TARGET_MODULES,
            lora_dropout=Config.LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        # Apply LoRA
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        return model
    
    @staticmethod
    def load_lora_checkpoint(model, checkpoint_path: str):
        """Load fine-tuned LoRA weights."""
        if not LORA_AVAILABLE:
            return model
        model = PeftModel.from_pretrained(model, checkpoint_path)
        return model
