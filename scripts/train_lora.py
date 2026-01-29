"""LoRA fine-tuning script for code generation tasks."""
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import sys
sys.path.append('..')
from app.config import Config
from app.lora_model import OptimizedModelLoader


def prepare_training_data():
    """Load and prepare training dataset."""
    # Example: Load code dataset (replace with your own)
    dataset = load_dataset("codeparrot/github-code", split="train", streaming=True)
    dataset = dataset.take(10000)  # Limit for demo
    
    def format_instruction(example):
        return {
            "text": f"### Task: Fix the following code\n### Code:\n{example['code']}\n### Fixed Code:\n"
        }
    
    dataset = dataset.map(format_instruction)
    return dataset


def train_lora_model(output_dir="./lora_checkpoints"):
    """Train LoRA adapter on code dataset."""
    print("Loading base model...")
    model, tokenizer = OptimizedModelLoader.load_model()
    
    # Prepare for training
    model = prepare_model_for_kbit_training(model)
    
    # LoRA configuration
    lora_config = LoraConfig(
        r=Config.LORA_R,
        lora_alpha=Config.LORA_ALPHA,
        target_modules=Config.LORA_TARGET_MODULES,
        lora_dropout=Config.LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=True,
        save_steps=100,
        logging_steps=10,
        optim="paged_adamw_8bit",
        warmup_steps=50,
        save_total_limit=3,
    )
    
    # Load dataset
    print("Loading training data...")
    dataset = prepare_training_data()
    
    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        peft_config=lora_config,
        dataset_text_field="text",
        max_seq_length=512,
        tokenizer=tokenizer,
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save final model
    model.save_pretrained(f"{output_dir}/final")
    tokenizer.save_pretrained(f"{output_dir}/final")
    
    print(f"✅ Training complete! Model saved to {output_dir}/final")


if __name__ == "__main__":
    train_lora_model()
