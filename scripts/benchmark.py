"""Benchmark script to measure and validate performance improvements."""
import time
import json
import torch
from pathlib import Path
import sys
sys.path.append('..')
from app.lora_model import OptimizedModelLoader
from app.config import Config


def benchmark_inference(model, tokenizer, prompts, num_runs=10):
    """Benchmark model inference speed."""
    latencies = []
    tokens_generated = []
    
    for prompt in prompts:
        for _ in range(num_runs):
            start = time.time()
            
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=1024,
                    temperature=0.3,
                    num_return_sequences=1,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
            
            latency = time.time() - start
            latencies.append(latency)
            tokens_generated.append(outputs.shape[1])
    
    return {
        "avg_latency": sum(latencies) / len(latencies),
        "min_latency": min(latencies),
        "max_latency": max(latencies),
        "avg_tokens": sum(tokens_generated) / len(tokens_generated),
        "tokens_per_second": sum(tokens_generated) / sum(latencies)
    }


def run_benchmark():
    """Run comprehensive benchmark."""
    test_prompts = [
        "Fix the bug in this Python function that calculates factorial",
        "Optimize this code for better performance",
        "Add error handling to this database query function"
    ]
    
    results = {}
    
    # Benchmark 1: Base model without optimizations
    print("Benchmarking base model (no optimizations)...")
    Config.ENABLE_QUANTIZATION = False
    Config.USE_GPU_OFFLOAD = False
    model_base, tokenizer = OptimizedModelLoader.load_model()
    results["base"] = benchmark_inference(model_base, tokenizer, test_prompts, num_runs=5)
    del model_base
    torch.cuda.empty_cache()
    
    # Benchmark 2: With 8-bit quantization
    print("Benchmarking with 8-bit quantization...")
    Config.ENABLE_QUANTIZATION = True
    Config.USE_GPU_OFFLOAD = False
    model_quant, tokenizer = OptimizedModelLoader.load_model()
    results["quantized"] = benchmark_inference(model_quant, tokenizer, test_prompts, num_runs=5)
    del model_quant
    torch.cuda.empty_cache()
    
    # Benchmark 3: With quantization + GPU offloading
    print("Benchmarking with quantization + GPU offloading...")
    Config.ENABLE_QUANTIZATION = True
    Config.USE_GPU_OFFLOAD = True
    model_opt, tokenizer = OptimizedModelLoader.load_model()
    results["optimized"] = benchmark_inference(model_opt, tokenizer, test_prompts, num_runs=5)
    
    # Calculate improvements
    baseline = results["base"]["avg_latency"]
    optimized = results["optimized"]["avg_latency"]
    improvement = ((baseline - optimized) / baseline) * 100
    
    results["improvement_percentage"] = improvement
    
    # Save results
    output_file = "benchmark_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "="*50)
    print("BENCHMARK RESULTS")
    print("="*50)
    print(f"Base Model Latency: {results['base']['avg_latency']:.3f}s")
    print(f"Optimized Latency: {results['optimized']['avg_latency']:.3f}s")
    print(f"Improvement: {improvement:.1f}%")
    print(f"Tokens/sec (base): {results['base']['tokens_per_second']:.1f}")
    print(f"Tokens/sec (optimized): {results['optimized']['tokens_per_second']:.1f}")
    print(f"\n✅ Results saved to {output_file}")


if __name__ == "__main__":
    run_benchmark()
