"""
Memory Optimization Utilities for Large Dataset Training
Helps RTX 3060 12GB handle full 1,637 pages / 4,000+ Q&A pairs
"""

import torch
import gc
import os
from pathlib import Path


def optimize_memory():
    """Apply system-wide memory optimizations"""
    
    # PyTorch optimizations
    torch.backends.cudnn.benchmark = True  # Faster convolutions
    torch.backends.cuda.matmul.allow_tf32 = True  # Faster matmul
    
    # Garbage collection
    gc.collect()
    torch.cuda.empty_cache()
    
    # Set memory allocator
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    
    print("✅ Memory optimizations applied")


def check_gpu_memory():
    """Check available GPU memory"""
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        total = torch.cuda.get_device_properties(device).total_memory / 1e9
        allocated = torch.cuda.memory_allocated(device) / 1e9
        reserved = torch.cuda.memory_reserved(device) / 1e9
        free = total - reserved
        
        print(f"\n🎮 GPU: {torch.cuda.get_device_name(device)}")
        print(f"   Total:     {total:.2f} GB")
        print(f"   Allocated: {allocated:.2f} GB")
        print(f"   Reserved:  {reserved:.2f} GB")
        print(f"   Free:      {free:.2f} GB")
        
        if free < 2:
            print("⚠️  WARNING: Less than 2GB free, may encounter OOM")
        
        return free
    else:
        print("❌ No GPU available")
        return 0


def clear_memory():
    """Force clear GPU memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    print("✅ Memory cleared")


def estimate_training_memory(
    num_samples: int,
    batch_size: int = 1,
    seq_length: int = 1024,
    lora_r: int = 32
):
    """Estimate memory requirements"""
    
    # Model memory (4-bit quantized Qwen2.5-7B)
    model_memory = 3.5  # GB (7B params in 4-bit)
    
    # LoRA adapters
    lora_memory = (lora_r * 0.001)  # Roughly 32MB for r=32
    
    # Optimizer states (paged_adamw_8bit)
    optimizer_memory = lora_memory * 2  # ~2x LoRA size
    
    # Gradients with checkpointing
    gradient_memory = lora_memory * 0.5  # Reduced with checkpointing
    
    # Activations (per batch)
    activation_memory = (batch_size * seq_length * 0.000001)  # Very rough
    
    # Total
    total = model_memory + lora_memory + optimizer_memory + gradient_memory + activation_memory
    
    print(f"\n📊 Estimated Memory Usage:")
    print(f"   Model (4-bit):     {model_memory:.2f} GB")
    print(f"   LoRA adapters:     {lora_memory:.2f} GB")
    print(f"   Optimizer:         {optimizer_memory:.2f} GB")
    print(f"   Gradients:         {gradient_memory:.2f} GB")
    print(f"   Activations:       {activation_memory:.2f} GB")
    print(f"   ─────────────────────────────")
    print(f"   Total:             {total:.2f} GB")
    print(f"   Recommended VRAM:  {total * 1.2:.2f} GB (with 20% buffer)")
    
    return total


def reduce_dataset_memory(dataset_path: Path, max_samples: int = None):
    """Reduce dataset memory footprint"""
    import json
    
    print(f"\n📦 Processing dataset: {dataset_path}")
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    original_size = len(data)
    
    if max_samples and len(data) > max_samples:
        # Keep most diverse samples
        data = data[:max_samples]
        print(f"   Reduced from {original_size} to {len(data)} samples")
    
    # Remove unnecessary fields
    cleaned_data = []
    for item in data:
        cleaned_item = {
            'instruction': item.get('instruction', ''),
            'input': item.get('input', ''),
            'output': item.get('output', '')
        }
        cleaned_data.append(cleaned_item)
    
    # Save cleaned dataset
    output_path = dataset_path.parent / f"{dataset_path.stem}_cleaned.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=2)
    
    print(f"   ✅ Cleaned dataset saved to {output_path}")
    return output_path


def monitor_training_memory(interval: int = 10):
    """Monitor memory during training (run in separate thread)"""
    import time
    import threading
    
    def monitor():
        while True:
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1e9
                reserved = torch.cuda.memory_reserved() / 1e9
                print(f"[Memory] Allocated: {allocated:.2f}GB | Reserved: {reserved:.2f}GB", end='\r')
            time.sleep(interval)
    
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    print("✅ Memory monitoring started")


def get_optimal_config_for_gpu():
    """Get optimal training config based on available GPU"""
    
    if not torch.cuda.is_available():
        print("❌ No GPU available, using CPU config (slow)")
        return {
            'batch_size': 1,
            'seq_length': 512,
            'grad_accumulation': 32,
            'lora_r': 16
        }
    
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    print(f"\n🎮 Detected GPU with {total_memory:.1f}GB VRAM")
    
    if total_memory < 8:
        print("⚠️  Low VRAM - using minimal config")
        return {
            'batch_size': 1,
            'seq_length': 512,
            'grad_accumulation': 16,
            'lora_r': 16,
            'load_in_8bit': True
        }
    
    elif total_memory < 16:
        print("✅ Medium VRAM (RTX 3060) - optimized config")
        return {
            'batch_size': 1,
            'seq_length': 1024,
            'grad_accumulation': 16,
            'lora_r': 32,
            'load_in_4bit': True,
            'gradient_checkpointing': True
        }
    
    elif total_memory < 24:
        print("✅ High VRAM - comfortable config")
        return {
            'batch_size': 2,
            'seq_length': 1024,
            'grad_accumulation': 8,
            'lora_r': 32,
            'load_in_4bit': True,
            'gradient_checkpointing': True
        }
    
    else:
        print("✅ Very High VRAM - optimal config")
        return {
            'batch_size': 4,
            'seq_length': 2048,
            'grad_accumulation': 4,
            'lora_r': 64,
            'load_in_4bit': False,
            'gradient_checkpointing': False
        }


def main():
    """Test memory utilities"""
    print("=" * 60)
    print("Memory Optimization Utilities".center(60))
    print("=" * 60)
    
    # Apply optimizations
    optimize_memory()
    
    # Check GPU
    free_memory = check_gpu_memory()
    
    # Get optimal config
    config = get_optimal_config_for_gpu()
    print(f"\n📋 Recommended Configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    # Estimate memory for large dataset
    print("\n" + "=" * 60)
    estimate_training_memory(
        num_samples=4000,
        batch_size=config['batch_size'],
        seq_length=config['seq_length'],
        lora_r=config['lora_r']
    )
    
    # Tips
    print("\n" + "=" * 60)
    print("💡 Memory Optimization Tips:")
    print("   1. Use batch_size=1 with gradient_accumulation=16")
    print("   2. Set max_seq_length=1024 (not 2048)")
    print("   3. Enable gradient_checkpointing=true")
    print("   4. Use load_in_4bit=true (not 8bit)")
    print("   5. Use paged_adamw_8bit optimizer")
    print("   6. Close other GPU applications")
    print("   7. Monitor with: nvidia-smi -l 1")
    print("=" * 60)


if __name__ == "__main__":
    main()
