"""
Verify that checkpoints are CVE-2025-32434 safe (no .pt/.pth files)
"""

from pathlib import Path
import sys

def check_checkpoint_safety(checkpoint_dir: Path):
    """Check if a checkpoint contains unsafe PyTorch files"""
    
    unsafe_files = []
    safe_files = []
    
    # Check for unsafe .pt and .pth files
    for pattern in ["*.pt", "*.pth", "*.bin"]:
        for file in checkpoint_dir.glob(pattern):
            unsafe_files.append(file.name)
    
    # Check for safe .safetensors files
    for file in checkpoint_dir.glob("*.safetensors"):
        safe_files.append(file.name)
    
    return unsafe_files, safe_files


def main():
    output_dir = Path("models/fine_tuned")
    
    if not output_dir.exists():
        print(f"❌ Output directory not found: {output_dir}")
        return 1
    
    # Find all checkpoints
    checkpoint_dirs = sorted(output_dir.glob("checkpoint-*"))
    
    if not checkpoint_dirs:
        print("⚠️  No checkpoints found")
        return 0
    
    print("=" * 70)
    print("CHECKPOINT SAFETY VERIFICATION".center(70))
    print("=" * 70)
    print()
    
    total_unsafe = 0
    
    for checkpoint_dir in checkpoint_dirs:
        print(f"📂 {checkpoint_dir.name}")
        
        unsafe_files, safe_files = check_checkpoint_safety(checkpoint_dir)
        
        if safe_files:
            print(f"   ✅ Safe files: {', '.join(safe_files)}")
        
        if unsafe_files:
            print(f"   ⚠️  UNSAFE FILES (CVE-2025-32434): {', '.join(unsafe_files)}")
            total_unsafe += len(unsafe_files)
        else:
            print(f"   ✓ No unsafe .pt/.pth/.bin files")
        
        print()
    
    print("=" * 70)
    
    if total_unsafe > 0:
        print(f"❌ FAILED: Found {total_unsafe} unsafe file(s)")
        print()
        print("To fix:")
        print("  1. Delete existing checkpoints with unsafe files")
        print("  2. Ensure save_only_model=True in TrainingArguments")
        print("  3. Restart training to create safe checkpoints")
        return 1
    else:
        print("✅ PASSED: All checkpoints are safe!")
        print()
        print("Your checkpoints use safetensors format and are protected against")
        print("CVE-2025-32434 (arbitrary code execution via malicious pickled files).")
        return 0


if __name__ == "__main__":
    sys.exit(main())
