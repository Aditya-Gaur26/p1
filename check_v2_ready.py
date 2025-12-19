"""
Verify V2 training prerequisites
"""

from pathlib import Path
import json

def check_v2_prerequisites():
    """Check if everything is ready for V2 training"""
    
    print("=" * 70)
    print("V2 TRAINING PREREQUISITES CHECK".center(70))
    print("=" * 70)
    print()
    
    checks = []
    
    # Check V1 model
    v1_model = Path("models/fine_tuned/adapter_model.safetensors")
    if v1_model.exists():
        print("✅ V1 Model found: models/fine_tuned")
        checks.append(True)
    else:
        print("❌ V1 Model NOT found: models/fine_tuned")
        print("   → Run V1 training first: python src/training/fine_tune.py")
        checks.append(False)
    
    # Check V2 training data
    train_file = Path("data/processed/train_finetune2.jsonl")
    if train_file.exists():
        lines = sum(1 for _ in open(train_file, encoding='utf-8'))
        print(f"✅ V2 Train data: {lines:,} samples")
        checks.append(True)
    else:
        print("❌ V2 Train data NOT found: data/processed/train_finetune2.jsonl")
        checks.append(False)
    
    # Check V2 validation data
    val_file = Path("data/processed/val_finetune2.jsonl")
    if val_file.exists():
        lines = sum(1 for _ in open(val_file, encoding='utf-8'))
        print(f"✅ V2 Val data: {lines:,} samples")
        checks.append(True)
    else:
        print("❌ V2 Val data NOT found: data/processed/val_finetune2.jsonl")
        checks.append(False)
    
    # Check output directory doesn't exist (fresh start)
    output_dir = Path("models/fine_tuned_v2")
    if output_dir.exists():
        checkpoints = list(output_dir.glob("checkpoint-*"))
        if checkpoints:
            print(f"⚠️  V2 output exists with {len(checkpoints)} checkpoint(s)")
            print(f"   → Training will resume from latest checkpoint")
        else:
            print(f"⚠️  V2 output directory exists but empty")
    else:
        print("✅ V2 output directory ready (will be created)")
    
    print()
    print("=" * 70)
    
    if all(checks):
        print("✅ ALL CHECKS PASSED - Ready for V2 training!".center(70))
        print()
        print("To start V2 training:")
        print("  powershell -ExecutionPolicy Bypass -File train_v2.ps1")
        print("  OR")
        print("  python src/training/fine_tune_v2.py")
        return 0
    else:
        print("❌ CHECKS FAILED - Fix issues above before training".center(70))
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(check_v2_prerequisites())
