"""
Clean up unsafe .pt/.pth/.bin files from existing checkpoints
"""

from pathlib import Path
import sys

def clean_checkpoint(checkpoint_dir: Path, dry_run=True):
    """Remove unsafe PyTorch files from a checkpoint"""
    
    removed_files = []
    
    # Remove unsafe .pt, .pth, and .bin files
    patterns = ["*.pt", "*.pth", "*.bin"]
    
    # Keep training_args.bin as it's metadata, not model state
    exclude = ["training_args.bin"]
    
    for pattern in patterns:
        for file in checkpoint_dir.glob(pattern):
            if file.name in exclude:
                continue
                
            if dry_run:
                removed_files.append(file.name)
            else:
                file.unlink()
                removed_files.append(file.name)
                print(f"   ✓ Deleted: {file.name}")
    
    return removed_files


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Clean unsafe files from checkpoints")
    parser.add_argument("--execute", action="store_true", 
                       help="Actually delete files (default is dry-run)")
    args = parser.parse_args()
    
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
    title = "DRY RUN: Preview files to delete" if not args.execute else "EXECUTING: Deleting unsafe files"
    print(title.center(70))
    print("=" * 70)
    print()
    
    total_removed = 0
    
    for checkpoint_dir in checkpoint_dirs:
        print(f"📂 {checkpoint_dir.name}")
        
        removed = clean_checkpoint(checkpoint_dir, dry_run=not args.execute)
        
        if removed:
            for file in removed:
                print(f"   {'Would delete' if not args.execute else 'Deleted'}: {file}")
            total_removed += len(removed)
        else:
            print(f"   ✓ No unsafe files found")
        
        print()
    
    print("=" * 70)
    
    if total_removed > 0:
        if args.execute:
            print(f"✅ Deleted {total_removed} unsafe file(s)")
        else:
            print(f"⚠️  Would delete {total_removed} file(s)")
            print()
            print("To actually delete these files, run:")
            print("  python cleanup_checkpoints.py --execute")
    else:
        print("✅ No unsafe files found - all checkpoints are clean!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
