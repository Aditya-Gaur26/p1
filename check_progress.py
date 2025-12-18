"""Quick progress checker for dataset generation"""

import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))
from src.utils.config import config

def check_progress():
    train_file = config.data_dir / "processed" / "train_llm.jsonl"
    val_file = config.data_dir / "processed" / "val_llm.jsonl"
    temp_file = config.data_dir / "processed" / "train_llm_temp.jsonl"
    
    train_count = 0
    val_count = 0
    temp_count = 0
    
    if train_file.exists():
        with open(train_file, 'r', encoding='utf-8') as f:
            train_count = sum(1 for line in f if line.strip())
    
    if val_file.exists():
        with open(val_file, 'r', encoding='utf-8') as f:
            val_count = sum(1 for line in f if line.strip())
    
    if temp_file.exists():
        with open(temp_file, 'r', encoding='utf-8') as f:
            temp_count = sum(1 for line in f if line.strip())
    
    total = train_count + val_count
    
    print(f"\n{'='*70}")
    print(f"DATASET GENERATION PROGRESS")
    print(f"{'='*70}")
    
    if temp_count > 0:
        print(f"IN PROGRESS - Questions generated: {temp_count}")
        print(f"Temp file: data\\processed\\train_llm_temp.jsonl")
        estimated_chunks = temp_count / 12  # Assuming 12 questions per chunk
        print(f"Estimated chunks processed: ~{estimated_chunks:.0f}/976")
        print(f"Progress: {(estimated_chunks/976)*100:.1f}%")
    elif total > 0:
        print(f"COMPLETED")
        print(f"Training samples: {train_count}")
        print(f"Validation samples: {val_count}")
        print(f"Total Q&A pairs: {total}")
    else:
        print(f"NOT STARTED or no data yet")
    
    print(f"\nExpected: ~10,000-12,000 questions from 976 chunks")
    print(f"{'='*70}\n")
    
    if total >= 10000:
        print("GENERATION COMPLETE! Ready to train.")
        print("Files: data\\processed\\train_llm.jsonl, val_llm.jsonl")
    elif temp_count > 0:
        print("Still generating... Questions are being saved continuously.")
        print("Safe to interrupt - progress is preserved!")
    else:
        print("Generation not started or files not created yet.")

if __name__ == "__main__":
    check_progress()
