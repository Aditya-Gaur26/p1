"""
Manual dataset generation helper - processes chunks in batches
Claude will generate questions for each batch through conversation
"""

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.config import config
from src.utils.helpers import load_json, save_jsonl, generate_id


def split_into_batches(batch_size=10):
    """Split book chunks into manageable batches for manual processing"""
    
    books_file = config.data_dir / "processed" / "books" / "all_pdfs_combined.json"
    data = load_json(books_file)
    
    all_chunks = []
    for file_data in data['files']:
        filename = file_data['filename']
        for i, chunk in enumerate(file_data['chunks'], 1):
            if len(chunk) > 200:  # Skip very short chunks
                all_chunks.append({
                    'chunk_id': f"{filename}_chunk_{i}",
                    'source': f"{filename} - Section {i}",
                    'text': chunk
                })
    
    # Create batch files
    batches_dir = config.data_dir / "processed" / "batches"
    batches_dir.mkdir(exist_ok=True)
    
    total_batches = (len(all_chunks) + batch_size - 1) // batch_size
    
    for batch_num in range(total_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, len(all_chunks))
        batch_chunks = all_chunks[start_idx:end_idx]
        
        batch_file = batches_dir / f"batch_{batch_num + 1:03d}.json"
        with open(batch_file, 'w', encoding='utf-8') as f:
            json.dump({
                'batch_number': batch_num + 1,
                'total_batches': total_batches,
                'chunks': batch_chunks
            }, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Created {total_batches} batches ({batch_size} chunks each)")
    print(f"📁 Saved to: {batches_dir}")
    print(f"\n📋 Total chunks: {len(all_chunks)}")
    print(f"📦 Batches: {total_batches}")
    print(f"📝 Chunks per batch: {batch_size}")
    
    return total_batches


def get_next_batch():
    """Get the next unprocessed batch"""
    batches_dir = config.data_dir / "processed" / "batches"
    responses_dir = config.data_dir / "processed" / "responses"
    responses_dir.mkdir(exist_ok=True)
    
    # Find all batch files
    batch_files = sorted(batches_dir.glob("batch_*.json"))
    
    for batch_file in batch_files:
        batch_num = batch_file.stem.split('_')[1]
        response_file = responses_dir / f"response_{batch_num}.jsonl"
        
        if not response_file.exists():
            # This batch needs processing
            with open(batch_file, 'r', encoding='utf-8') as f:
                batch_data = json.load(f)
            
            print(f"\n{'='*70}")
            print(f"📦 Batch {batch_data['batch_number']}/{batch_data['total_batches']}")
            print(f"{'='*70}\n")
            
            for i, chunk_data in enumerate(batch_data['chunks'], 1):
                print(f"\n--- Chunk {i}/{len(batch_data['chunks'])} ---")
                print(f"Source: {chunk_data['source']}")
                print(f"Text preview: {chunk_data['text'][:200]}...")
                print(f"Length: {len(chunk_data['text'])} chars\n")
            
            return batch_data, response_file
    
    print("✅ All batches processed!")
    return None, None


def save_response(response_file, qa_pairs):
    """Save generated Q&A pairs"""
    training_examples = []
    
    for qa in qa_pairs:
        example = {
            "id": generate_id(qa['question'] + qa.get('source', '')),
            "instruction": qa['question'],
            "input": "",
            "output": qa['answer'],
            "source": qa.get('source', 'unknown'),
            "question_type": qa.get('type', 'general'),
            "difficulty": qa.get('difficulty', 'medium'),
            "text": f"### Instruction:\n{qa['question']}\n\n### Response:\n{qa['answer']}"
        }
        training_examples.append(example)
    
    save_jsonl(training_examples, response_file)
    print(f"\n✅ Saved {len(training_examples)} Q&A pairs to {response_file}")


def combine_all_responses():
    """Combine all response files into final dataset"""
    responses_dir = config.data_dir / "processed" / "responses"
    
    all_data = []
    response_files = sorted(responses_dir.glob("response_*.jsonl"))
    
    for response_file in response_files:
        with open(response_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    all_data.append(json.loads(line))
    
    if not all_data:
        print("❌ No responses found!")
        return
    
    # Shuffle and split
    import random
    random.shuffle(all_data)
    
    split_idx = int(len(all_data) * 0.9)
    train_data = all_data[:split_idx]
    val_data = all_data[split_idx:]
    
    # Save final dataset
    output_dir = config.data_dir / "processed"
    save_jsonl(train_data, output_dir / "train_claude.jsonl")
    save_jsonl(val_data, output_dir / "val_claude.jsonl")
    
    print(f"\n{'='*70}")
    print(f"✅ DATASET COMPLETE!")
    print(f"{'='*70}")
    print(f"\n📊 Statistics:")
    print(f"   Total Q&A pairs: {len(all_data)}")
    print(f"   Training set: {len(train_data)}")
    print(f"   Validation set: {len(val_data)}")
    print(f"\n📁 Saved to:")
    print(f"   {output_dir / 'train_claude.jsonl'}")
    print(f"   {output_dir / 'val_claude.jsonl'}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", action="store_true", help="Split into batches")
    parser.add_argument("--next", action="store_true", help="Show next batch to process")
    parser.add_argument("--combine", action="store_true", help="Combine all responses")
    parser.add_argument("--batch-size", type=int, default=10, help="Chunks per batch")
    
    args = parser.parse_args()
    
    if args.split:
        split_into_batches(args.batch_size)
    elif args.next:
        batch_data, response_file = get_next_batch()
        if batch_data:
            print(f"\n💡 Copy these chunks to Claude and ask for questions!")
            print(f"   Then save the JSON response to: {response_file}")
    elif args.combine:
        combine_all_responses()
    else:
        print("Usage:")
        print("  python manual_dataset_helper.py --split          # Split books into batches")
        print("  python manual_dataset_helper.py --next           # Show next batch")
        print("  python manual_dataset_helper.py --combine        # Combine all responses")
