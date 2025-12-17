"""
Enhanced Dataset Creation with LLM Augmentation
Drop-in replacement for create_dataset.py that uses LLM for better quality

Usage:
  python src/data_processing/create_dataset_llm.py --provider ollama --model llama3.2:3b
  python src/data_processing/create_dataset_llm.py --provider openai --model gpt-4
"""

import sys
import random
import argparse
from pathlib import Path
from typing import List, Dict, Any

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.config import config
from src.utils.helpers import save_jsonl, load_json
from src.data_processing.llm_augmentation import LLMQuestionGenerator, generate_llm_enhanced_dataset


def load_extracted_content() -> List[Dict[str, str]]:
    """Load all extracted PDF and slide content"""
    
    all_content = []
    
    # Load PDFs
    books_file = config.data_dir / "processed" / "books" / "all_pdfs_combined.json"
    if books_file.exists():
        print(f"📚 Loading PDFs from {books_file}")
        data = load_json(books_file)
        
        for file_data in data['files']:
            filename = file_data['filename']
            for i, chunk in enumerate(file_data['chunks'], 1):
                if len(chunk) > 100:  # Skip short chunks
                    all_content.append({
                        'content': chunk,
                        'source': f"{filename} - Section {i}"
                    })
        
        print(f"   ✓ Loaded {len(all_content)} chunks from PDFs")
    
    # Load Slides
    slides_file = config.data_dir / "processed" / "slides" / "all_slides_combined.json"
    if slides_file.exists():
        print(f"📊 Loading slides from {slides_file}")
        data = load_json(slides_file)
        
        slide_count = 0
        for file_data in data['files']:
            filename = file_data['filename']
            for slide in file_data['slides']:
                content = slide['content']
                if len(content) > 50:
                    all_content.append({
                        'content': content,
                        'source': f"{filename} - Slide {slide['slide_number']}"
                    })
                    slide_count += 1
        
        print(f"   ✓ Loaded {slide_count} slides")
    
    if not all_content:
        print("\n❌ No content found!")
        print("Run these first:")
        print("  python src/data_processing/extract_pdfs.py")
        print("  python src/data_processing/extract_slides.py")
        sys.exit(1)
    
    print(f"\n📊 Total content chunks: {len(all_content)}")
    return all_content


def create_llm_enhanced_dataset(
    provider: str = "ollama",
    model: str = "llama3.2:3b",
    questions_per_chunk: int = 3,
    max_chunks: int = None
):
    """
    Create training dataset using LLM augmentation
    
    Args:
        provider: "ollama" (local) or "openai" (API)
        model: Model name
        questions_per_chunk: How many questions to generate per content chunk
        max_chunks: Limit chunks for testing (None = all)
    """
    
    print("=" * 70)
    print("LLM-Enhanced Dataset Creation".center(70))
    print("=" * 70)
    print(f"\n🤖 Using: {provider} / {model}")
    
    # Load content
    all_content = load_extracted_content()
    
    # Limit for testing
    if max_chunks and len(all_content) > max_chunks:
        print(f"\n⚠️  Limiting to first {max_chunks} chunks for testing")
        all_content = all_content[:max_chunks]
    
    # Initialize LLM generator
    print(f"\n🔧 Initializing LLM generator...")
    try:
        generator = LLMQuestionGenerator(provider=provider, model=model)
    except Exception as e:
        print(f"\n❌ Failed to initialize LLM: {e}")
        print("\nTroubleshooting:")
        if provider == "ollama":
            print("  1. Install Ollama: https://ollama.ai")
            print("  2. Start Ollama: ollama serve")
            print(f"  3. Pull model: ollama pull {model}")
        elif provider == "openai":
            print("  1. Set API key: export OPENAI_API_KEY=sk-...")
            print("  2. Or add to configs/api_config.yaml")
        sys.exit(1)
    
    # Generate dataset
    training_data = generate_llm_enhanced_dataset(
        content_chunks=all_content,
        llm_generator=generator,
        questions_per_chunk=questions_per_chunk
    )
    
    if not training_data:
        print("\n❌ No training data generated!")
        sys.exit(1)
    
    # Shuffle
    random.shuffle(training_data)
    
    # Split train/val
    split_idx = int(len(training_data) * 0.9)
    train_data = training_data[:split_idx]
    val_data = training_data[split_idx:]
    
    # Save
    output_dir = config.data_dir / "processed"
    save_jsonl(train_data, output_dir / "train.jsonl")
    save_jsonl(val_data, output_dir / "val.jsonl")
    
    # Statistics
    print(f"\n" + "=" * 70)
    print("Dataset Statistics".center(70))
    print("=" * 70)
    print(f"\n📊 Total examples: {len(training_data)}")
    print(f"   Training: {len(train_data)} ({len(train_data)/len(training_data)*100:.1f}%)")
    print(f"   Validation: {len(val_data)} ({len(val_data)/len(training_data)*100:.1f}%)")
    
    # Question type distribution
    question_types = {}
    for item in train_data:
        qtype = item.get("question_type", "general")
        question_types[qtype] = question_types.get(qtype, 0) + 1
    
    print(f"\n📈 Question Types:")
    for qtype, count in sorted(question_types.items(), key=lambda x: x[1], reverse=True):
        pct = (count / len(train_data)) * 100
        print(f"   {qtype:<20} {count:>5} ({pct:>5.1f}%)")
    
    # Difficulty distribution
    difficulties = {}
    for item in train_data:
        diff = item.get("difficulty", "medium")
        difficulties[diff] = difficulties.get(diff, 0) + 1
    
    print(f"\n📊 Difficulty Levels:")
    for diff, count in sorted(difficulties.items()):
        pct = (count / len(train_data)) * 100
        print(f"   {diff:<10} {count:>5} ({pct:>5.1f}%)")
    
    print(f"\n💾 Saved to: {output_dir}")
    print("=" * 70)
    
    # Quality check
    print(f"\n🔍 Quick Quality Check:")
    print(f"   Average question length: {sum(len(d['instruction']) for d in train_data)/len(train_data):.0f} chars")
    print(f"   Average answer length: {sum(len(d['output']) for d in train_data)/len(train_data):.0f} chars")
    
    short_answers = [d for d in train_data if len(d['output']) < 50]
    if short_answers:
        print(f"   ⚠️  {len(short_answers)} answers < 50 chars ({len(short_answers)/len(train_data)*100:.1f}%)")
    else:
        print(f"   ✅ All answers >= 50 chars")
    
    refusal_count = sum(1 for d in train_data if d.get('question_type') == 'refusal')
    print(f"   ✅ Refusal examples: {refusal_count} ({refusal_count/len(train_data)*100:.1f}%)")
    
    print(f"\n✅ LLM-enhanced dataset created successfully!")
    print(f"\nNext steps:")
    print(f"  1. Check quality: python diagnose.py")
    print(f"  2. Train model: python src/training/fine_tune.py")


def main():
    parser = argparse.ArgumentParser(description="Create LLM-enhanced training dataset")
    
    parser.add_argument(
        "--provider",
        type=str,
        default="ollama",
        choices=["ollama", "openai", "huggingface"],
        help="LLM provider (default: ollama)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="llama3.2:3b",
        help="Model name (default: llama3.2:3b for Ollama, gpt-4 for OpenAI)"
    )
    
    parser.add_argument(
        "--questions-per-chunk",
        type=int,
        default=3,
        help="Questions to generate per content chunk (default: 3)"
    )
    
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Limit chunks for testing (default: None = all)"
    )
    
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: process only 10 chunks"
    )
    
    args = parser.parse_args()
    
    # Adjust model defaults
    if args.provider == "openai" and args.model == "llama3.2:3b":
        args.model = "gpt-4"
        print(f"ℹ️  Using OpenAI default model: {args.model}")
    
    # Test mode
    if args.test:
        args.max_chunks = 10
        print("ℹ️  Test mode: processing 10 chunks only")
    
    try:
        create_llm_enhanced_dataset(
            provider=args.provider,
            model=args.model,
            questions_per_chunk=args.questions_per_chunk,
            max_chunks=args.max_chunks
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
