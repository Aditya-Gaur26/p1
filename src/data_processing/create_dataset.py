"""
Create training dataset from extracted course materials
"""

import sys
import json
import random
from pathlib import Path
from typing import List, Dict, Any

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.config import config
from src.utils.helpers import save_jsonl, load_json, extract_topics, generate_id


# Sample Q&A templates for generating synthetic training data
QA_TEMPLATES = [
    {
        "type": "definition",
        "patterns": [
            "What is {topic}?",
            "Define {topic}.",
            "Explain {topic}.",
            "What do you mean by {topic}?"
        ]
    },
    {
        "type": "explanation",
        "patterns": [
            "Explain how {topic} works.",
            "How does {topic} function?",
            "Describe the working of {topic}.",
            "Explain the mechanism of {topic}."
        ]
    },
    {
        "type": "comparison",
        "patterns": [
            "What is the difference between {topic1} and {topic2}?",
            "Compare {topic1} and {topic2}.",
            "How is {topic1} different from {topic2}?"
        ]
    },
    {
        "type": "advantage",
        "patterns": [
            "What are the advantages of {topic}?",
            "Why use {topic}?",
            "What are the benefits of {topic}?"
        ]
    },
    {
        "type": "example",
        "patterns": [
            "Give an example of {topic}.",
            "Provide examples of {topic}.",
            "What are some examples of {topic}?"
        ]
    }
]


def create_training_example(content: str, source: str, topics: List[str] = None) -> Dict[str, Any]:
    """Create a training example from content"""
    if topics is None:
        topics = extract_topics(content)
    
    # For now, create a simple instruction-response pair
    # You can enhance this with more sophisticated generation
    
    return {
        "id": generate_id(content[:100]),
        "instruction": "Explain the following concept from the course material.",
        "input": "",
        "output": content,
        "source": source,
        "topics": topics,
        "text": f"### Instruction:\nExplain the following concept from the course material.\n\n### Response:\n{content}"
    }


def generate_qa_pairs(content: str, source: str) -> List[Dict[str, Any]]:
    """Generate Q&A pairs from content (enhanced version)"""
    qa_pairs = []
    topics = extract_topics(content)
    
    # Create general explanation
    qa_pairs.append({
        "id": generate_id(content[:100]),
        "instruction": f"Explain the concept covered in {source}.",
        "input": "",
        "output": content,
        "source": source,
        "topics": topics,
        "text": f"### Instruction:\nExplain the concept covered in {source}.\n\n### Response:\n{content}"
    })
    
    # Create topic-specific questions
    for topic in topics[:3]:  # Limit to top 3 topics
        qa_pairs.append({
            "id": generate_id(f"{topic}_{content[:50]}"),
            "instruction": f"What is {topic}? Explain with reference to the course material.",
            "input": "",
            "output": content,
            "source": source,
            "topics": [topic],
            "text": f"### Instruction:\nWhat is {topic}? Explain with reference to the course material.\n\n### Response:\n{content}"
        })
    
    return qa_pairs


def process_slides(slides_dir: Path) -> List[Dict[str, Any]]:
    """Process extracted slides into training examples"""
    training_data = []
    
    combined_file = slides_dir / "all_slides_combined.json"
    if not combined_file.exists():
        print("⚠️  No slides data found. Run extract_slides.py first.")
        return training_data
    
    data = load_json(combined_file)
    print(f"📊 Processing {data['total_files']} slide files...")
    
    for file_data in data['files']:
        filename = file_data['filename']
        for slide in file_data['slides']:
            content = slide['content']
            if len(content) > 50:  # Skip very short content
                source = f"{filename} - Slide {slide['slide_number']}"
                qa_pairs = generate_qa_pairs(content, source)
                training_data.extend(qa_pairs)
    
    print(f"  ✓ Generated {len(training_data)} examples from slides")
    return training_data


def process_books(books_dir: Path) -> List[Dict[str, Any]]:
    """Process extracted books into training examples"""
    training_data = []
    
    combined_file = books_dir / "all_pdfs_combined.json"
    if not combined_file.exists():
        print("⚠️  No PDF data found. Run extract_pdfs.py first.")
        return training_data
    
    data = load_json(combined_file)
    print(f"📚 Processing {data['total_files']} PDF files...")
    
    for file_data in data['files']:
        filename = file_data['filename']
        for i, chunk in enumerate(file_data['chunks'], 1):
            if len(chunk) > 100:  # Skip very short chunks
                source = f"{filename} - Section {i}"
                qa_pairs = generate_qa_pairs(chunk, source)
                training_data.extend(qa_pairs)
    
    print(f"  ✓ Generated {len(training_data)} examples from books")
    return training_data


def create_dataset():
    """Main function to create training dataset"""
    print("=" * 60)
    print("Creating Training Dataset".center(60))
    print("=" * 60)
    
    # Collect all training data
    all_training_data = []
    
    # Process slides
    slides_dir = config.data_dir / "processed" / "slides"
    all_training_data.extend(process_slides(slides_dir))
    
    # Process books
    books_dir = config.data_dir / "processed" / "books"
    all_training_data.extend(process_books(books_dir))
    
    if not all_training_data:
        print("\n❌ No training data generated!")
        print("Please ensure you have run:")
        print("  1. extract_slides.py")
        print("  2. extract_pdfs.py")
        return
    
    # Shuffle the data
    random.shuffle(all_training_data)
    
    # Split into train and validation
    split_idx = int(len(all_training_data) * 0.9)
    train_data = all_training_data[:split_idx]
    val_data = all_training_data[split_idx:]
    
    # Save datasets
    output_dir = config.data_dir / "processed"
    save_jsonl(train_data, output_dir / "train.jsonl")
    save_jsonl(val_data, output_dir / "val.jsonl")
    
    print(f"\n✅ Dataset created successfully!")
    print(f"📊 Total examples: {len(all_training_data)}")
    print(f"   Training: {len(train_data)}")
    print(f"   Validation: {len(val_data)}")
    print(f"📁 Saved to: {output_dir}")


def main():
    create_dataset()


if __name__ == "__main__":
    main()
