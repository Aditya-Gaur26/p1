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
from src.data_processing.data_augmentation import (
    DataAugmentor, SemanticChunker, augment_dataset, create_contrastive_pairs
)


# Enhanced Q&A templates for diverse question types
QA_TEMPLATES = [
    {
        "type": "conceptual",
        "patterns": [
            "What is {topic}?",
            "Define {topic}.",
            "Explain {topic}.",
            "What do you mean by {topic}?",
            "Describe the concept of {topic}."
        ]
    },
    {
        "type": "procedural",
        "patterns": [
            "Explain how {topic} works.",
            "How does {topic} function?",
            "Describe the working of {topic}.",
            "Explain the mechanism of {topic}.",
            "What is the process of {topic}?",
            "Walk me through {topic} step by step."
        ]
    },
    {
        "type": "comparative",
        "patterns": [
            "What is the difference between {topic1} and {topic2}?",
            "Compare {topic1} and {topic2}.",
            "How is {topic1} different from {topic2}?",
            "Compare and contrast {topic1} with {topic2}.",
            "What are the similarities and differences between {topic1} and {topic2}?"
        ]
    },
    {
        "type": "analytical",
        "patterns": [
            "What are the advantages of {topic}?",
            "What are the disadvantages of {topic}?",
            "Why use {topic}?",
            "What are the benefits of {topic}?",
            "Analyze the trade-offs of {topic}.",
            "What are the limitations of {topic}?"
        ]
    },
    {
        "type": "application",
        "patterns": [
            "When should you use {topic}?",
            "What are the applications of {topic}?",
            "In what scenarios is {topic} used?",
            "Give real-world examples of {topic}.",
            "How is {topic} applied in practice?"
        ]
    },
    {
        "type": "troubleshooting",
        "patterns": [
            "What problems can occur with {topic}?",
            "Why would {topic} fail?",
            "How do you debug issues with {topic}?",
            "What are common errors in {topic}?"
        ]
    },
    {
        "type": "deep_understanding",
        "patterns": [
            "Explain {topic} in detail.",
            "Provide a comprehensive explanation of {topic}.",
            "What are all the components of {topic}?",
            "Explain {topic} with examples and diagrams."
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
    """Generate diverse Q&A pairs from content (enhanced version)"""
    qa_pairs = []
    topics = extract_topics(content)
    augmentor = DataAugmentor()
    
    # Create general explanation with reasoning chain
    base_answer = augmentor.add_reasoning_chain("", content)
    qa_pairs.append({
        "id": generate_id(content[:100]),
        "instruction": f"Explain the concept covered in {source}.",
        "input": "",
        "output": base_answer,
        "source": source,
        "topics": topics,
        "text": f"### Instruction:\nExplain the concept covered in {source}.\n\n### Response:\n{base_answer}"
    })
    
    # Create diverse topic-specific questions using templates
    for i, topic in enumerate(topics[:3]):  # Limit to top 3 topics
        # Select random template type
        template_type = random.choice(QA_TEMPLATES)
        pattern = random.choice(template_type["patterns"])
        
        # Handle comparison questions differently
        if "{topic1}" in pattern and len(topics) > 1:
            other_topic = topics[(i + 1) % len(topics)]
            question = pattern.format(topic1=topic, topic2=other_topic)
        else:
            question = pattern.format(topic=topic)
        
        qa_pairs.append({
            "id": generate_id(f"{topic}_{content[:50]}_{i}"),
            "instruction": question,
            "input": "",
            "output": content,
            "source": source,
            "topics": [topic],
            "question_type": template_type["type"],
            "text": f"### Instruction:\n{question}\n\n### Response:\n{content}"
        })
    
    # Generate paraphrased versions for first question
    if qa_pairs:
        first_q = qa_pairs[0]["instruction"]
        paraphrases = augmentor.paraphrase_question(first_q)
        for j, paraphrase in enumerate(paraphrases[1:3], 1):  # Add 2 paraphrases
            new_qa = qa_pairs[0].copy()
            new_qa["id"] = f"{qa_pairs[0]['id']}_para_{j}"
            new_qa["instruction"] = paraphrase
            new_qa["text"] = f"### Instruction:\n{paraphrase}\n\n### Response:\n{new_qa['output']}"
            qa_pairs.append(new_qa)
    
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
    print("Creating Enhanced Training Dataset".center(60))
    print("=" * 60)
    
    # Initialize semantic chunker
    chunker = SemanticChunker(chunk_size=512, overlap=50)
    
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
    
    print(f"\n📊 Generated {len(all_training_data)} base examples")
    
    # Apply data augmentation
    print("\n🔄 Applying data augmentation...")
    all_training_data = augment_dataset(all_training_data, augmentation_factor=2)
    
    # Shuffle the data
    random.shuffle(all_training_data)
    
    # Split into train and validation (90/10 split)
    split_idx = int(len(all_training_data) * 0.9)
    train_data = all_training_data[:split_idx]
    val_data = all_training_data[split_idx:]
    
    # Save datasets
    output_dir = config.data_dir / "processed"
    save_jsonl(train_data, output_dir / "train.jsonl")
    save_jsonl(val_data, output_dir / "val.jsonl")
    
    # Generate statistics
    question_types = {}
    for item in train_data:
        qtype = item.get("question_type", "general")
        question_types[qtype] = question_types.get(qtype, 0) + 1
    
    print(f"\n✅ Enhanced dataset created successfully!")
    print(f"📊 Total examples: {len(all_training_data)}")
    print(f"   Training: {len(train_data)}")
    print(f"   Validation: {len(val_data)}")
    print(f"\n📈 Question type distribution:")
    for qtype, count in sorted(question_types.items(), key=lambda x: x[1], reverse=True):
        print(f"   {qtype}: {count}")
    print(f"\n📁 Saved to: {output_dir}")



def main():
    create_dataset()


if __name__ == "__main__":
    main()
