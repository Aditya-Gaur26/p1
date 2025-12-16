"""
Helper utility functions
"""

import json
import time
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
from functools import wraps


def timer(func):
    """Decorator to measure function execution time"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"⏱️  {func.__name__} took {end_time - start_time:.2f} seconds")
        return result
    return wrapper


def load_jsonl(filepath: Path) -> List[Dict[str, Any]]:
    """Load JSONL file"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict[str, Any]], filepath: Path):
    """Save data to JSONL file"""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"✓ Saved {len(data)} items to {filepath}")


def load_json(filepath: Path) -> Dict[str, Any]:
    """Load JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Dict[str, Any], filepath: Path):
    """Save data to JSON file"""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved to {filepath}")


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    
    return chunks


def clean_text(text: str) -> str:
    """Clean and normalize text"""
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove special characters but keep basic punctuation
    # text = re.sub(r'[^\w\s.,!?;:()\-]', '', text)
    
    return text.strip()


def generate_id(text: str) -> str:
    """Generate unique ID from text"""
    return hashlib.md5(text.encode()).hexdigest()[:16]


def format_prompt(instruction: str, context: Optional[str] = None, 
                  template: str = "alpaca") -> str:
    """Format prompt for training/inference"""
    if template == "alpaca":
        if context:
            return f"""Below is an instruction that describes a task, paired with context that provides further information. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Context:
{context}

### Response:
"""
        else:
            return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""
    elif template == "chatml":
        messages = [
            {"role": "system", "content": "You are an expert tutor in Operating Systems and Networks."},
            {"role": "user", "content": instruction}
        ]
        if context:
            messages.insert(1, {"role": "assistant", "content": f"Context: {context}"})
        return messages
    
    return instruction


def extract_topics(text: str) -> List[str]:
    """Extract topic keywords from text (simple version)"""
    # Common OS/Networks topics
    topics_dict = {
        'process', 'thread', 'scheduling', 'synchronization', 'deadlock',
        'memory', 'virtual memory', 'paging', 'segmentation', 'cache',
        'file system', 'disk', 'io', 'interrupt', 'system call',
        'tcp', 'udp', 'ip', 'routing', 'protocol', 'osi', 'network',
        'socket', 'http', 'dns', 'firewall', 'security', 'encryption'
    }
    
    text_lower = text.lower()
    found_topics = [topic for topic in topics_dict if topic in text_lower]
    
    return list(set(found_topics))


def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate simple word-based similarity"""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union) if union else 0.0


def truncate_text(text: str, max_length: int = 200, suffix: str = "...") -> str:
    """Truncate text to max length"""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)].rsplit(' ', 1)[0] + suffix


def print_section(title: str, char: str = "="):
    """Print formatted section header"""
    print(f"\n{char * 60}")
    print(f"{title.center(60)}")
    print(f"{char * 60}\n")


if __name__ == "__main__":
    # Test utilities
    test_text = "This is a sample text for testing the chunking function."
    chunks = chunk_text(test_text, chunk_size=5, overlap=2)
    print("Chunks:", chunks)
    
    topics = extract_topics("Process scheduling and deadlock detection in operating systems")
    print("Topics:", topics)
