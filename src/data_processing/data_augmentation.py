"""
Advanced data augmentation techniques for training dataset
"""

import random
import re
from typing import List, Dict, Any, Tuple
from pathlib import Path


class DataAugmentor:
    """Data augmentation utilities for QA pairs"""
    
    def __init__(self):
        self.paraphrase_patterns = {
            "What is": ["Define", "Explain", "Describe", "What do you mean by"],
            "How does": ["Explain how", "Describe the process of", "What is the mechanism of"],
            "Why": ["What is the reason for", "Explain why", "What causes"],
            "When": ["In what situation", "Under what circumstances", "At what point"],
        }
        
    def paraphrase_question(self, question: str) -> List[str]:
        """Generate paraphrased versions of a question"""
        paraphrases = [question]
        
        for pattern, alternatives in self.paraphrase_patterns.items():
            if question.startswith(pattern):
                for alt in alternatives:
                    new_q = question.replace(pattern, alt, 1)
                    paraphrases.append(new_q)
                break
        
        return paraphrases
    
    def create_negative_samples(self, correct_answer: str, all_answers: List[str]) -> List[str]:
        """Create negative samples for contrastive learning"""
        negatives = []
        
        # Sample random wrong answers
        for answer in random.sample(all_answers, min(3, len(all_answers))):
            if answer != correct_answer:
                negatives.append(answer)
        
        return negatives
    
    def add_reasoning_chain(self, question: str, answer: str) -> str:
        """Add chain-of-thought reasoning to answer"""
        # Simple heuristic: if answer is long, add thinking steps
        if len(answer) > 200:
            reasoning_prefix = "Let me explain this step by step:\n\n"
            
            # Try to split into logical chunks
            sentences = answer.split('. ')
            if len(sentences) >= 3:
                formatted_answer = reasoning_prefix
                for i, sent in enumerate(sentences, 1):
                    if sent.strip():
                        formatted_answer += f"{i}. {sent.strip()}.\n"
                return formatted_answer
        
        return answer
    
    def generate_multi_turn(self, content: str, source: str) -> List[Dict[str, Any]]:
        """Generate multi-turn conversation examples"""
        conversations = []
        
        # Split content into logical segments
        segments = self._split_into_segments(content)
        
        if len(segments) >= 2:
            # Create follow-up questions
            conversation = {
                "turns": [
                    {"role": "user", "content": f"Explain {source.split('-')[0].strip()}"},
                    {"role": "assistant", "content": segments[0]},
                    {"role": "user", "content": "Can you elaborate more on that?"},
                    {"role": "assistant", "content": segments[1]},
                ]
            }
            conversations.append(conversation)
        
        return conversations
    
    def _split_into_segments(self, text: str) -> List[str]:
        """Split text into logical segments"""
        # Split by double newlines or numbered points
        segments = re.split(r'\n\n+|\d+\.\s+', text)
        return [s.strip() for s in segments if len(s.strip()) > 50]


class SemanticChunker:
    """Semantic-aware text chunking"""
    
    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_by_sentences(self, text: str) -> List[str]:
        """Chunk text by sentences while respecting semantic boundaries"""
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence.split())
            
            # If adding this sentence exceeds chunk size
            if current_length + sentence_length > self.chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                # Keep last few sentences for overlap
                overlap_sentences = self._get_overlap_sentences(current_chunk)
                current_chunk = overlap_sentences
                current_length = sum(len(s.split()) for s in current_chunk)
            
            current_chunk.append(sentence)
            current_length += sentence_length
        
        # Add remaining chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def chunk_by_topic(self, text: str) -> List[Dict[str, Any]]:
        """Chunk text by detecting topic boundaries"""
        # Simple heuristic: split by headers or numbered sections
        chunks = []
        
        # Detect section headers (e.g., "1. Introduction", "### Topic")
        sections = re.split(r'\n(?=#{1,3}\s+|\d+\.\s+[A-Z])', text)
        
        for section in sections:
            if len(section.strip()) > 100:
                # Extract title if present
                lines = section.split('\n', 1)
                title = lines[0].strip() if len(lines) > 1 else "Section"
                content = lines[1] if len(lines) > 1 else section
                
                chunks.append({
                    "title": title,
                    "content": content.strip(),
                    "length": len(content.split())
                })
        
        return chunks
    
    def _get_overlap_sentences(self, sentences: List[str]) -> List[str]:
        """Get last few sentences for overlap"""
        overlap_words = 0
        overlap_sentences = []
        
        for sentence in reversed(sentences):
            sentence_words = len(sentence.split())
            if overlap_words + sentence_words > self.overlap:
                break
            overlap_sentences.insert(0, sentence)
            overlap_words += sentence_words
        
        return overlap_sentences


def create_contrastive_pairs(question: str, correct_answer: str, 
                            wrong_answers: List[str]) -> List[Dict[str, Any]]:
    """Create contrastive learning pairs"""
    pairs = []
    
    # Positive pair
    pairs.append({
        "question": question,
        "answer": correct_answer,
        "label": 1
    })
    
    # Negative pairs
    for wrong_answer in wrong_answers[:2]:  # Limit to 2 negatives
        pairs.append({
            "question": question,
            "answer": wrong_answer,
            "label": 0
        })
    
    return pairs


def augment_dataset(data: List[Dict[str, Any]], augmentation_factor: int = 2) -> List[Dict[str, Any]]:
    """Augment entire dataset"""
    augmentor = DataAugmentor()
    augmented_data = list(data)  # Keep original
    
    print(f"🔄 Augmenting dataset (factor: {augmentation_factor})...")
    
    for item in data[:len(data)//2]:  # Augment half of the dataset
        # Paraphrase questions
        if "instruction" in item:
            paraphrases = augmentor.paraphrase_question(item["instruction"])
            for paraphrase in paraphrases[1:augmentation_factor]:
                new_item = item.copy()
                new_item["instruction"] = paraphrase
                new_item["id"] = f"{item['id']}_aug_{len(augmented_data)}"
                augmented_data.append(new_item)
    
    print(f"  ✓ Augmented from {len(data)} to {len(augmented_data)} examples")
    return augmented_data
