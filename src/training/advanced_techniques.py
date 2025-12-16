"""
Advanced training techniques: curriculum learning, contrastive learning
"""

import random
from typing import List, Dict, Any, Tuple
from datasets import Dataset
import numpy as np


class CurriculumLearner:
    """Curriculum learning - train from easy to hard examples"""
    
    def __init__(self, difficulty_metric: str = "length"):
        """
        Initialize curriculum learner
        
        Args:
            difficulty_metric: How to measure difficulty ("length", "complexity", "custom")
        """
        self.difficulty_metric = difficulty_metric
    
    def estimate_difficulty(self, example: Dict[str, Any]) -> float:
        """
        Estimate difficulty of an example
        
        Returns:
            Difficulty score (0-1, where 1 is hardest)
        """
        if self.difficulty_metric == "length":
            # Based on text length
            instruction = example.get('instruction', '')
            output = example.get('output', '')
            total_length = len(instruction.split()) + len(output.split())
            
            # Normalize to 0-1 (assuming max ~500 words)
            return min(total_length / 500.0, 1.0)
        
        elif self.difficulty_metric == "complexity":
            # Based on question complexity
            instruction = example.get('instruction', '').lower()
            
            complexity = 0.0
            
            # Complex question words
            if any(word in instruction for word in ['why', 'how', 'analyze', 'evaluate']):
                complexity += 0.3
            
            # Multiple questions
            if instruction.count('?') > 1:
                complexity += 0.2
            
            # Technical terms (simple heuristic)
            tech_terms = ['protocol', 'algorithm', 'synchronization', 'deadlock', 
                         'virtualization', 'routing', 'scheduling']
            complexity += 0.1 * sum(1 for term in tech_terms if term in instruction)
            
            # Output length
            output_len = len(example.get('output', '').split())
            if output_len > 200:
                complexity += 0.3
            
            return min(complexity, 1.0)
        
        else:  # custom
            # Use predefined difficulty if available
            return example.get('difficulty', 0.5)
    
    def sort_by_difficulty(self, dataset: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], float]]:
        """
        Sort dataset by difficulty
        
        Returns:
            List of (example, difficulty_score) tuples
        """
        scored_examples = []
        
        for example in dataset:
            difficulty = self.estimate_difficulty(example)
            scored_examples.append((example, difficulty))
        
        # Sort by difficulty
        scored_examples.sort(key=lambda x: x[1])
        
        return scored_examples
    
    def create_curriculum_batches(self, dataset: List[Dict[str, Any]], 
                                 num_epochs: int = 5) -> List[List[Dict[str, Any]]]:
        """
        Create curriculum batches for training
        
        Args:
            dataset: Training dataset
            num_epochs: Number of training epochs
        
        Returns:
            List of datasets for each epoch
        """
        # Sort by difficulty
        sorted_data = self.sort_by_difficulty(dataset)
        
        epoch_datasets = []
        
        for epoch in range(num_epochs):
            # Gradually increase difficulty threshold
            difficulty_threshold = (epoch + 1) / num_epochs
            
            # Include examples up to threshold
            epoch_data = [
                example for example, difficulty in sorted_data
                if difficulty <= difficulty_threshold or epoch == num_epochs - 1
            ]
            
            # Add some randomness to avoid overfitting to easy examples
            random.shuffle(epoch_data)
            
            epoch_datasets.append(epoch_data)
            
            print(f"  Epoch {epoch + 1}: {len(epoch_data)} examples (threshold: {difficulty_threshold:.2f})")
        
        return epoch_datasets


class ContrastiveLearner:
    """Contrastive learning for better representations"""
    
    def __init__(self, negative_samples: int = 2):
        """
        Initialize contrastive learner
        
        Args:
            negative_samples: Number of negative samples per positive
        """
        self.negative_samples = negative_samples
    
    def create_contrastive_pairs(self, dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Create contrastive learning pairs
        
        Each example gets:
        - 1 positive pair (question + correct answer)
        - N negative pairs (question + wrong answers)
        """
        contrastive_dataset = []
        
        # Group examples by topic for better negatives
        topic_groups = {}
        for example in dataset:
            topics = example.get('topics', ['general'])
            for topic in topics:
                if topic not in topic_groups:
                    topic_groups[topic] = []
                topic_groups[topic].append(example)
        
        for example in dataset:
            instruction = example['instruction']
            correct_output = example['output']
            topics = example.get('topics', ['general'])
            
            # Positive pair
            contrastive_dataset.append({
                **example,
                'label': 1,
                'pair_type': 'positive'
            })
            
            # Negative pairs - use answers from similar topics
            negative_count = 0
            for topic in topics:
                if topic in topic_groups and negative_count < self.negative_samples:
                    # Sample negative from same topic
                    candidates = [e for e in topic_groups[topic] 
                                if e['output'] != correct_output]
                    
                    if candidates:
                        negative = random.choice(candidates)
                        contrastive_dataset.append({
                            'id': f"{example['id']}_neg_{negative_count}",
                            'instruction': instruction,
                            'input': example.get('input', ''),
                            'output': negative['output'],
                            'source': example.get('source', ''),
                            'topics': topics,
                            'label': 0,
                            'pair_type': 'negative',
                            'text': f"### Instruction:\n{instruction}\n\n### Response:\n{negative['output']}"
                        })
                        negative_count += 1
            
            # Fill remaining negatives with random samples
            while negative_count < self.negative_samples:
                negative = random.choice(dataset)
                if negative['output'] != correct_output:
                    contrastive_dataset.append({
                        'id': f"{example['id']}_neg_{negative_count}",
                        'instruction': instruction,
                        'input': example.get('input', ''),
                        'output': negative['output'],
                        'source': example.get('source', ''),
                        'topics': topics,
                        'label': 0,
                        'pair_type': 'negative',
                        'text': f"### Instruction:\n{instruction}\n\n### Response:\n{negative['output']}"
                    })
                    negative_count += 1
        
        return contrastive_dataset


def apply_mixup(batch_embeddings, alpha: float = 0.2):
    """
    Apply mixup data augmentation to embeddings
    
    Args:
        batch_embeddings: Batch of embeddings
        alpha: Mixup parameter
    
    Returns:
        Mixed embeddings
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = len(batch_embeddings)
    index = np.random.permutation(batch_size)
    
    mixed_embeddings = lam * batch_embeddings + (1 - lam) * batch_embeddings[index]
    
    return mixed_embeddings


def create_multi_task_dataset(base_dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Create multi-task learning dataset
    
    Tasks:
    - Question answering
    - Summarization
    - Error detection
    - Concept explanation
    """
    multi_task_dataset = []
    
    for example in base_dataset:
        content = example.get('output', '')
        source = example.get('source', '')
        
        # Task 1: Original QA
        multi_task_dataset.append({
            **example,
            'task': 'qa'
        })
        
        # Task 2: Summarization (if content is long enough)
        if len(content.split()) > 100:
            multi_task_dataset.append({
                'id': f"{example['id']}_summary",
                'instruction': f"Summarize the following content from {source}:",
                'input': content,
                'output': content[:len(content)//2] + "...",  # Simple truncation for demo
                'task': 'summarization'
            })
        
        # Task 3: Concept extraction
        topics = example.get('topics', [])
        if topics:
            multi_task_dataset.append({
                'id': f"{example['id']}_concepts",
                'instruction': "What are the key concepts in this content?",
                'input': content,
                'output': ", ".join(topics),
                'task': 'concept_extraction'
            })
    
    return multi_task_dataset
