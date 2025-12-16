"""
Optimized query processor with caching and batch inference
"""

import sys
import hashlib
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from functools import lru_cache
from collections import OrderedDict

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.config import config


class LRUCache:
    """Simple LRU cache implementation"""
    
    def __init__(self, capacity: int = 100):
        self.cache = OrderedDict()
        self.capacity = capacity
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if key not in self.cache:
            return None
        
        # Move to end (most recently used)
        self.cache.move_to_end(key)
        return self.cache[key]
    
    def put(self, key: str, value: Any) -> None:
        """Put value in cache"""
        if key in self.cache:
            # Update existing
            self.cache.move_to_end(key)
        else:
            # Add new
            if len(self.cache) >= self.capacity:
                # Remove least recently used
                self.cache.popitem(last=False)
            self.cache[key] = value
    
    def clear(self):
        """Clear cache"""
        self.cache.clear()
    
    def size(self) -> int:
        """Get cache size"""
        return len(self.cache)


class CachedQueryProcessor:
    """Query processor with response caching"""
    
    def __init__(self, rag_system, model, cache_size: int = 100, 
                 enable_cache: bool = True):
        """
        Initialize cached query processor
        
        Args:
            rag_system: RAG system instance
            model: Model loader instance
            cache_size: Maximum number of cached responses
            enable_cache: Whether to enable caching
        """
        self.rag_system = rag_system
        self.model = model
        self.enable_cache = enable_cache
        
        if enable_cache:
            self.response_cache = LRUCache(capacity=cache_size)
            self.cache_hits = 0
            self.cache_misses = 0
        
    def _get_cache_key(self, question: str, **kwargs) -> str:
        """Generate cache key from question and parameters"""
        # Include relevant parameters in key
        key_data = {
            'question': question.lower().strip(),
            'max_tokens': kwargs.get('max_new_tokens', 512),
            'temperature': kwargs.get('temperature', 0.7),
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def answer(self, question: str, use_cache: bool = True, **kwargs) -> Dict[str, Any]:
        """
        Answer question with caching
        
        Args:
            question: Question to answer
            use_cache: Whether to use cache for this query
            **kwargs: Additional generation parameters
        
        Returns:
            Dictionary with answer and metadata
        """
        # Check cache
        if self.enable_cache and use_cache:
            cache_key = self._get_cache_key(question, **kwargs)
            cached_result = self.response_cache.get(cache_key)
            
            if cached_result is not None:
                self.cache_hits += 1
                print("✓ Using cached response")
                return {**cached_result, 'cached': True}
            else:
                self.cache_misses += 1
        
        # Get context from RAG
        context, sources = self.rag_system.get_context(question)
        
        # Format prompt
        prompt = self._format_prompt(question, context, **kwargs)
        
        # Generate answer
        answer = self.model.generate(prompt, **kwargs)
        
        # Prepare result
        result = {
            'question': question,
            'answer': answer,
            'sources': sources,
            'context_length': len(context),
            'cached': False
        }
        
        # Cache result
        if self.enable_cache and use_cache:
            self.response_cache.put(cache_key, result)
        
        return result
    
    def answer_batch(self, questions: List[str], batch_size: int = 4, 
                    **kwargs) -> List[Dict[str, Any]]:
        """
        Process multiple questions efficiently
        
        Args:
            questions: List of questions
            batch_size: Batch size for inference
            **kwargs: Additional generation parameters
        
        Returns:
            List of results
        """
        results = []
        
        print(f"🔄 Processing {len(questions)} questions in batches of {batch_size}...")
        
        for i in range(0, len(questions), batch_size):
            batch = questions[i:i+batch_size]
            print(f"  Batch {i//batch_size + 1}/{(len(questions)-1)//batch_size + 1}")
            
            # Check cache for each question
            batch_results = []
            uncached_questions = []
            uncached_indices = []
            
            for j, question in enumerate(batch):
                if self.enable_cache:
                    cache_key = self._get_cache_key(question, **kwargs)
                    cached = self.response_cache.get(cache_key)
                    if cached:
                        batch_results.append((j, {**cached, 'cached': True}))
                        self.cache_hits += 1
                        continue
                
                uncached_questions.append(question)
                uncached_indices.append(j)
                self.cache_misses += 1
            
            # Process uncached questions
            if uncached_questions:
                # Get contexts
                batch_contexts = []
                batch_sources = []
                for question in uncached_questions:
                    context, sources = self.rag_system.get_context(question)
                    batch_contexts.append(context)
                    batch_sources.append(sources)
                
                # Format prompts
                batch_prompts = [
                    self._format_prompt(q, ctx, **kwargs) 
                    for q, ctx in zip(uncached_questions, batch_contexts)
                ]
                
                # Batch generation (if model supports it)
                try:
                    batch_answers = self.model.generate_batch(batch_prompts, **kwargs)
                except AttributeError:
                    # Fallback to sequential generation
                    batch_answers = [self.model.generate(p, **kwargs) for p in batch_prompts]
                
                # Create results
                for idx, question, answer, sources, context in zip(
                    uncached_indices, uncached_questions, batch_answers, 
                    batch_sources, batch_contexts
                ):
                    result = {
                        'question': question,
                        'answer': answer,
                        'sources': sources,
                        'context_length': len(context),
                        'cached': False
                    }
                    
                    # Cache result
                    if self.enable_cache:
                        cache_key = self._get_cache_key(question, **kwargs)
                        self.response_cache.put(cache_key, result)
                    
                    batch_results.append((idx, result))
            
            # Sort by original order and add to results
            batch_results.sort(key=lambda x: x[0])
            results.extend([r[1] for r in batch_results])
        
        return results
    
    def _format_prompt(self, question: str, context: str, **kwargs) -> str:
        """Format prompt with context and question"""
        system_prompt = getattr(config, 'system_prompt', 
                               'You are an expert tutor. Provide accurate and detailed answers.')
        
        prompt = f"""{system_prompt}

### Context (Course Materials):
{context}

### Question:
{question}

### Answer:"""
        
        return prompt
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if not self.enable_cache:
            return {'cache_enabled': False}
        
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0.0
        
        return {
            'cache_enabled': True,
            'cache_size': self.response_cache.size(),
            'cache_capacity': self.response_cache.capacity,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate
        }
    
    def clear_cache(self):
        """Clear response cache"""
        if self.enable_cache:
            self.response_cache.clear()
            self.cache_hits = 0
            self.cache_misses = 0
            print("✓ Cache cleared")
