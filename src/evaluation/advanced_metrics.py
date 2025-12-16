"""
Advanced evaluation metrics including semantic similarity and BERTScore
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
from collections import defaultdict

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.config import config


class AdvancedEvaluator:
    """Enhanced evaluation with semantic and factual metrics"""
    
    def __init__(self, use_semantic: bool = True, use_bertscore: bool = True):
        """
        Initialize evaluator with advanced metrics
        
        Args:
            use_semantic: Use semantic similarity
            use_bertscore: Use BERTScore
        """
        self.use_semantic = use_semantic
        self.use_bertscore = use_bertscore
        
        # Load models lazily
        self.semantic_model = None
        self.bertscore_loaded = False
        
    def _load_semantic_model(self):
        """Lazy load semantic similarity model"""
        if self.semantic_model is None:
            from sentence_transformers import SentenceTransformer
            print("📦 Loading semantic similarity model...")
            self.semantic_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    
    def _compute_bertscore(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Compute BERTScore"""
        try:
            from bert_score import score as bert_score
            P, R, F1 = bert_score(predictions, references, lang='en', verbose=False)
            return {
                'bertscore_precision': P.mean().item(),
                'bertscore_recall': R.mean().item(),
                'bertscore_f1': F1.mean().item()
            }
        except ImportError:
            print("⚠️  bert-score not installed. Run: pip install bert-score")
            return {'bertscore_f1': 0.0}
    
    def calculate_rouge(self, prediction: str, reference: str) -> Dict[str, float]:
        """Calculate ROUGE scores"""
        try:
            from rouge_score import rouge_scorer
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            scores = scorer.score(reference, prediction)
            
            return {
                'rouge1': scores['rouge1'].fmeasure,
                'rouge2': scores['rouge2'].fmeasure,
                'rougeL': scores['rougeL'].fmeasure
            }
        except ImportError:
            print("⚠️  rouge-score not installed. Run: pip install rouge-score")
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
    
    def calculate_bleu(self, prediction: str, reference: str) -> float:
        """Calculate BLEU score"""
        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            reference_tokens = [reference.split()]
            prediction_tokens = prediction.split()
            smoothie = SmoothingFunction().method4
            return sentence_bleu(reference_tokens, prediction_tokens, smoothing_function=smoothie)
        except ImportError:
            print("⚠️  nltk not installed. Run: pip install nltk")
            return 0.0
    
    def calculate_semantic_similarity(self, prediction: str, reference: str) -> float:
        """Calculate semantic similarity using sentence embeddings"""
        if not self.use_semantic:
            return 0.0
        
        self._load_semantic_model()
        
        # Encode both texts
        embeddings = self.semantic_model.encode([prediction, reference])
        
        # Compute cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        
        return float(similarity)
    
    def check_factual_consistency(self, prediction: str, reference: str) -> Dict[str, Any]:
        """
        Check factual consistency between prediction and reference
        Simple heuristic-based approach
        """
        # Extract key facts (simple version - can be enhanced with NLI models)
        pred_sentences = prediction.split('.')
        ref_sentences = reference.split('.')
        
        # Check overlap
        pred_words = set(prediction.lower().split())
        ref_words = set(reference.lower().split())
        
        overlap = len(pred_words & ref_words)
        total = len(ref_words)
        
        consistency_score = overlap / total if total > 0 else 0.0
        
        return {
            'factual_consistency': consistency_score,
            'word_overlap': overlap,
            'reference_words': total
        }
    
    def calculate_all_metrics(self, prediction: str, reference: str) -> Dict[str, float]:
        """Calculate all available metrics"""
        metrics = {}
        
        # Traditional metrics
        metrics.update(self.calculate_rouge(prediction, reference))
        metrics['bleu'] = self.calculate_bleu(prediction, reference)
        
        # Semantic similarity
        if self.use_semantic:
            metrics['semantic_similarity'] = self.calculate_semantic_similarity(prediction, reference)
        
        # Factual consistency
        consistency = self.check_factual_consistency(prediction, reference)
        metrics['factual_consistency'] = consistency['factual_consistency']
        
        # Length metrics
        metrics['prediction_length'] = len(prediction.split())
        metrics['reference_length'] = len(reference.split())
        metrics['length_ratio'] = metrics['prediction_length'] / max(metrics['reference_length'], 1)
        
        return metrics
    
    def evaluate_dataset(self, predictions: List[str], references: List[str],
                        questions: List[str] = None) -> Dict[str, Any]:
        """
        Evaluate entire dataset
        
        Args:
            predictions: List of model predictions
            references: List of reference answers
            questions: Optional list of questions for context
        
        Returns:
            Dictionary with aggregated metrics
        """
        all_metrics = defaultdict(list)
        
        print(f"📊 Evaluating {len(predictions)} examples...")
        
        for i, (pred, ref) in enumerate(zip(predictions, references)):
            if i % 50 == 0:
                print(f"  Progress: {i}/{len(predictions)}")
            
            # Calculate metrics for this example
            metrics = self.calculate_all_metrics(pred, ref)
            
            # Store each metric
            for key, value in metrics.items():
                all_metrics[key].append(value)
        
        # Calculate BERTScore on entire dataset (more efficient)
        if self.use_bertscore:
            print("  Computing BERTScore...")
            bertscore_metrics = self._compute_bertscore(predictions, references)
            for key, value in bertscore_metrics.items():
                all_metrics[key] = [value]  # Already aggregated
        
        # Aggregate results
        aggregated = {}
        for key, values in all_metrics.items():
            if len(values) > 1:  # Not already aggregated
                aggregated[f'{key}_mean'] = np.mean(values)
                aggregated[f'{key}_std'] = np.std(values)
                aggregated[f'{key}_median'] = np.median(values)
            else:
                aggregated[key] = values[0]
        
        return aggregated
    
    def compare_models(self, results1: Dict[str, float], results2: Dict[str, float],
                      model1_name: str = "Model 1", model2_name: str = "Model 2") -> None:
        """
        Compare two models' results
        
        Args:
            results1: Metrics for first model
            results2: Metrics for second model
            model1_name: Name of first model
            model2_name: Name of second model
        """
        print("\n" + "=" * 80)
        print(f"Model Comparison: {model1_name} vs {model2_name}".center(80))
        print("=" * 80)
        
        # Get common metrics
        common_metrics = set(results1.keys()) & set(results2.keys())
        
        print(f"\n{'Metric':<30} {model1_name:<20} {model2_name:<20} {'Improvement':<10}")
        print("-" * 80)
        
        for metric in sorted(common_metrics):
            if '_mean' in metric:
                val1 = results1[metric]
                val2 = results2[metric]
                improvement = ((val2 - val1) / val1 * 100) if val1 > 0 else 0
                
                print(f"{metric:<30} {val1:<20.4f} {val2:<20.4f} {improvement:+.2f}%")
        
        print("=" * 80)


def calculate_perplexity(model, dataset) -> float:
    """Calculate perplexity on dataset"""
    # Placeholder - implement based on your model
    # This would require the actual model to compute log likelihoods
    pass
