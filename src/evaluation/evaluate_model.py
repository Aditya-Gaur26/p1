"""
Evaluate model on test questions (e.g., endsem questions)
"""

import sys
import json
from pathlib import Path
from typing import List, Dict, Any
import argparse
from datetime import datetime
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.config import config
from src.utils.helpers import save_json, load_json
from src.inference.query_processor import QueryProcessor


class ModelEvaluator:
    """Evaluate model performance on test questions"""
    
    def __init__(self, processor: QueryProcessor = None):
        if processor is None:
            self.processor = QueryProcessor(use_rag=True, use_enrichment=True)
        else:
            self.processor = processor
        
        # Initialize ROUGE scorer
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'],
            use_stemmer=True
        )
    
    def calculate_metrics(self, prediction: str, reference: str) -> Dict[str, float]:
        """Calculate evaluation metrics"""
        
        metrics = {}
        
        # ROUGE scores
        rouge_scores = self.rouge_scorer.score(reference, prediction)
        metrics['rouge1'] = rouge_scores['rouge1'].fmeasure
        metrics['rouge2'] = rouge_scores['rouge2'].fmeasure
        metrics['rougeL'] = rouge_scores['rougeL'].fmeasure
        
        # BLEU score
        reference_tokens = reference.split()
        prediction_tokens = prediction.split()
        smoothing = SmoothingFunction().method1
        metrics['bleu'] = sentence_bleu(
            [reference_tokens],
            prediction_tokens,
            smoothing_function=smoothing
        )
        
        # Simple word overlap (F1)
        ref_words = set(reference.lower().split())
        pred_words = set(prediction.lower().split())
        
        if pred_words:
            precision = len(ref_words & pred_words) / len(pred_words)
            recall = len(ref_words & pred_words) / len(ref_words) if ref_words else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics['precision'] = precision
            metrics['recall'] = recall
            metrics['f1'] = f1
        else:
            metrics['precision'] = 0
            metrics['recall'] = 0
            metrics['f1'] = 0
        
        return metrics
    
    def evaluate_question(self, question: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single question"""
        
        q_text = question.get('question', '')
        reference_answer = question.get('answer', '')
        
        print(f"\n📝 Evaluating: {q_text[:100]}...")
        
        # Get model answer
        result = self.processor.answer(q_text, include_enrichment=True)
        
        # Calculate metrics if reference answer exists
        metrics = {}
        if reference_answer:
            metrics = self.calculate_metrics(result['answer'], reference_answer)
            print(f"   ROUGE-L: {metrics['rougeL']:.3f} | F1: {metrics['f1']:.3f}")
        
        return {
            'question': q_text,
            'reference_answer': reference_answer,
            'model_answer': result['answer'],
            'metrics': metrics,
            'sources': result['sources'],
            'youtube_videos_count': len(result['youtube_videos']),
            'research_papers_count': len(result['research_papers']),
            'processing_time': result['processing_time'],
            'full_result': result
        }
    
    def evaluate_dataset(self, test_file: Path) -> Dict[str, Any]:
        """Evaluate on entire test dataset"""
        
        print("=" * 70)
        print("Model Evaluation".center(70))
        print("=" * 70)
        
        # Load test questions
        if not test_file.exists():
            raise FileNotFoundError(f"Test file not found: {test_file}")
        
        test_data = load_json(test_file)
        questions = test_data if isinstance(test_data, list) else test_data.get('questions', [])
        
        print(f"\n📊 Evaluating on {len(questions)} questions...")
        
        # Evaluate each question
        results = []
        for i, question in enumerate(questions, 1):
            print(f"\n[{i}/{len(questions)}]", end=" ")
            result = self.evaluate_question(question)
            results.append(result)
        
        # Calculate aggregate metrics
        aggregate_metrics = self._calculate_aggregate_metrics(results)
        
        # Create evaluation report
        report = {
            'timestamp': datetime.now().isoformat(),
            'test_file': str(test_file),
            'total_questions': len(questions),
            'aggregate_metrics': aggregate_metrics,
            'individual_results': results
        }
        
        return report
    
    def _calculate_aggregate_metrics(self, results: List[Dict]) -> Dict[str, Any]:
        """Calculate aggregate metrics from results"""
        
        # Filter results with metrics
        results_with_metrics = [r for r in results if r['metrics']]
        
        if not results_with_metrics:
            return {'note': 'No reference answers provided for metric calculation'}
        
        # Average metrics
        metrics = {}
        for key in ['rouge1', 'rouge2', 'rougeL', 'bleu', 'f1', 'precision', 'recall']:
            values = [r['metrics'][key] for r in results_with_metrics if key in r['metrics']]
            if values:
                metrics[f'avg_{key}'] = sum(values) / len(values)
        
        # Enrichment statistics
        total_youtube = sum(r['youtube_videos_count'] for r in results)
        total_papers = sum(r['research_papers_count'] for r in results)
        
        metrics['enrichment_stats'] = {
            'avg_youtube_per_question': total_youtube / len(results),
            'avg_papers_per_question': total_papers / len(results),
            'questions_with_youtube': sum(1 for r in results if r['youtube_videos_count'] > 0),
            'questions_with_papers': sum(1 for r in results if r['research_papers_count'] > 0),
        }
        
        # Performance statistics
        avg_time = sum(r['processing_time'] for r in results) / len(results)
        metrics['avg_processing_time'] = avg_time
        
        return metrics
    
    def print_report(self, report: Dict[str, Any]):
        """Print evaluation report"""
        
        print("\n\n" + "=" * 70)
        print("EVALUATION REPORT".center(70))
        print("=" * 70)
        
        print(f"\n📅 Date: {report['timestamp']}")
        print(f"📁 Test File: {report['test_file']}")
        print(f"📊 Total Questions: {report['total_questions']}")
        
        metrics = report['aggregate_metrics']
        
        if 'note' not in metrics:
            print("\n📈 Core Metrics:")
            print(f"  ROUGE-1:    {metrics.get('avg_rouge1', 0):.3f}")
            print(f"  ROUGE-2:    {metrics.get('avg_rouge2', 0):.3f}")
            print(f"  ROUGE-L:    {metrics.get('avg_rougeL', 0):.3f}")
            print(f"  BLEU:       {metrics.get('avg_bleu', 0):.3f}")
            print(f"  F1 Score:   {metrics.get('avg_f1', 0):.3f}")
        
        if 'enrichment_stats' in metrics:
            stats = metrics['enrichment_stats']
            print("\n🌟 Enrichment Features:")
            print(f"  Avg YouTube videos/question:  {stats['avg_youtube_per_question']:.1f}")
            print(f"  Avg Papers/question:          {stats['avg_papers_per_question']:.1f}")
            print(f"  Questions with YouTube:       {stats['questions_with_youtube']}/{report['total_questions']}")
            print(f"  Questions with Papers:        {stats['questions_with_papers']}/{report['total_questions']}")
        
        print(f"\n⏱️  Avg Processing Time: {metrics.get('avg_processing_time', 0):.2f}s")
        print("=" * 70)
    
    def save_report(self, report: Dict[str, Any], output_file: Path):
        """Save evaluation report to file"""
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        save_json(report, output_file)
        
        # Also save a summary
        summary_file = output_file.parent / f"{output_file.stem}_summary.txt"
        with open(summary_file, 'w') as f:
            # Redirect print to file
            import io
            from contextlib import redirect_stdout
            
            f_io = io.StringIO()
            with redirect_stdout(f_io):
                self.print_report(report)
            f.write(f_io.getvalue())
        
        print(f"\n💾 Report saved to: {output_file}")
        print(f"💾 Summary saved to: {summary_file}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on test questions")
    parser.add_argument(
        "--test-file",
        type=str,
        default="./data/evaluation/endsem_questions.json",
        help="Path to test questions file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for results"
    )
    
    args = parser.parse_args()
    
    # Default output file
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"./outputs/results/evaluation_{timestamp}.json"
    
    # Run evaluation
    evaluator = ModelEvaluator()
    report = evaluator.evaluate_dataset(Path(args.test_file))
    
    # Print report
    evaluator.print_report(report)
    
    # Save report
    evaluator.save_report(report, Path(args.output))


if __name__ == "__main__":
    main()
