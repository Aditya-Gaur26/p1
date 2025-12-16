"""
Test script to verify all improvements are working correctly
Run this after installing dependencies to ensure everything is set up
"""

import sys
from pathlib import Path
import time

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def test_imports():
    """Test that all new modules can be imported"""
    print("\n" + "="*60)
    print("Testing Module Imports".center(60))
    print("="*60)
    
    tests = [
        ("Data Augmentation", "from src.data_processing.data_augmentation import DataAugmentor, SemanticChunker"),
        ("Hybrid Retriever", "from src.inference.hybrid_retriever import HybridRetriever, QueryExpander"),
        ("Cached Processor", "from src.inference.cached_processor import CachedQueryProcessor, LRUCache"),
        ("Advanced Metrics", "from src.evaluation.advanced_metrics import AdvancedEvaluator"),
        ("Training Techniques", "from src.training.advanced_techniques import CurriculumLearner, ContrastiveLearner"),
        ("Text Preprocessing", "from src.utils.text_preprocessing import AdvancedTextCleaner"),
    ]
    
    passed = 0
    for name, import_stmt in tests:
        try:
            exec(import_stmt)
            print(f"✓ {name:<25} OK")
            passed += 1
        except Exception as e:
            print(f"✗ {name:<25} FAILED: {str(e)[:40]}")
    
    print(f"\nImport Tests: {passed}/{len(tests)} passed")
    return passed == len(tests)


def test_query_expansion():
    """Test query expansion functionality"""
    print("\n" + "="*60)
    print("Testing Query Expansion".center(60))
    print("="*60)
    
    try:
        from src.inference.hybrid_retriever import QueryExpander
        
        expander = QueryExpander()
        test_cases = [
            ("What is TCP?", "TCP"),
            ("Explain process scheduling", "process"),
            ("How does DNS work?", "DNS"),
        ]
        
        passed = 0
        for original, keyword in test_cases:
            expanded = expander.expand_query(original)
            if keyword.lower() in expanded.lower() and len(expanded) > len(original):
                print(f"✓ Expanded: {original[:30]}")
                print(f"  → {expanded[:80]}...")
                passed += 1
            else:
                print(f"✗ Failed to expand: {original}")
        
        print(f"\nQuery Expansion: {passed}/{len(test_cases)} passed")
        return passed == len(test_cases)
    
    except Exception as e:
        print(f"✗ Query expansion test failed: {e}")
        return False


def test_text_cleaning():
    """Test text preprocessing"""
    print("\n" + "="*60)
    print("Testing Text Preprocessing".center(60))
    print("="*60)
    
    try:
        from src.utils.text_preprocessing import AdvancedTextCleaner
        
        cleaner = AdvancedTextCleaner()
        test_cases = [
            ("Visit  http://example.com  now!", "Visit now!"),
            ("Email: test@test.com  ", "Email:"),
            ("Hellllo   world!", "Hello world!"),
        ]
        
        passed = 0
        for dirty, expected_contains in test_cases:
            clean = cleaner.clean_text(dirty)
            if expected_contains in clean and len(clean) < len(dirty):
                print(f"✓ Cleaned: {dirty[:40]} → {clean[:40]}")
                passed += 1
            else:
                print(f"✗ Cleaning failed for: {dirty}")
        
        print(f"\nText Cleaning: {passed}/{len(test_cases)} passed")
        return passed == len(test_cases)
    
    except Exception as e:
        print(f"✗ Text cleaning test failed: {e}")
        return False


def test_caching():
    """Test LRU cache functionality"""
    print("\n" + "="*60)
    print("Testing Cache Performance".center(60))
    print("="*60)
    
    try:
        from src.inference.cached_processor import LRUCache
        
        cache = LRUCache(capacity=3)
        
        # Test cache operations
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        cache.put("key3", "value3")
        
        # Test hit
        start = time.time()
        val = cache.get("key1")
        hit_time = (time.time() - start) * 1000
        
        # Test miss
        start = time.time()
        miss = cache.get("nonexistent")
        miss_time = (time.time() - start) * 1000
        
        # Test LRU eviction
        cache.put("key4", "value4")  # Should evict key2
        evicted = cache.get("key2")
        
        if val == "value1" and miss is None and evicted is None:
            print(f"✓ Cache operations working correctly")
            print(f"  Cache hit time: {hit_time:.3f}ms")
            print(f"  Cache miss time: {miss_time:.3f}ms")
            print(f"  LRU eviction: Working")
            return True
        else:
            print(f"✗ Cache operations failed")
            return False
    
    except Exception as e:
        print(f"✗ Cache test failed: {e}")
        return False


def test_data_augmentation():
    """Test data augmentation"""
    print("\n" + "="*60)
    print("Testing Data Augmentation".center(60))
    print("="*60)
    
    try:
        from src.data_processing.data_augmentation import DataAugmentor
        
        augmentor = DataAugmentor()
        
        # Test paraphrasing
        question = "What is TCP?"
        paraphrases = augmentor.paraphrase_question(question)
        
        # Test reasoning chain
        answer = "This is a test answer. " * 20
        enhanced = augmentor.add_reasoning_chain("", answer)
        
        if len(paraphrases) > 1 and "step by step" in enhanced.lower():
            print(f"✓ Data augmentation working")
            print(f"  Paraphrases generated: {len(paraphrases)}")
            print(f"  Reasoning chain added: {'Yes' if len(enhanced) > len(answer) else 'No'}")
            return True
        else:
            print(f"✗ Data augmentation failed")
            return False
    
    except Exception as e:
        print(f"✗ Data augmentation test failed: {e}")
        return False


def test_semantic_chunking():
    """Test semantic chunking"""
    print("\n" + "="*60)
    print("Testing Semantic Chunking".center(60))
    print("="*60)
    
    try:
        from src.data_processing.data_augmentation import SemanticChunker
        
        chunker = SemanticChunker(chunk_size=50, overlap=10)
        
        # Test text
        text = "This is sentence one. This is sentence two. This is sentence three. " * 5
        
        chunks = chunker.chunk_by_sentences(text)
        
        if len(chunks) >= 2:
            print(f"✓ Semantic chunking working")
            print(f"  Input length: {len(text.split())} words")
            print(f"  Chunks created: {len(chunks)}")
            print(f"  Avg chunk size: {sum(len(c.split()) for c in chunks) / len(chunks):.1f} words")
            return True
        else:
            print(f"✗ Semantic chunking failed")
            return False
    
    except Exception as e:
        print(f"✗ Semantic chunking test failed: {e}")
        return False


def test_curriculum_learning():
    """Test curriculum learning"""
    print("\n" + "="*60)
    print("Testing Curriculum Learning".center(60))
    print("="*60)
    
    try:
        from src.training.advanced_techniques import CurriculumLearner
        
        curriculum = CurriculumLearner(difficulty_metric="length")
        
        # Create test dataset
        test_data = [
            {"instruction": "Q1" * 5, "output": "A1" * 50},   # Easy
            {"instruction": "Q2" * 20, "output": "A2" * 200}, # Hard
            {"instruction": "Q3" * 10, "output": "A3" * 100}, # Medium
        ]
        
        sorted_data = curriculum.sort_by_difficulty(test_data)
        difficulties = [d for _, d in sorted_data]
        
        if difficulties == sorted(difficulties):
            print(f"✓ Curriculum learning working")
            print(f"  Difficulties: {[f'{d:.2f}' for d in difficulties]}")
            print(f"  Sorted correctly: Yes")
            return True
        else:
            print(f"✗ Curriculum learning failed")
            return False
    
    except Exception as e:
        print(f"✗ Curriculum learning test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("🚀 Testing Project Improvements".center(60))
    print("="*60)
    print("\nThis will test all newly implemented features...")
    
    results = []
    
    # Run tests
    results.append(("Module Imports", test_imports()))
    results.append(("Query Expansion", test_query_expansion()))
    results.append(("Text Cleaning", test_text_cleaning()))
    results.append(("Caching", test_caching()))
    results.append(("Data Augmentation", test_data_augmentation()))
    results.append(("Semantic Chunking", test_semantic_chunking()))
    results.append(("Curriculum Learning", test_curriculum_learning()))
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary".center(60))
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}  {name}")
    
    print(f"\n{'='*60}")
    print(f"Total: {passed}/{total} tests passed ({passed/total*100:.0f}%)".center(60))
    print(f"{'='*60}")
    
    if passed == total:
        print("\n🎉 All improvements are working correctly!")
        print("✅ You're ready to use the enhanced features!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        print("Please check the error messages above and ensure:")
        print("  1. All dependencies are installed: pip install -r requirements.txt")
        print("  2. Python version >= 3.8")
        print("  3. Project structure is intact")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
