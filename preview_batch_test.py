"""
Preview the batch testing setup - shows what will be tested
"""

import json

def preview_batch_test():
    """Show what questions will be tested"""
    
    questions_file = "test_questions.json"
    
    print("=" * 70)
    print("BATCH TEST PREVIEW".center(70))
    print("=" * 70)
    print()
    
    # Load questions
    with open(questions_file, 'r', encoding='utf-8') as f:
        questions = json.load(f)
    
    print(f"📋 Total questions: {len(questions)}\n")
    
    # Group by difficulty
    by_difficulty = {}
    by_topic = {}
    
    for q in questions:
        diff = q.get('difficulty', 'unknown')
        topic = q.get('topic', 'unknown')
        by_difficulty[diff] = by_difficulty.get(diff, 0) + 1
        by_topic[topic] = by_topic.get(topic, 0) + 1
    
    print("📊 Breakdown by difficulty:")
    for diff, count in sorted(by_difficulty.items()):
        print(f"   {diff}: {count}")
    
    print(f"\n📚 Breakdown by topic:")
    for topic, count in sorted(by_topic.items()):
        print(f"   {topic}: {count}")
    
    print("\n" + "=" * 70)
    print("QUESTIONS TO BE TESTED:")
    print("=" * 70)
    
    for i, q in enumerate(questions, 1):
        print(f"\n[{i}] {q.get('topic', 'Unknown Topic')} ({q.get('difficulty', 'N/A')})")
        print(f"Q: {q['question'][:100]}..." if len(q['question']) > 100 else f"Q: {q['question']}")
    
    print("\n" + "=" * 70)
    print(f"\n⏱️  Estimated time: ~{len(questions) * 10} seconds (~10s per question)")
    print(f"💾 Output will be saved to: test_results.json")
    print("\nTo run the batch test:")
    print("  python batch_test_model.py")


if __name__ == "__main__":
    preview_batch_test()
