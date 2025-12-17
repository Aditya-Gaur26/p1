"""
Diagnose Training Data Quality and Model Issues
Identifies why your model hallucinates

Usage: python diagnose.py
"""

import json
import sys
from pathlib import Path
from collections import Counter, defaultdict
import statistics

sys.path.append(str(Path(__file__).parent))
from src.utils.config import config


def analyze_training_data(file_path: Path):
    """Analyze training data quality"""
    
    print("=" * 70)
    print("TRAINING DATA QUALITY ANALYSIS".center(70))
    print("=" * 70)
    
    if not file_path.exists():
        print(f"\n❌ Training file not found: {file_path}")
        print("Run: python src/data_processing/create_dataset.py")
        return None
    
    # Load data
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except:
                pass
    
    if not data:
        print("\n❌ No data found in training file!")
        return None
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total samples: {len(data)}")
    
    # Analyze lengths
    instruction_lengths = [len(d.get('instruction', '')) for d in data]
    output_lengths = [len(d.get('output', '')) for d in data]
    
    print(f"\n📏 Length Analysis:")
    print(f"   Instruction length:")
    print(f"      Mean: {statistics.mean(instruction_lengths):.0f} chars")
    print(f"      Median: {statistics.median(instruction_lengths):.0f} chars")
    
    print(f"\n   Output length:")
    print(f"      Mean: {statistics.mean(output_lengths):.0f} chars")
    print(f"      Median: {statistics.median(output_lengths):.0f} chars")
    print(f"      Min: {min(output_lengths):.0f} chars")
    
    # Quality checks
    print(f"\n🔍 Quality Checks:")
    
    # Check 1: Too short answers
    short_answers = [d for d in data if len(d.get('output', '')) < 50]
    if short_answers:
        print(f"   ⚠️  {len(short_answers)} answers < 50 chars ({len(short_answers)/len(data)*100:.1f}%)")
        print(f"       Example: \"{short_answers[0].get('output', '')[:50]}\"")
    else:
        print(f"   ✅ All answers >= 50 chars")
    
    # Check 2: Generic questions
    generic_patterns = ['explain the concept', 'describe the following', 'what is this']
    generic_questions = [d for d in data if any(p in d.get('instruction', '').lower() for p in generic_patterns)]
    if generic_questions:
        print(f"   ⚠️  {len(generic_questions)} generic questions ({len(generic_questions)/len(data)*100:.1f}%)")
        print(f"       Example: \"{generic_questions[0].get('instruction', '')}\"")
    else:
        print(f"   ✅ Questions are specific")
    
    # Check 3: Duplicates
    instruction_counter = Counter([d.get('instruction', '') for d in data])
    duplicates = [k for k, v in instruction_counter.items() if v > 1]
    if duplicates:
        print(f"   ⚠️  {len(duplicates)} duplicate questions")
    else:
        print(f"   ✅ No duplicates")
    
    # Check 4: Question diversity
    question_starters = defaultdict(int)
    for d in data:
        instruction = d.get('instruction', '')
        if instruction:
            starter = ' '.join(instruction.split()[:2]).lower()
            question_starters[starter] += 1
    
    print(f"\n📝 Question Diversity (Top 5):")
    top_starters = sorted(question_starters.items(), key=lambda x: x[1], reverse=True)[:5]
    for starter, count in top_starters:
        pct = (count / len(data)) * 100
        print(f"   '{starter}...': {count} ({pct:.1f}%)")
    
    if top_starters[0][1] / len(data) > 0.5:
        print(f"   ⚠️  >50% questions start same way → Low diversity")
    else:
        print(f"   ✅ Good diversity")
    
    # Check 5: "I don't know" examples
    refusal_keywords = ["outside the scope", "not covered", "cannot answer", "don't have information"]
    refusal_examples = [d for d in data if any(k in d.get('output', '').lower() for k in refusal_keywords)]
    
    print(f"\n🚫 Refusal Examples:")
    if refusal_examples:
        print(f"   ✅ {len(refusal_examples)} examples ({len(refusal_examples)/len(data)*100:.1f}%)")
    else:
        print(f"   ⚠️  NO REFUSAL EXAMPLES → Will hallucinate!")
        print(f"       Add 10+ examples where model says 'I don't know'")
    
    # Overall score
    print(f"\n" + "=" * 70)
    print("QUALITY SCORE".center(70))
    print("=" * 70)
    
    score = 100
    
    if len(short_answers) / len(data) > 0.1:
        score -= 20
        print(f"   -20: >10% answers too short")
    
    if len(generic_questions) / len(data) > 0.2:
        score -= 15
        print(f"   -15: >20% questions generic")
    
    if len(duplicates) > len(data) * 0.05:
        score -= 10
        print(f"   -10: >5% duplicates")
    
    if top_starters[0][1] / len(data) > 0.5:
        score -= 15
        print(f"   -15: Low question diversity")
    
    if not refusal_examples:
        score -= 25
        print(f"   -25: NO refusal examples")
    
    print(f"\n{'=' * 70}")
    if score >= 90:
        print(f"   🎯 EXCELLENT ({score}/100) - Ready to train!")
    elif score >= 70:
        print(f"   ✅ GOOD ({score}/100) - Minor improvements recommended")
    elif score >= 50:
        print(f"   ⚠️  FAIR ({score}/100) - Fix issues before training")
    else:
        print(f"   ❌ POOR ({score}/100) - Must improve data quality!")
    print(f"{'=' * 70}")
    
    # Show samples
    print(f"\n📋 Sample Examples:")
    for i in range(min(2, len(data))):
        print(f"\n   Example {i+1}:")
        print(f"   Q: {data[i].get('instruction', '')[:70]}...")
        print(f"   A: {data[i].get('output', '')[:120]}...")
    
    return score


def check_config():
    """Check training configuration"""
    
    print("\n" + "=" * 70)
    print("CONFIGURATION CHECK".center(70))
    print("=" * 70)
    
    try:
        cfg = config.get_training_config()
    except:
        print("\n⚠️  Could not load config")
        return
    
    # LoRA
    lora_r = cfg.get('lora', {}).get('r', 0)
    if lora_r >= 32:
        print(f"   ✅ LoRA rank: {lora_r}")
    else:
        print(f"   ⚠️  LoRA rank: {lora_r} → Increase to 32+")
    
    # Learning rate
    lr = cfg.get('training', {}).get('learning_rate', 0)
    if 1e-5 <= lr <= 1e-4:
        print(f"   ✅ Learning rate: {lr}")
    else:
        print(f"   ⚠️  Learning rate: {lr} → Use 2e-5 to 1e-4")
    
    # Epochs
    epochs = cfg.get('training', {}).get('num_train_epochs', 0)
    if 2 <= epochs <= 4:
        print(f"   ✅ Epochs: {epochs}")
    else:
        print(f"   ⚠️  Epochs: {epochs} → Use 2-4 for large dataset")
    
    # Sequence length
    max_seq = cfg.get('training', {}).get('max_seq_length', 0)
    if 512 <= max_seq <= 1024:
        print(f"   ✅ Max seq length: {max_seq}")
    else:
        print(f"   ⚠️  Max seq length: {max_seq} → Use 512-1024")


def check_vectordb():
    """Check vector database"""
    
    print("\n" + "=" * 70)
    print("VECTOR DATABASE CHECK".center(70))
    print("=" * 70)
    
    vectordb_path = config.vectordb_dir / "course_materials"
    
    if not vectordb_path.exists():
        print(f"\n❌ Not found: {vectordb_path}")
        print(f"   Run: python src/data_processing/build_vectordb.py")
        return
    
    try:
        import chromadb
        from chromadb.config import Settings
        
        client = chromadb.PersistentClient(
            path=str(vectordb_path),
            settings=Settings(anonymized_telemetry=False)
        )
        
        collection = client.get_collection("course_materials")
        count = collection.count()
        
        if count == 0:
            print(f"   ❌ Database empty!")
        elif count < 100:
            print(f"   ⚠️  Only {count} documents → Expected 1000+")
        else:
            print(f"   ✅ {count} documents")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")


def suggest_fixes(score):
    """Suggest fixes"""
    
    print("\n" + "=" * 70)
    print("RECOMMENDED ACTIONS".center(70))
    print("=" * 70)
    
    if score is None:
        print(f"\n1. Extract PDFs:")
        print(f"   python src/data_processing/extract_pdfs.py")
        print(f"\n2. Extract slides:")
        print(f"   python src/data_processing/extract_slides.py")
        print(f"\n3. Create dataset:")
        print(f"   python src/data_processing/create_dataset.py")
        return
    
    if score < 70:
        print(f"\n🔴 CRITICAL: Fix data quality first!")
        print(f"\n1. Edit src/data_processing/create_dataset.py:")
        print(f"   • Add minimum length: if len(content) < 100: skip")
        print(f"   • Use specific questions with topic names")
        print(f"   • Add 10 'I don't know' examples")
        print(f"\n2. Regenerate:")
        print(f"   python src/data_processing/create_dataset.py")
        print(f"\n3. Re-run this diagnostic")
    else:
        print(f"\n🟢 Proceed with training!")
        print(f"\n1. Build vector DB:")
        print(f"   python src/data_processing/build_vectordb.py")
        print(f"\n2. Train model:")
        print(f"   python src/training/fine_tune.py")
        print(f"\n3. Monitor with TensorBoard:")
        print(f"   tensorboard --logdir models/fine_tuned/logs")
    
    print(f"\n📚 Full guide: docs/COMPLETE_WORKFLOW_AND_FIXES.md")


def main():
    """Run diagnostic"""
    
    print("\n" + "=" * 70)
    print("🔍 ANTI-HALLUCINATION DIAGNOSTIC".center(70))
    print("=" * 70)
    
    train_file = config.data_dir / "processed" / "train.jsonl"
    
    score = analyze_training_data(train_file)
    
    if score is not None:
        check_config()
        check_vectordb()
    
    suggest_fixes(score)
    
    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
