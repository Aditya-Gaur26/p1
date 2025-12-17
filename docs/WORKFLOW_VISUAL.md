# 🎯 Project Workflow - Visual Overview

## Complete Pipeline (6 Phases)

```
┌─────────────────────────────────────────────────────────────────────┐
│                    YOUR RAW MATERIALS                                │
│  📄 PDFs (962 + 675 pages)   📊 PowerPoint Slides                   │
│  └─ data/raw/pdfs/            └─ data/raw/slides/                   │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE 1: EXTRACTION  ⏱️ ~30 min                                     │
│  ─────────────────────────────────────────────────────────────────  │
│  Command: python src/data_processing/extract_pdfs.py                │
│           python src/data_processing/extract_slides.py              │
│  ─────────────────────────────────────────────────────────────────  │
│  What happens:                                                       │
│  • Reads PDFs page by page                                           │
│  • Extracts text from slides + speaker notes                         │
│  • Removes headers, footers, page numbers                            │
│  • Chunks content semantically                                       │
│  ─────────────────────────────────────────────────────────────────  │
│  Output:                                                             │
│  ✓ data/processed/books/all_pdfs_combined.json                      │
│  ✓ data/processed/slides/all_slides_combined.json                   │
│  ─────────────────────────────────────────────────────────────────  │
│  ⚠️  CHECK: Open JSON files → Verify text is readable                │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE 2: DATASET CREATION  ⏱️ ~15 min                               │
│  ─────────────────────────────────────────────────────────────────  │
│  Command: python src/data_processing/create_dataset.py              │
│  ─────────────────────────────────────────────────────────────────  │
│  What happens:                                                       │
│  • Reads extracted content from Phase 1                              │
│  • For each chunk:                                                   │
│    ├─ Generates 3-5 questions (diverse types)                        │
│    ├─ Creates paraphrased versions                                   │
│    ├─ Adds reasoning chains to answers                               │
│    └─ Formats as: Instruction → Response                             │
│  • Applies data augmentation (2x)                                    │
│  • Splits 90% train / 10% validation                                 │
│  ─────────────────────────────────────────────────────────────────  │
│  Output:                                                             │
│  ✓ data/processed/train.jsonl    (3600 examples)                    │
│  ✓ data/processed/val.jsonl      (400 examples)                     │
│  ─────────────────────────────────────────────────────────────────  │
│  🔍 CRITICAL CHECKPOINT: Run diagnostic!                             │
│     python diagnose.py                                               │
│  ─────────────────────────────────────────────────────────────────  │
│  Quality Score Target: 70+ / 100                                     │
│  If < 70 → FIX DATA BEFORE TRAINING!                                 │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE 3: TRAINING  ⏱️ ~6-8 hours (RTX 3060)                         │
│  ─────────────────────────────────────────────────────────────────  │
│  Command: python src/training/fine_tune.py                           │
│  ─────────────────────────────────────────────────────────────────  │
│  What happens:                                                       │
│  1. Loads Qwen2.5-7B-Instruct (base model)                          │
│     • Downloads from HuggingFace (~14GB)                             │
│     • Quantizes to 4-bit (~3.5GB in VRAM)                            │
│  2. Adds LoRA adapters                                               │
│     • r=32 → ~64M trainable parameters                               │
│     • Targets: q_proj, k_proj, v_proj, o_proj, gates, embeddings    │
│  3. Trains for 3 epochs                                              │
│     • Batch size=1, grad accumulation=16 (effective batch=16)        │
│     • Learning rate=5e-5 with cosine schedule                        │
│     • Gradient checkpointing + 4-bit optimizer                       │
│  4. Saves checkpoints every 200 steps                                │
│  ─────────────────────────────────────────────────────────────────  │
│  Output:                                                             │
│  ✓ models/fine_tuned/adapter_model.bin  (~500MB)                    │
│  ✓ models/fine_tuned/adapter_config.json                            │
│  ✓ models/fine_tuned/logs/ (TensorBoard)                            │
│  ─────────────────────────────────────────────────────────────────  │
│  📊 MONITOR TRAINING:                                                │
│  • GPU usage:     nvidia-smi -l 1                                    │
│  • Training logs: tensorboard --logdir models/fine_tuned/logs        │
│  • Watch for: eval_loss should decrease like train_loss             │
│  ─────────────────────────────────────────────────────────────────  │
│  ⚠️  STOP IF: eval_loss increases while train_loss decreases         │
│              (= overfitting → model memorizing, not learning)        │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE 4: VECTOR DATABASE  ⏱️ ~20 min                                │
│  ─────────────────────────────────────────────────────────────────  │
│  Command: python src/data_processing/build_vectordb.py              │
│  ─────────────────────────────────────────────────────────────────  │
│  What happens:                                                       │
│  • Re-reads all PDF/slide content                                    │
│  • Chunks into 512-token segments (overlap=50)                       │
│  • Embeds each chunk with sentence-transformers                      │
│  • Stores vectors in ChromaDB                                        │
│  ─────────────────────────────────────────────────────────────────  │
│  Output:                                                             │
│  ✓ data/vectordb/course_materials/ (ChromaDB)                       │
│  ─────────────────────────────────────────────────────────────────  │
│  Purpose: Retrieval-Augmented Generation (RAG)                       │
│  • At inference: retrieves relevant context for each question        │
│  • Prevents hallucination by grounding answers in PDFs               │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE 5: INFERENCE  ⏱️ ~2 sec per query                             │
│  ─────────────────────────────────────────────────────────────────  │
│  Command: python src/inference/query_processor.py                   │
│  ─────────────────────────────────────────────────────────────────  │
│  User Query: "What is virtual memory?"                               │
│       ↓                                                              │
│  1. Query Expansion                                                  │
│     "What is virtual memory?" → "virtual memory, paging, swap"       │
│       ↓                                                              │
│  2. Hybrid Retrieval (RAG)                                           │
│     • Dense: Semantic similarity (embeddings)                        │
│     • Sparse: Keyword matching (BM25)                                │
│     • Reranking: Cross-encoder scoring                               │
│     → Top 5 most relevant chunks from vector DB                      │
│       ↓                                                              │
│  3. Prompt Construction                                              │
│     Context: <5 retrieved chunks>                                    │
│     Question: What is virtual memory?                                │
│     Answer based on context:                                         │
│       ↓                                                              │
│  4. Model Generation                                                 │
│     Fine-tuned Qwen2.5 generates answer                              │
│     • Uses LoRA adapters trained on your data                        │
│     • Constrained to context from vector DB                          │
│       ↓                                                              │
│  5. Post-processing                                                  │
│     • Removes repetitions                                            │
│     • Cites sources (PDF page numbers)                               │
│       ↓                                                              │
│  Output: "Virtual memory is a memory management technique..."        │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE 6: EVALUATION  ⏱️ ~10 min                                     │
│  ─────────────────────────────────────────────────────────────────  │
│  Command: python src/evaluation/evaluate_model.py                   │
│  ─────────────────────────────────────────────────────────────────  │
│  What happens:                                                       │
│  • Loads test questions from data/evaluation/endsem_questions.json  │
│  • Generates answers with your fine-tuned model                      │
│  • Compares with reference answers                                   │
│  • Calculates metrics:                                               │
│    ├─ BLEU: Word-level overlap (0-100)                               │
│    ├─ ROUGE-L: Longest common subsequence                            │
│    ├─ BERTScore: Semantic similarity                                 │
│    └─ Faithfulness: Grounded in context?                             │
│  ─────────────────────────────────────────────────────────────────  │
│  Target Scores (Good Model):                                         │
│  • BLEU > 40                                                         │
│  • ROUGE-L > 0.6                                                     │
│  • BERTScore > 0.85                                                  │
│  • Faithfulness > 90%                                                │
│  ─────────────────────────────────────────────────────────────────  │
│  If scores low:                                                      │
│  → Check training data quality (python diagnose.py)                  │
│  → Try different hyperparameters                                     │
│  → Add more training examples                                        │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🚨 Why Models Hallucinate

### Root Cause Analysis:

```
LOW QUALITY DATA (50% of cases)
  ├─ Short answers (< 50 chars)
  ├─ Generic questions ("What is this?")
  ├─ No specific details from PDFs
  └─ Fix: Improve create_dataset.py

OVERFITTING (30% of cases)
  ├─ Too many epochs (>5)
  ├─ High learning rate (>1e-4)
  ├─ Low LoRA rank (<16)
  └─ Fix: Lower LR, fewer epochs, increase r

NO REFUSAL EXAMPLES (15% of cases)
  ├─ Model never trained to say "I don't know"
  ├─ Generates plausible-sounding text when uncertain
  └─ Fix: Add 10-20 out-of-scope Q&A with refusals

WEAK RETRIEVAL (5% of cases)
  ├─ Vector DB empty or wrong chunks
  ├─ Retrieves irrelevant context
  └─ Fix: Rebuild vector DB, improve chunking
```

---

## 🎯 Your Action Plan (Bulls-Eye Training)

### Current Status:
- ✅ Config optimized for RTX 3060 12GB
- ✅ Memory settings tuned (batch=1, seq=1024)
- ⚠️  Training data quality: UNKNOWN

### Step-by-Step:

#### 1. **Diagnose Current State** (5 min)
```bash
python diagnose.py
```
This checks:
- Training data quality (score /100)
- Config settings
- Vector DB status

#### 2. **Fix Data Issues** (if score < 70)
Edit [src/data_processing/create_dataset.py](src/data_processing/create_dataset.py):
- Line 130: Add minimum length check
- Line 140: Use specific question templates
- Line 280: Add refusal examples

Then regenerate:
```bash
python src/data_processing/create_dataset.py
python diagnose.py  # Re-check
```

#### 3. **Train Model** (6-8 hours)
```bash
# Start training
python src/training/fine_tune.py

# In another terminal, monitor:
nvidia-smi -l 1
tensorboard --logdir models/fine_tuned/logs
```

Watch for:
- ✅ `train_loss`: 2.5 → 1.8 → 1.2 (smooth decrease)
- ✅ `eval_loss`: 2.3 → 1.7 → 1.3 (follows train_loss)
- ❌ If `eval_loss` increases → STOP (overfitting)

#### 4. **Build Vector DB** (20 min)
```bash
python src/data_processing/build_vectordb.py
```

#### 5. **Test Model** (manual)
```bash
python src/inference/query_processor.py
```

Test cases:
1. Question from PDFs → Should answer correctly
2. Detailed technical question → Should cite sources
3. Out-of-scope question → Should refuse

#### 6. **Evaluate** (10 min)
```bash
python src/evaluation/evaluate_model.py
```

Target: Faithfulness > 90%

---

## 📊 Quick Reference

### Files to Monitor:

| File | Purpose | Check |
|------|---------|-------|
| `data/processed/train.jsonl` | Training data | Quality score > 70 |
| `configs/training_config.yaml` | Hyperparameters | LR=5e-5, r=32, epochs=3 |
| `models/fine_tuned/logs/` | Training metrics | eval_loss decreases |
| `data/vectordb/course_materials/` | RAG database | Document count > 1000 |

### Key Commands:

```bash
# Full pipeline (if starting fresh)
python src/data_processing/extract_pdfs.py
python src/data_processing/extract_slides.py
python src/data_processing/create_dataset.py
python diagnose.py                              # CHECK DATA!
python src/training/fine_tune.py
python src/data_processing/build_vectordb.py
python src/inference/query_processor.py

# Just training (if data exists)
python diagnose.py                              # ALWAYS check first!
python src/training/fine_tune.py
```

### Troubleshooting:

| Problem | Solution |
|---------|----------|
| OOM error | Reduce `max_seq_length: 1024 → 512` |
| Hallucinating | Add refusal examples, lower LR |
| Generic answers | Improve training data specificity |
| Slow training | Reduce to 2 epochs |
| Wrong answers | Check vector DB retrieval |

---

## 📚 Documentation Guide

Read in this order:
1. [QUICKSTART.md](../QUICKSTART.md) - Basic setup & commands
2. **WORKFLOW_VISUAL.md** (this file) - Understanding the pipeline
3. [COMPLETE_WORKFLOW_AND_FIXES.md](COMPLETE_WORKFLOW_AND_FIXES.md) - Deep dive into hallucination fixes

---

**Next Step**: Run `python diagnose.py` to check your training data! 🎯
