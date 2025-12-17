# 🎯 COMPLETE PROJECT EXPLANATION

## Overview

You have a **RAG (Retrieval-Augmented Generation) system** that:
1. Reads your OS & Networks PDFs/slides
2. Fine-tunes Qwen2.5-7B model on that content
3. Answers questions by retrieving relevant context and generating responses

**Your Problem**: Model was trained once but **hallucinates** (makes up answers)

**This Document**: Complete explanation of workflow + how to fix hallucinations

---

## 🏗️ System Architecture

```
┌──────────────────┐
│  YOUR PDFs       │  962 + 675 pages
│  YOUR SLIDES     │  PowerPoint files
└────────┬─────────┘
         │
         ↓ [EXTRACT]
┌──────────────────┐
│  JSON TEXT DATA  │  All text extracted & cleaned
└────────┬─────────┘
         │
         ↓ [CREATE Q&A]
┌──────────────────┐
│  TRAINING DATA   │  3000-4000 Q&A pairs
│  train.jsonl     │  {"question": "...", "answer": "..."}
└────────┬─────────┘
         │
         ↓ [TRAIN]
┌──────────────────┐
│  FINE-TUNED      │  Qwen2.5 + Your Knowledge
│  MODEL           │  LoRA adapters (~500MB)
└────────┬─────────┘
         │
         ↓ [USE]
┌──────────────────┐
│  VECTOR DB       │  Chunks embedded for fast retrieval
│  + MODEL         │  RAG system
└────────┬─────────┘
         │
         ↓ [QUERY]
┌──────────────────┐
│  USER QUESTION   │  "What is virtual memory?"
│       ↓          │
│  RETRIEVE CHUNKS │  Find relevant PDF sections
│       ↓          │
│  GENERATE ANSWER │  Model uses retrieved context
│       ↓          │
│  FINAL ANSWER    │  Grounded in your PDFs
└──────────────────┘
```

---

## 📋 Complete Workflow (6 Phases)

### **PHASE 1: Extract Content** 

**What it does:**
- Reads PDFs page by page
- Extracts text from slides
- Cleans formatting (removes headers, page numbers)
- Saves as JSON

**Commands:**
```bash
python src/data_processing/extract_pdfs.py
python src/data_processing/extract_slides.py
```

**Output:**
- `data/processed/books/all_pdfs_combined.json`
- `data/processed/slides/all_slides_combined.json`

**Why important:**
- If extraction is bad (gibberish text) → Everything downstream fails
- **Check this manually!**

**Example output:**
```json
{
  "files": [
    {
      "filename": "OS_Textbook.pdf",
      "chunks": [
        "Process scheduling involves allocating CPU time to different processes...",
        "Virtual memory allows processes to use more memory than physically available..."
      ]
    }
  ]
}
```

---

### **PHASE 2: Create Training Dataset**

**What it does:**
- Reads extracted JSON from Phase 1
- For each text chunk:
  - Generates 3-5 questions using templates
  - Creates answers from the chunk content
  - Adds paraphrased versions
  - Formats as `Instruction → Response`
- Augments data (2x factor)
- Splits 90% train / 10% validation

**Command:**
```bash
python src/data_processing/create_dataset.py
```

**Output:**
- `data/processed/train.jsonl` (~3600 examples)
- `data/processed/val.jsonl` (~400 examples)

**Example training sample:**
```json
{
  "instruction": "What is the difference between preemptive and non-preemptive scheduling?",
  "input": "",
  "output": "Preemptive scheduling allows the OS to interrupt a running process and allocate CPU to another process (e.g., Round Robin). Non-preemptive scheduling runs a process until completion (e.g., FCFS). The key difference is that preemptive provides better response time but has context-switching overhead.",
  "text": "### Instruction:\nWhat is the difference between preemptive and non-preemptive scheduling?\n\n### Response:\nPreemptive scheduling allows..."
}
```

**🚨 CRITICAL CHECKPOINT:**

This is where most hallucinations come from! **You MUST check data quality.**

```bash
python diagnose.py
```

**What to look for:**
- ✅ Quality score > 70/100
- ✅ Answers are 100+ characters (not single words)
- ✅ Questions are specific (not "Explain the concept")
- ✅ Has "I don't know" examples (for out-of-scope questions)

**If score < 70 → FIX BEFORE TRAINING!**

---

### **PHASE 3: Train Model**

**What it does:**
1. Downloads Qwen2.5-7B-Instruct (~14GB) from HuggingFace
2. Quantizes to 4-bit (~3.5GB in VRAM)
3. Adds LoRA adapters (small trainable layers)
4. Trains for 3 epochs on your data
5. Saves checkpoints

**Command:**
```bash
python src/training/fine_tune.py
```

**Training Details:**

```yaml
# What's being trained:
Base Model: Qwen2.5-7B-Instruct (7 billion parameters)
Method: LoRA (Low-Rank Adaptation)
Trainable Params: ~64 million (0.9% of total)
Quantization: 4-bit (saves memory)

# Training config (optimized for RTX 3060 12GB):
Batch Size: 1 (per GPU)
Gradient Accumulation: 16 (effective batch = 16)
Learning Rate: 5e-5 (low to prevent forgetting)
Epochs: 3
Max Sequence Length: 1024 tokens
Optimizer: paged_adamw_8bit (memory efficient)
```

**Time:** 6-8 hours on RTX 3060

**What happens internally:**

```
Epoch 1:
  Step 1: Load batch → Forward pass → Calculate loss → Backward pass → Update LoRA weights
  Step 2: ...
  Step 200: Save checkpoint
  ...
  End of Epoch 1: Calculate validation loss

Epoch 2: Repeat...

Epoch 3: Repeat...

Final: Save best model based on lowest validation loss
```

**Monitor training:**

```bash
# GPU usage
nvidia-smi -l 1

# Training progress
tensorboard --logdir models/fine_tuned/logs
```

**What to watch:**

✅ **Good Training:**
```
Epoch 1: train_loss=2.5, eval_loss=2.3
Epoch 2: train_loss=1.8, eval_loss=1.7
Epoch 3: train_loss=1.2, eval_loss=1.3
```
Both losses decrease together (gap < 0.3)

❌ **Bad Training (Overfitting):**
```
Epoch 1: train_loss=2.5, eval_loss=2.3
Epoch 2: train_loss=1.2, eval_loss=2.4  ← eval_loss increases!
Epoch 3: train_loss=0.5, eval_loss=2.8  ← model memorizing
```
Train loss drops fast, eval loss increases = **HALLUCINATION GUARANTEED**

**Output:**
- `models/fine_tuned/adapter_model.bin` (~500MB)
- `models/fine_tuned/adapter_config.json`

---

### **PHASE 4: Build Vector Database**

**What it does:**
- Re-reads all PDF/slide content
- Chunks into 512-token segments (with 50-token overlap)
- Embeds each chunk using sentence-transformers
- Stores in ChromaDB for fast retrieval

**Command:**
```bash
python src/data_processing/build_vectordb.py
```

**Purpose: Retrieval-Augmented Generation (RAG)**

At inference time:
1. User asks: "What is paging?"
2. System searches vector DB for chunks about "paging"
3. Retrieves top 5 most relevant chunks
4. Sends chunks + question to model
5. Model generates answer BASED ON CHUNKS

**This prevents hallucination by grounding answers in your PDFs!**

**Output:**
- `data/vectordb/course_materials/` (ChromaDB)

**Check it worked:**
```bash
python diagnose.py
# Should show: "✅ 3000+ documents in vector DB"
```

---

### **PHASE 5: Inference (Using the Model)**

**Interactive Mode:**
```bash
python src/inference/query_processor.py
```

**What happens when you ask a question:**

```
User Input: "What is virtual memory?"
     ↓
[1] Query Expansion
    "virtual memory" → "virtual memory, paging, swap space"
     ↓
[2] Hybrid Retrieval (from Vector DB)
    • Dense: Semantic similarity using embeddings
    • Sparse: Keyword matching (BM25)
    • Reranking: Cross-encoder scoring
    → Retrieves top 5 chunks:
      1. "Virtual memory is a memory management technique..."
      2. "Paging divides virtual memory into fixed-size pages..."
      3. "The OS uses page tables to map virtual to physical..."
      4. "Page faults occur when accessing non-resident pages..."
      5. "Swap space on disk extends physical memory..."
     ↓
[3] Prompt Construction
    System: You are a teaching assistant. Answer based ONLY on context.
    Context: <5 chunks above>
    Question: What is virtual memory?
    Answer:
     ↓
[4] Model Generation
    Fine-tuned Qwen2.5 generates response using retrieved context
     ↓
[5] Output
    "Virtual memory is a memory management technique that allows
     processes to use more memory than physically available. The OS
     uses paging to divide virtual memory into fixed-size pages,
     mapped to physical memory via page tables. When a page is not
     in physical memory, a page fault occurs and the OS loads it
     from swap space on disk."
```

**Key insight:** Model can ONLY answer based on retrieved chunks. If retrieval fails, answer quality drops.

---

### **PHASE 6: Evaluation**

**Command:**
```bash
python src/evaluation/evaluate_model.py
```

**What it does:**
- Loads test questions
- Generates answers with your model
- Compares to reference answers
- Calculates metrics

**Metrics:**

| Metric | What it measures | Target |
|--------|------------------|--------|
| **BLEU** | Word-level overlap | > 40 |
| **ROUGE-L** | Longest common subsequence | > 0.6 |
| **BERTScore** | Semantic similarity | > 0.85 |
| **Faithfulness** | Answer grounded in context? | > 90% |

**Most Important: Faithfulness**
- If faithfulness < 80% → Model is hallucinating
- Means model invents facts not in retrieved context

---

## 🚨 Why Your Model Hallucinated

Based on your report, likely causes (in order of probability):

### 1. **Poor Training Data Quality** (70% chance)

**Problem:**
- Questions too generic: "Explain the concept from the course"
- Answers too short: Single sentences
- No specific details from your PDFs

**How this causes hallucination:**
- Model learns pattern: "When asked vague question → give vague answer"
- Model never learns to cite specific facts from PDFs
- Model fills gaps with plausible-sounding text

**Fix:**
```bash
# Check your data
python diagnose.py

# If score < 70, edit src/data_processing/create_dataset.py
# Then regenerate:
python src/data_processing/create_dataset.py
```

### 2. **No "I Don't Know" Examples** (20% chance)

**Problem:**
- Training data has 100% of questions answerable
- Model never learns to refuse

**How this causes hallucination:**
- User asks: "Explain quantum computing in OS"
- Model has never seen "I don't know" response
- Model generates plausible-sounding quantum+OS text

**Fix:**
Add 10-20 examples like:
```json
{
  "instruction": "What is the capital of France?",
  "output": "This question is outside the scope of the course materials on Operating Systems and Networks. I can only answer questions based on the provided lecture materials."
}
```

### 3. **Overfitting** (8% chance)

**Problem:**
- Too many epochs (>5)
- Too high learning rate (>1e-4)
- Model memorizes training set instead of learning concepts

**Signs:**
- Training loss very low (< 0.5)
- Validation loss high (> 2.0)
- Model repeats exact training examples but fails on variations

**Fix:**
```yaml
# configs/training_config.yaml
training:
  learning_rate: 2.0e-5  # Lower
  num_train_epochs: 2     # Fewer
  lora_dropout: 0.1       # More regularization
```

### 4. **Weak Retrieval** (2% chance)

**Problem:**
- Vector DB empty or has wrong chunks
- Retrieval always returns irrelevant context
- Model has no choice but to guess

**Fix:**
```bash
# Rebuild vector DB
python src/data_processing/build_vectordb.py

# Verify
python diagnose.py
# Should show: "✅ Vector database found: 3000+ documents"
```

---

## 🎯 Your Action Plan (Bulls-Eye Training)

### Step 1: Diagnose (5 minutes)

```bash
python diagnose.py
```

This will tell you:
- Training data quality score (0-100)
- What's wrong with your data
- What to fix

### Step 2: Fix Data (if needed)

**If quality score < 70:**

Edit `src/data_processing/create_dataset.py`:

**Change 1: Add minimum length**
```python
def generate_qa_pairs(content: str, source: str) -> List[Dict]:
    # Add this check
    if len(content) < 100:
        return []  # Skip short content
    
    # Rest of function...
```

**Change 2: Use specific questions**
```python
# Replace generic questions with specific ones
question = f"Explain {topic} as discussed in the lecture on {source}."
# Instead of: "Explain the concept"
```

**Change 3: Add refusal examples**
```python
# At end of create_dataset() function
refusal_examples = [
    {
        "instruction": "What is quantum entanglement?",
        "output": "This topic is not covered in the Operating Systems course materials.",
        "text": "### Instruction:\nWhat is quantum entanglement?\n\n### Response:\nThis topic is not covered..."
    },
    # Add 9 more...
]
all_training_data.extend(refusal_examples)
```

**Then regenerate:**
```bash
python src/data_processing/create_dataset.py
python diagnose.py  # Should now be > 70
```

### Step 3: Train Model (6-8 hours)

```bash
python src/training/fine_tune.py
```

**In another terminal, monitor:**
```bash
nvidia-smi -l 1
tensorboard --logdir models/fine_tuned/logs
```

**Stop training if:**
- eval_loss starts increasing
- GPU memory exceeds 11.5GB (RTX 3060 limit)

### Step 4: Build Vector DB (20 minutes)

```bash
python src/data_processing/build_vectordb.py
```

### Step 5: Test (manual)

```bash
python src/inference/query_processor.py
```

**Test cases:**

1. **Known question (from PDFs):**
   - Q: "What is process scheduling?"
   - Expected: Detailed answer with specific algorithms

2. **Out-of-scope question:**
   - Q: "What is blockchain?"
   - Expected: "Not covered in course materials"

3. **Adversarial question:**
   - Q: "Tell me about Windows 11 features"
   - Expected: Either refuses OR only mentions concepts from PDFs

### Step 6: Evaluate

```bash
python src/evaluation/evaluate_model.py
```

**Target scores:**
- Faithfulness > 90%
- BERTScore > 0.85

**If scores low:**
- Go back to Step 1
- Check training data quality again

---

## 📊 Key Files & Their Purpose

| File | Purpose | When to Check |
|------|---------|---------------|
| `data/processed/train.jsonl` | Training data | After create_dataset.py |
| `configs/training_config.yaml` | Hyperparameters | Before training |
| `models/fine_tuned/adapter_model.bin` | Trained model | After training |
| `data/vectordb/course_materials/` | RAG database | After build_vectordb.py |
| `docs/COMPLETE_WORKFLOW_AND_FIXES.md` | Detailed guide | When stuck |

---

## 🔧 Common Issues & Fixes

### "Model still hallucinates after retraining"

1. Check training data: `python diagnose.py`
2. If score < 70: Fix data and retrain
3. If score > 70: Add more "I don't know" examples
4. Lower learning rate to 2e-5

### "Training is too slow"

- Reduce epochs: 3 → 2
- Reduce eval frequency: `eval_steps: 200 → 500`
- Use fewer training examples (sample 2000 instead of 4000)

### "Out of memory (OOM) error"

```yaml
# configs/training_config.yaml
training:
  max_seq_length: 512        # Half of current
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 16
```

### "Model gives generic answers"

- Training data has generic answers
- Regenerate data with specific details
- Check that PDF extraction worked (Phase 1)

### "Wrong answers (not hallucination)"

- Vector DB retrieval failing
- Rebuild: `python src/data_processing/build_vectordb.py`
- Check chunking strategy (might be too small/large)

---

## 📚 Documentation Reading Order

1. **README.md** - Project overview
2. **QUICKSTART.md** - Basic setup & commands
3. **docs/WORKFLOW_VISUAL.md** - Visual pipeline (this file)
4. **docs/COMPLETE_WORKFLOW_AND_FIXES.md** - Deep dive & troubleshooting

---

## ✅ Success Checklist

Your model is **bulls-eye** when:

- [ ] `python diagnose.py` shows quality > 70
- [ ] Training eval_loss decreases smoothly
- [ ] Answers known questions correctly
- [ ] Refuses out-of-scope questions
- [ ] Faithfulness score > 90%
- [ ] Manually test 20 questions - all correct
- [ ] Cites sources (PDF sections) in answers

---

## 🎓 Next Steps

**Right now:**
```bash
python diagnose.py
```

This will show you exactly what's wrong!

**Then follow the action plan above.**

**Questions?** See `docs/COMPLETE_WORKFLOW_AND_FIXES.md` for detailed explanations.

---

**Your model will hit bulls-eye once your training data is high quality.** That's the #1 factor! 🎯
