# 🎯 Complete Workflow & Anti-Hallucination Guide

**Goal**: Train Qwen2.5 model that gives accurate, grounded answers—NOT hallucinations.

---

## 🔍 Why Your Model Hallucinates (Root Causes)

### 1. **Poor Quality Training Data** ⚠️ MOST COMMON
- **Problem**: Short, vague, or generic Q&A pairs
- **Example Bad**:
  ```json
  {"question": "Explain the concept", "answer": "This is about operating systems"}
  ```
- **Example Good**:
  ```json
  {
    "question": "What is the difference between preemptive and non-preemptive scheduling?",
    "answer": "Preemptive scheduling allows the OS to interrupt a running process and allocate CPU to another process. Example: Round Robin. Non-preemptive scheduling runs a process until completion. Example: FCFS. Key difference: preemptive provides better response time but has context-switching overhead."
  }
  ```

### 2. **Model Learns to "Make Things Up"**
- **Problem**: Training data has inconsistent sources
- Model learns patterns: "When I don't know → generate plausible-sounding text"
- **Fix**: All answers must come DIRECTLY from your PDFs/slides

### 3. **Weak Retrieval (RAG Issues)**
- **Problem**: RAG retrieves wrong context → Model answers based on wrong info
- **Fix**: Improve chunking + retrieval

### 4. **Overfitting on Training Format**
- **Problem**: Model memorizes question patterns, not knowledge
- **Fix**: Diverse question templates + augmentation

### 5. **Low LoRA Rank**
- **Problem**: r=8 or r=16 → Too few learnable parameters
- **Fix**: Use r=32 or r=64 for complex domains

---

## 📊 Complete Workflow (6 Steps)

### **PHASE 1: Data Preparation** 🏗️

#### Step 1.1: Extract Content from PDFs
```bash
python src/data_processing/extract_pdfs.py
```

**What it does:**
- Reads PDFs from `data/raw/pdfs/`
- Extracts text page by page
- Saves to `data/processed/books/all_pdfs_combined.json`

**Output format:**
```json
{
  "files": [
    {
      "filename": "OS_Textbook.pdf",
      "chunks": [
        "Process scheduling is...",
        "Memory management involves..."
      ]
    }
  ]
}
```

**⚠️ Check Quality:**
```bash
# Open the output file and verify:
# 1. Text is readable (not gibberish)
# 2. No repeated headers/footers
# 3. Tables/diagrams extracted properly
```

---

#### Step 1.2: Extract Content from Slides
```bash
python src/data_processing/extract_slides.py
```

**What it does:**
- Reads PPTX files from `data/raw/slides/`
- Extracts text + speaker notes
- Saves to `data/processed/slides/all_slides_combined.json`

**⚠️ Check Quality:**
- Slides often have minimal text
- Verify speaker notes are included
- Check for bullet points formatting

---

### **PHASE 2: Create Training Dataset** 📝

#### Step 2.1: Generate Q&A Pairs
```bash
python src/data_processing/create_dataset.py
```

**What it does:**
1. Reads extracted content from Phase 1
2. For each chunk/slide:
   - Generates 3-5 questions using templates
   - Creates paraphrased versions
   - Adds reasoning chains
3. Applies augmentation (2x factor)
4. Splits 90% train / 10% validation
5. Saves to `data/processed/train.jsonl` and `val.jsonl`

**Critical: Data Format**
```json
{
  "instruction": "What is virtual memory?",
  "input": "",
  "output": "Virtual memory is a memory management technique...",
  "text": "### Instruction:\nWhat is virtual memory?\n\n### Response:\nVirtual memory is..."
}
```

**The `text` field is what the model actually sees!**

---

#### Step 2.2: **INSPECT YOUR TRAINING DATA** 🔍

**This is the #1 step people skip that causes hallucinations!**

```powershell
# Open train.jsonl in a text editor
code data/processed/train.jsonl

# Or check first 10 examples:
Get-Content data/processed/train.jsonl -First 10 | ConvertFrom-Json | Format-List
```

**What to check:**

✅ **Good Signs:**
- Answers are 2-5 sentences minimum (not 1-word answers)
- Answers contain specific details from your PDFs
- Questions are diverse (not all "What is...")
- No generic answers like "It's a concept in OS"

❌ **Bad Signs:**
- Answers are too short (< 20 words)
- Generic questions: "Explain the concept"
- Answers don't match questions
- Repeated/duplicate content

**If data is bad → Training will fail → Hallucinations guaranteed**

---

### **PHASE 3: Training** 🚀

#### Step 3.1: Check Configuration
```bash
# View current config
Get-Content configs/training_config.yaml
```

**Critical settings for anti-hallucination:**

```yaml
# LoRA: Higher rank = More capacity
lora:
  r: 32                    # ✅ Good for large dataset
  lora_alpha: 64           # Keep 2*r
  target_modules:          # ✅ Include all projection layers
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj
    - embed_tokens

# Training: Lower LR = Less overfitting
training:
  learning_rate: 5.0e-5    # ✅ Lower LR prevents memorization
  num_train_epochs: 3      # ✅ 3 epochs for large dataset
  warmup_ratio: 0.1        # ✅ Gradual learning
  max_seq_length: 1024     # ✅ Enough context
```

---

#### Step 3.2: Start Training
```bash
python src/training/fine_tune.py
```

**What happens:**
1. Loads Qwen2.5-7B base model (4-bit quantized)
2. Adds LoRA adapters
3. Loads train.jsonl and val.jsonl
4. Trains for 3 epochs
5. Saves checkpoints to `models/fine_tuned/`

**Monitor training:**
```powershell
# Check GPU usage
nvidia-smi -l 1

# View logs in real-time
Get-Content models/fine_tuned/logs/events.* -Wait

# Or use TensorBoard:
tensorboard --logdir models/fine_tuned/logs
```

**Training metrics to watch:**

✅ **Good Training:**
- `train_loss` decreases smoothly: 2.5 → 1.8 → 1.2
- `eval_loss` decreases similarly: 2.3 → 1.7 → 1.3
- `eval_loss` stays close to `train_loss` (gap < 0.3)

❌ **Bad Training (Overfitting):**
- `train_loss` drops very fast: 2.5 → 0.5
- `eval_loss` increases or stays high: 2.3 → 2.5
- Large gap (> 0.5) = Model memorizing, not learning

**If overfitting detected:**
1. Reduce epochs: 3 → 2
2. Increase dropout: 0.05 → 0.1
3. Lower learning rate: 5e-5 → 2e-5
4. Add more training data

---

### **PHASE 4: Build Vector Database (RAG)** 🗄️

#### Step 4.1: Create Vector DB
```bash
python src/data_processing/build_vectordb.py
```

**What it does:**
- Chunks all PDF/slide content semantically
- Embeds chunks with `sentence-transformers/all-mpnet-base-v2`
- Stores in ChromaDB at `data/vectordb/course_materials/`

**Critical: Chunking Strategy**
- **Chunk size**: 512 tokens (with 50 overlap)
- **Why**: Balance between context and specificity
- Too large chunks (2048) → Retrieves irrelevant info
- Too small chunks (128) → Missing context

---

### **PHASE 5: Testing & Inference** 🧪

#### Step 5.1: Interactive Query
```bash
python src/inference/query_processor.py
```

**Workflow:**
1. You enter question: "What is paging?"
2. System retrieves top-5 relevant chunks from vector DB
3. Sends to fine-tuned model with prompt:
   ```
   Context: <retrieved chunks>
   Question: What is paging?
   Answer:
   ```
4. Model generates answer based on context

**Anti-hallucination mechanism:**
- If no relevant chunks found → Model says "I don't have information"
- Model is trained to only use provided context

---

#### Step 5.2: Evaluate on Test Set
```bash
python src/evaluation/evaluate_model.py
```

**Metrics:**
- **BLEU**: Measures word overlap (0-100, higher = better)
- **ROUGE-L**: Measures longest common subsequence
- **BERTScore**: Semantic similarity (0-1)
- **Faithfulness**: Are answers grounded in context?

**Target scores for good model:**
- BLEU > 40
- ROUGE-L > 0.6
- BERTScore > 0.85
- Faithfulness > 90%

---

## 🛠️ Fixing Hallucinations (After Training)

### Diagnosis: Test Your Model

**Step 1: Ask a question NOT in your training data**
```python
# Question about a topic you know is NOT in PDFs
question = "What is quantum computing in operating systems?"

# If model gives a detailed answer → HALLUCINATING
# If model says "Not covered in course" → GOOD
```

**Step 2: Check training data leakage**
```bash
# Search if question was in training set
Select-String -Path data/processed/train.jsonl -Pattern "quantum computing"
```

---

### Fix 1: Improve Training Data Quality

**Before training again, fix create_dataset.py:**

```python
# Add minimum length check
def generate_qa_pairs(content: str, source: str) -> List[Dict]:
    qa_pairs = []
    
    # Skip short content
    if len(content) < 100:  # ⚠️ CRITICAL: Minimum 100 chars
        return []
    
    # Ensure answers are detailed
    if len(content) < 200:
        # Add more context
        content = f"{content}\n\nThis is explained in {source}."
    
    # Create question with specific topic
    topics = extract_topics(content)
    for topic in topics[:3]:  # Limit to top 3 topics
        question = f"Explain {topic} as covered in the course."
        
        qa_pairs.append({
            "instruction": question,
            "output": content,  # Full context as answer
            "text": f"### Instruction:\n{question}\n\n### Response:\n{content}"
        })
    
    return qa_pairs
```

**Key changes:**
1. ✅ Minimum content length = 100 chars
2. ✅ Questions reference "course" → trains model to stay grounded
3. ✅ Answers include full context, not just topic name

---

### Fix 2: Add "I Don't Know" Examples

**Create calibration dataset:**

```python
# Add to create_dataset.py
def add_negative_examples(dataset: List[Dict]) -> List[Dict]:
    """Add examples where model should refuse to answer"""
    
    negative_prompts = [
        "What is the meaning of life?",
        "Explain quantum mechanics in OS",
        "Who won the World Cup?",
        "Write a poem about processes"
    ]
    
    refusal_response = (
        "This question is outside the scope of the course materials I was trained on. "
        "I can only answer questions about Operating Systems and Computer Networks "
        "based on the provided lecture materials."
    )
    
    for prompt in negative_prompts:
        dataset.append({
            "instruction": prompt,
            "output": refusal_response,
            "text": f"### Instruction:\n{prompt}\n\n### Response:\n{refusal_response}"
        })
    
    return dataset
```

**Then regenerate dataset:**
```bash
python src/data_processing/create_dataset.py
```

---

### Fix 3: Adjust Training Config

**For less hallucination, modify `configs/training_config.yaml`:**

```yaml
training:
  learning_rate: 2.0e-5        # Lower LR = Less creative = Less hallucination
  num_train_epochs: 2          # Fewer epochs = Less overfitting
  warmup_ratio: 0.15           # Longer warmup = Smoother learning
  
lora:
  r: 64                        # Higher rank IF you have >2000 examples
  lora_dropout: 0.1            # More dropout = Less memorization
```

**Re-train with new config:**
```bash
python src/training/fine_tune.py
```

---

### Fix 4: Improve RAG Retrieval

**Issue**: If retrieval fails, model falls back to making things up.

**Solution: Verify retrieval works**

```bash
python optimize_memory.py
```

Look for:
```
✓ Hybrid retriever initialized with 4000 documents
```

If documents = 0 → Vector DB is empty!

**Rebuild vector DB:**
```bash
python src/data_processing/build_vectordb.py
```

---

### Fix 5: Add System Prompt (Inference)

**Modify query_processor.py to constrain model:**

```python
def generate_answer(self, question: str, context: str) -> str:
    system_prompt = (
        "You are a teaching assistant for Operating Systems and Computer Networks. "
        "Answer ONLY based on the provided context from course materials. "
        "If the context doesn't contain the answer, say 'This is not covered in the course materials.' "
        "Be specific and use examples from the context. "
        "Do NOT make up information."
    )
    
    full_prompt = f"{system_prompt}\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    
    # Generate
    response = self.model.generate(full_prompt)
    return response
```

**This forces model to stay grounded!**

---

## 🎯 Bulls-Eye Training Recipe

### For Your 1,637 Pages Dataset

**Step-by-step:**

1. **Extract all PDFs** ✅
   ```bash
   python src/data_processing/extract_pdfs.py
   ```
   - Check output quality
   - Verify ~1,637 pages extracted

2. **Generate high-quality Q&A** ✅
   ```bash
   # Edit create_dataset.py to add:
   # - Minimum content length (100 chars)
   # - Specific question templates
   # - "I don't know" examples
   
   python src/data_processing/create_dataset.py
   ```
   - Target: 3,000-4,000 Q&A pairs
   - Inspect first 20 manually

3. **Train with anti-hallucination config** ✅
   ```yaml
   # configs/training_config.yaml
   lora:
     r: 32
     lora_dropout: 0.1
   
   training:
     learning_rate: 5.0e-5
     num_train_epochs: 3
     max_seq_length: 1024
     warmup_ratio: 0.1
   ```
   
   ```bash
   python src/training/fine_tune.py
   ```
   - Monitor eval_loss
   - Stop if overfitting detected

4. **Build RAG vector DB** ✅
   ```bash
   python src/data_processing/build_vectordb.py
   ```
   - Verify document count matches training samples

5. **Test systematically** ✅
   ```bash
   python src/evaluation/evaluate_model.py
   ```
   - Check faithfulness score
   - Manually test 10 questions

6. **Iterate** 🔄
   - If hallucinating → Lower LR, add "I don't know" examples
   - If too conservative → Increase LoRA rank
   - If retrieval fails → Improve chunking

---

## 📊 Debugging Checklist

**Before re-training, verify:**

- [ ] PDFs extracted correctly (readable text in JSON)
- [ ] Training data has >1,000 examples
- [ ] Each example has >100 chars answer
- [ ] Questions are diverse (not all "What is...")
- [ ] Answers come from your PDFs (not generic)
- [ ] Added "I don't know" examples for out-of-scope questions
- [ ] Config uses learning_rate < 1e-4
- [ ] Config uses num_epochs = 2-3 (not 5+)
- [ ] LoRA r = 32 or higher
- [ ] Vector DB has same number of docs as training examples

**During training, monitor:**

- [ ] train_loss decreases smoothly
- [ ] eval_loss decreases and stays close to train_loss
- [ ] GPU memory stays < 11GB (for RTX 3060)
- [ ] No OOM errors

**After training, test:**

- [ ] Ask question from training set → Should answer correctly
- [ ] Ask question NOT in training set but in PDFs → Should answer with "not covered"
- [ ] Ask completely unrelated question → Should refuse
- [ ] Compare answer to PDF source → Should match facts

---

## 🚨 Common Mistakes

1. **Training on generic questions**: "Explain the concept" → Model learns to be vague
2. **Short answers**: "It's a scheduling algorithm" → Model learns minimal responses
3. **Too many epochs**: 10 epochs → Overfitting guaranteed
4. **High learning rate**: 1e-3 → Model forgets base knowledge
5. **Low LoRA rank**: r=8 → Not enough capacity for complex domain
6. **Ignoring eval_loss**: Only watching train_loss → Overfitting undetected
7. **No retrieval testing**: Assuming RAG works → Silent failures

---

## 📈 Expected Timeline (RTX 3060 12GB)

| Phase | Time | Critical Check |
|-------|------|----------------|
| Extract PDFs | 30 min | Text quality |
| Create Dataset | 15 min | Q&A quality |
| Training (3 epochs) | 6-8 hours | Eval loss |
| Build Vector DB | 20 min | Doc count |
| Testing | 30 min | Faithfulness |
| **Total** | **~10 hours** | |

---

## 🎓 Success Criteria

Your model is **bulls-eye** when:

✅ Answers questions accurately based on PDFs
✅ Says "not covered" for out-of-scope questions
✅ Provides specific details (not generic)
✅ BERTScore > 0.85
✅ Faithfulness > 90%
✅ You manually verify 20 questions and all are correct

---

## 🔧 Quick Fixes Reference

| Problem | Fix |
|---------|-----|
| Hallucinating facts | Add "I don't know" examples + lower LR |
| Generic answers | Improve training data quality |
| Wrong answers | Improve RAG retrieval (better chunking) |
| OOM errors | Reduce max_seq_length to 512 |
| Slow training | Reduce batch size, increase grad accumulation |
| Overfitting | Reduce epochs, increase dropout |

---

**Next Step**: Run diagnostic on your current model and training data!

```bash
# Check training data quality
python -c "import json; data=[json.loads(l) for l in open('data/processed/train.jsonl')]; print(f'Samples: {len(data)}'); print(f'Avg answer length: {sum(len(d[\"output\"]) for d in data)/len(data):.0f} chars'); print('Sample:', data[0])"
```

This will show if your data is the problem! 🎯
