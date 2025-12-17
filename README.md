# 🎓 Qwen2.5 Fine-tuning for OS & Networks RAG System

Train a custom AI teaching assistant that answers questions about Operating Systems and Computer Networks using **your own course materials** (PDFs, slides). Includes anti-hallucination measures and RAG for accurate, grounded answers.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📖 Table of Contents

- [Prerequisites](#-prerequisites)
- [Installation](#-installation-from-scratch)
- [Step-by-Step Workflow](#-step-by-step-workflow)
- [Advanced Options](#-advanced-options)
- [Troubleshooting](#-troubleshooting)
- [Documentation](#-documentation)

---

## 🎯 What You'll Build

A fine-tuned AI model that:
- ✅ Answers questions based on **your course PDFs and slides**
- ✅ Uses RAG (Retrieval-Augmented Generation) to stay grounded
- ✅ Refuses out-of-scope questions (no hallucinations)
- ✅ Generates step-by-step explanations with examples
- ✅ Runs locally on your GPU or in the cloud

**Example:**
```
You: "What is the difference between preemptive and non-preemptive scheduling?"

AI: "Let me explain step by step:

1. Preemptive Scheduling:
   - OS can interrupt a running process
   - Example: Round Robin, Priority with preemption
   - Better response time, higher overhead

2. Non-Preemptive Scheduling:
   - Process runs until completion or blocks
   - Example: FCFS, SJF
   - Lower overhead, may cause starvation

Key Trade-off: Preemptive provides better responsiveness but requires context switching.

(Source: OS_Textbook.pdf - Chapter 5)"
```

---

## 📋 Prerequisites

### **Hardware Requirements**

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **GPU** | None (CPU only) | RTX 3060 12GB+ |
| **RAM** | 16GB | 32GB |
| **Storage** | 20GB free | 50GB free |
| **OS** | Windows 10/11 | Windows 11 |

**Note:** Training on CPU is ~10x slower. GPU strongly recommended.

### **Software Requirements**

- **Python 3.8 to 3.11** (3.12+ not supported by some packages)
- **Git** (to clone repository)
- **Visual Studio Build Tools** (for some Python packages on Windows)

**Optional but Recommended:**
- **CUDA 11.8+** (for GPU acceleration)
- **Tesseract OCR** (for diagram text extraction)
- **Ollama** (for LLM-enhanced dataset generation)

---

## 🚀 Installation (From Scratch)

### **Step 1: Clone Repository**

```bash
# Clone the project
git clone https://github.com/your-username/qwen-finetuning-rag.git
cd qwen-finetuning-rag
```

### **Step 2: Create Virtual Environment**

```bash
# Windows PowerShell
python -m venv venv
.\venv\Scripts\Activate.ps1

# Windows CMD
python -m venv venv
venv\Scripts\activate.bat

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

**Verify Python version:**
```bash
python --version
# Should show: Python 3.8.x to 3.11.x
```

### **Step 3: Install Dependencies**

```bash
# Upgrade pip first
python -m pip install --upgrade pip

# Install PyTorch with CUDA (for GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Or CPU-only version
pip install torch torchvision torchaudio

# Install remaining packages
pip install -r requirements.txt
```

**This will take 5-10 minutes** (downloads ~5GB of packages)

### **Step 4: Verify Installation**

```bash
# Test imports
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
```

**Expected output:**
```
PyTorch: 2.1.0+cu118
Transformers: 4.36.0
CUDA Available: True
```

### **Step 5: Optional - Install Ollama (for LLM-enhanced datasets)**

**Why?** Generates much better training data (85-95/100 quality vs 55-70/100)

```bash
# Windows (PowerShell as Admin)
winget install Ollama.Ollama

# Or download from: https://ollama.ai

# Start Ollama (in a separate terminal)
ollama serve

# Pull a model (in another terminal)
ollama pull llama3.2:3b
# This downloads ~2GB
```

### **Step 6: Optional - Install Tesseract OCR (for diagrams)**

```bash
# Windows (PowerShell as Admin)
choco install tesseract

# Or download from: https://github.com/UB-Mannheim/tesseract/wiki

# Verify
tesseract --version
```

---

## 📚 Step-by-Step Workflow

Follow these steps **in order** for best results:

---

### **STEP 1: Prepare Your Training Materials** (5 minutes)

Add your course materials to the `data/raw/` directory:

```bash
# Create directories (already exist, but verify)
mkdir -p data/raw/pdfs
mkdir -p data/raw/slides
```

**Copy your files:**
```bash
# Windows
copy "C:\path\to\your\textbooks\*.pdf" data\raw\pdfs\
copy "C:\path\to\your\slides\*.pptx" data\raw\slides\

# Linux/macOS
cp ~/Documents/textbooks/*.pdf data/raw/pdfs/
cp ~/Documents/slides/*.pptx data/raw/slides/
```

**Expected structure:**
```
data/raw/
├── pdfs/
│   ├── Operating_Systems_Textbook.pdf
│   ├── Networks_Guide.pdf
│   └── Lecture_Notes.pdf
└── slides/
    ├── Week1_Introduction.pptx
    ├── Week2_Processes.pptx
    └── Week3_Memory.pptx
```

**Tips:**
- ✅ Use descriptive filenames
- ✅ Include textbooks, lecture notes, reference materials
- ✅ Both PDFs and PPTX slides are supported
- ⚠️ Avoid scanned PDFs with no text layer (OCR quality varies)

---

### **STEP 2: Extract Content from Materials** (30-45 minutes)

Extract text from PDFs and slides:

```bash
# Extract PDFs (takes ~1 min per 100 pages)
python src/data_processing/extract_pdfs.py

# Extract slides (takes ~30 sec per 50 slides)
python src/data_processing/extract_slides.py
```

**What happens:**
- Reads each PDF page by page
- Extracts text from slides + speaker notes
- Removes headers, footers, page numbers
- Saves to `data/processed/books/` and `data/processed/slides/`

**Verify extraction worked:**
```bash
# Check output files exist
ls data/processed/books/all_pdfs_combined.json
ls data/processed/slides/all_slides_combined.json

# Peek at content (Windows)
Get-Content data\processed\books\all_pdfs_combined.json -First 50

# Peek at content (Linux/macOS)
head -n 50 data/processed/books/all_pdfs_combined.json
```

**Should see:** Readable text from your PDFs (not gibberish or empty)

---

### **STEP 3: Create Training Dataset** (15-60 minutes)

Generate Q&A pairs from extracted content.

#### **Option A: Rule-Based (Fast, 15 min) - Quick Testing**

```bash
python src/data_processing/create_dataset.py
```

**Pros:**
- ✅ Fast (~15 min for 1000 chunks)
- ✅ No external dependencies
- ✅ Good for quick prototyping

**Cons:**
- ⚠️ Lower quality (score: 55-70/100)
- ⚠️ Generic questions
- ⚠️ No refusal examples → Higher hallucination risk

---

#### **Option B: LLM-Enhanced (Slower, 60 min) - Production Quality** ⭐ **RECOMMENDED**

**Setup Ollama first (if not done):**
```bash
# Install Ollama
winget install Ollama.Ollama

# Start Ollama (in separate terminal)
ollama serve

# Pull model (in another terminal)
ollama pull llama3.2:3b
```

**Generate dataset:**
```bash
# Test with 10 chunks first (2 min)
python src/data_processing/create_dataset_llm.py --test

# Check quality
python diagnose.py
# Should show: Quality Score: 85+/100

# If quality looks good, generate full dataset (60 min)
python src/data_processing/create_dataset_llm.py --provider ollama --model llama3.2:3b
```

**Pros:**
- ✅ High quality (score: 85-95/100)
- ✅ Diverse, specific questions
- ✅ Structured answers with examples
- ✅ Automatic refusal examples → Low hallucination

**Cons:**
- ⚠️ Slower (~60 min for 1000 chunks)
- ⚠️ Requires Ollama or OpenAI API

---

### **STEP 4: Check Data Quality** ⚠️ **CRITICAL CHECKPOINT**

**This step determines if your model will hallucinate!**

```bash
python diagnose.py
```

**Understanding the output:**

```
Quality Score: 72/100
  ✅ All answers >= 50 chars
  ⚠️  35% questions generic (20% threshold)
  ✅ No duplicates
  ✅ Good diversity
  ⚠️  NO REFUSAL EXAMPLES → Will hallucinate!

Average answer length: 156 chars
Question diversity: Top pattern is 28%
Refusal examples: 0
```

**What to do:**

| Score | Action |
|-------|--------|
| **85-100** | ✅ Excellent! Proceed to training |
| **70-84** | ⚠️ Good, but consider LLM enhancement |
| **50-69** | ❌ Fix data quality first (see below) |
| **< 50** | ❌ Must regenerate with LLM or fix create_dataset.py |

**If score < 70:**
- Use LLM-enhanced generation (Option B above)
- Or manually edit `src/data_processing/create_dataset.py` to improve quality
- See: [docs/COMPLETE_WORKFLOW_AND_FIXES.md](docs/COMPLETE_WORKFLOW_AND_FIXES.md)

---

### **STEP 5: Train the Model** (6-8 hours on RTX 3060)

**Before training, check your config:**
```bash
# View current training configuration
cat configs/training_config.yaml

# Key settings (already optimized for RTX 3060 12GB):
# - batch_size: 1
# - gradient_accumulation: 16
# - max_seq_length: 1024
# - learning_rate: 5e-5
# - epochs: 3
```

**Start training:**
```bash
python src/training/fine_tune.py
```

**Monitor training (in another terminal):**
```bash
# GPU usage
nvidia-smi -l 1

# Training logs (after training starts)
tensorboard --logdir models/fine_tuned/logs
# Open browser: http://localhost:6006
```

**What to watch:**

✅ **Good training:**
```
Epoch 1: train_loss=2.45, eval_loss=2.38
Epoch 2: train_loss=1.82, eval_loss=1.78
Epoch 3: train_loss=1.25, eval_loss=1.31
```
Both losses decrease, gap < 0.3 → Model learning properly

❌ **Bad training (overfitting):**
```
Epoch 1: train_loss=2.45, eval_loss=2.38
Epoch 2: train_loss=1.12, eval_loss=2.51
Epoch 3: train_loss=0.48, eval_loss=2.89
```
Train loss drops fast, eval loss increases → Model memorizing → **Will hallucinate!**

**If overfitting detected:**
- Stop training (Ctrl+C)
- Edit `configs/training_config.yaml`:
  - Reduce `num_train_epochs: 3` → `2`
  - Increase `lora_dropout: 0.05` → `0.1`
  - Lower `learning_rate: 5e-5` → `2e-5`
- Restart training

**Training output:**
- `models/fine_tuned/adapter_model.bin` (~500MB) - Your trained model
- `models/fine_tuned/adapter_config.json` - LoRA configuration
- `models/fine_tuned/logs/` - TensorBoard logs

**Time estimates:**

| GPU | Time |
|-----|------|
| RTX 4090 | 2-3 hours |
| RTX 3060 | 6-8 hours |
| RTX 2060 | 10-12 hours |
| CPU | 3-5 days ⚠️ |

---

### **STEP 6: Build Vector Database for RAG** (20 minutes)

Create searchable database of your course materials:

```bash
python src/data_processing/build_vectordb.py
```

**What happens:**
- Re-reads all PDF/slide content
- Chunks into 512-token segments (overlap=50)
- Embeds chunks using sentence-transformers
- Stores in ChromaDB at `data/vectordb/course_materials/`

**Verify:**
```bash
python diagnose.py
# Should show: "✅ Vector database found: 3000+ documents"
```

**Why this matters:**
- RAG retrieves relevant context for each question
- Model generates answers based on retrieved context
- **Prevents hallucination** by grounding answers in your PDFs

---

### **STEP 7: Test Your Model** (5 minutes)

**Interactive testing:**
```bash
python src/inference/query_processor.py
```

**Example session:**
```
🤖 Model loaded successfully!
📚 Vector DB: 3247 documents

Enter question (or 'quit'): What is process scheduling?

🔍 Retrieved 5 relevant chunks from vector database...

Answer:
Process scheduling is the mechanism by which the OS decides which process runs 
on the CPU and for how long. Common algorithms include:

1. First-Come First-Served (FCFS): Executes processes in arrival order
2. Shortest Job First (SJF): Prioritizes processes with shortest execution time
3. Round Robin (RR): Allocates fixed time quantum to each process
4. Priority Scheduling: Runs highest priority process first

Trade-off: FCFS is simple but may cause convoy effect. Round Robin provides
better response time but has context-switching overhead.

(Source: OS_Textbook.pdf - Chapter 5, pages 45-47)

---

Enter question (or 'quit'): What is blockchain?

Answer:
This topic is outside the scope of the Operating Systems and Computer Networks 
course materials. I can only answer questions based on the provided lecture content.

---
```

**Test cases to try:**

1. **Known topic from PDFs:**
   - "Explain virtual memory"
   - "What causes deadlock?"
   - "Compare TCP and UDP"

2. **Out-of-scope question:**
   - "What is machine learning?"
   - "Explain blockchain"
   - "Who is the president?"

3. **Edge cases:**
   - Very long question
   - Ambiguous question
   - Question with typos

**Success criteria:**
- ✅ Answers questions from PDFs accurately
- ✅ Refuses out-of-scope questions
- ✅ Cites sources (PDF names, page numbers)
- ✅ No hallucinated facts

---

### **STEP 8: Evaluate Model Performance** (10 minutes)

Run automated evaluation:

```bash
python src/evaluation/evaluate_model.py
```

**Output:**
```
============================================================
                   Evaluation Results
============================================================

Overall Metrics:
  BLEU Score:        42.3
  ROUGE-L:           0.68
  BERTScore (F1):    0.87
  Faithfulness:      93.2%

Question Type Performance:
  Conceptual:        BLEU=45.2, ROUGE=0.71, Faithfulness=95%
  Procedural:        BLEU=39.8, ROUGE=0.65, Faithfulness=91%
  Comparison:        BLEU=43.1, ROUGE=0.69, Faithfulness=94%

✅ Model is performing well!
```

**Target scores:**

| Metric | Target | Meaning |
|--------|--------|---------|
| **BLEU** | > 40 | Word-level overlap with reference |
| **ROUGE-L** | > 0.6 | Longest common subsequence |
| **BERTScore** | > 0.85 | Semantic similarity |
| **Faithfulness** | > 90% | Answers grounded in context |

**Most important: Faithfulness**
- If < 80% → Model is hallucinating
- Check training data quality: `python diagnose.py`
- Consider retraining with LLM-enhanced data

---

## 🎓 You're Done! 🎉

Your model is now trained and ready to use!

**Next steps:**

1. **Deploy for your use:**
   ```bash
   # Start interactive assistant
   python src/inference/query_processor.py
   ```

2. **Integrate into applications:**
   ```python
   from src.inference.query_processor import QueryProcessor
   
   processor = QueryProcessor()
   answer = processor.process("What is paging?")
   print(answer)
   ```

3. **Share with classmates:**
   - Share the trained model (`models/fine_tuned/`)
   - Or share training instructions (this README!)

---

## 🔧 Advanced Options

### **Using OpenAI Instead of Ollama**

```bash
# Set API key
export OPENAI_API_KEY="sk-your-key-here"

# Generate dataset with GPT-4
python src/data_processing/create_dataset_llm.py \
    --provider openai \
    --model gpt-4-turbo-preview

# Costs: ~$2-5 for 1000 chunks (high quality)
```

### **Using HuggingFace Models**

```bash
# Use local HuggingFace model (free, GPU required)
python src/data_processing/create_dataset_llm.py \
    --provider huggingface \
    --model mistralai/Mistral-7B-Instruct-v0.2
```

### **Extracting Images and Diagrams**

```bash
# Extract images from PDFs
python src/data_processing/extract_images.py

# Process with OCR (requires Tesseract)
python src/data_processing/ocr_processor.py

# Use vision-language model for diagram understanding
python src/inference/vision_language.py
```

See: [docs/MULTIMEDIA_COMPLETE.md](docs/MULTIMEDIA_COMPLETE.md)

### **Enrichment Features**

```bash
# Auto-suggest YouTube videos for topics
python src/enrichment/youtube_suggester.py

# Find research papers (uses Semantic Scholar API)
python src/enrichment/paper_search.py

# Generate concept maps
python src/enrichment/concept_mapper.py
```

### **Memory Optimization**

If you have less than 12GB VRAM, edit `configs/training_config.yaml`:

```yaml
# For 8GB VRAM:
max_seq_length: 512          # down from 1024
gradient_accumulation_steps: 32  # up from 16

# For 6GB VRAM:
max_seq_length: 384
gradient_accumulation_steps: 64
load_in_4bit: true           # enable 4-bit quantization
```

### **Advanced Training Techniques**

```python
# Use advanced techniques (gradient checkpointing, Flash Attention 2)
python src/training/fine_tune.py --use-advanced-techniques

# Resume training from checkpoint
python src/training/fine_tune.py --resume-from models/fine_tuned/checkpoint-500
```

---

## 🐛 Troubleshooting Guide

### **Installation Issues**

| Problem | Solution |
|---------|----------|
| **`torch.cuda.is_available() = False`** | Reinstall PyTorch with CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu118` |
| **`ModuleNotFoundError: transformers`** | Install dependencies: `pip install -r requirements.txt` |
| **Tesseract not found** | Install Tesseract and add to PATH: [Download](https://github.com/UB-Mannheim/tesseract/wiki) |
| **Ollama connection refused** | Start Ollama: `ollama serve` (in separate terminal) |

### **Data Quality Issues**

| Problem | Solution |
|---------|----------|
| **Quality score < 70** | Use LLM-enhanced dataset: `python src/data_processing/create_dataset_llm.py` |
| **No refusal examples** | LLM dataset auto-generates these. Rule-based does not. |
| **Duplicate questions** | Increase chunk diversity in `create_dataset.py` |
| **Generic questions** | Use `--model llama3.2:3b` or larger for better question generation |

### **Training Issues**

| Problem | Solution |
|---------|----------|
| **`CUDA out of memory`** | Reduce `max_seq_length: 512` or `batch_size: 1` with higher `gradient_accumulation` |
| **Training loss not decreasing** | Check learning rate (try `2e-5` to `5e-5`), verify data quality score > 70 |
| **Eval loss increasing (overfitting)** | Reduce epochs to 2, increase `lora_dropout: 0.1`, lower learning rate |
| **Training too slow** | Reduce `max_seq_length: 768`, ensure CUDA is enabled, use `load_in_4bit: true` |

### **Inference Issues**

| Problem | Solution |
|---------|----------|
| **Model hallucinating** | Check faithfulness score < 90% → Retrain with better data |
| **Slow inference** | Enable caching: edit `configs/model_config.yaml` → `cache_enabled: true` |
| **Vector DB not found** | Build vector database: `python src/data_processing/build_vectordb.py` |
| **Out-of-scope questions not refused** | Verify refusal examples exist: `python diagnose.py` |

### **Performance Issues**

| Problem | Solution |
|---------|----------|
| **Low BLEU/ROUGE scores** | Retrain with higher quality dataset (LLM-enhanced) |
| **Poor retrieval relevance** | Increase chunk overlap in `build_vectordb.py`, use hybrid retrieval |
| **Answers too short** | Check average answer length in diagnose.py (should be > 100 chars) |
| **Answers too long/verbose** | Adjust `max_new_tokens` in `configs/model_config.yaml` |

---

## 📂 Project Structure

```
p1/
│
├── configs/                      # Configuration files
│   ├── api_config.yaml           # API keys (Ollama, OpenAI, etc.)
│   ├── model_config.yaml         # Model inference settings
│   └── training_config.yaml      # Training hyperparameters
│
├── data/                         # All data files
│   ├── raw/                      # Your input materials
│   │   ├── pdfs/                 # PDF textbooks
│   │   └── slides/               # PowerPoint slides
│   ├── processed/                # Extracted text
│   │   ├── books/                # PDF content (JSON)
│   │   └── slides/               # Slide content (JSON)
│   ├── training/                 # Training datasets
│   │   └── qa_dataset.json       # Generated Q&A pairs
│   ├── vectordb/                 # Vector database (ChromaDB)
│   └── evaluation/               # Test questions
│       └── endsem_questions.json # Evaluation dataset
│
├── src/                          # Source code
│   ├── data_processing/          # Data extraction & generation
│   │   ├── extract_pdfs.py       # Extract text from PDFs
│   │   ├── extract_slides.py     # Extract text from PPTX
│   │   ├── create_dataset.py     # Rule-based Q&A generation
│   │   ├── create_dataset_llm.py # LLM-enhanced Q&A generation
│   │   ├── llm_augmentation.py   # LLM integration (Ollama/OpenAI)
│   │   ├── build_vectordb.py     # Build ChromaDB vector database
│   │   ├── extract_images.py     # Extract images from PDFs
│   │   └── ocr_processor.py      # OCR for diagrams
│   │
│   ├── training/                 # Model training
│   │   ├── fine_tune.py          # Main training script (LoRA)
│   │   └── advanced_techniques.py # Advanced training methods
│   │
│   ├── inference/                # Model inference & RAG
│   │   ├── query_processor.py    # Main query interface
│   │   ├── rag_system.py         # RAG implementation
│   │   ├── hybrid_retriever.py   # Dense + BM25 retrieval
│   │   ├── model_loader.py       # Model loading utilities
│   │   ├── cached_processor.py   # Query caching (Redis)
│   │   └── vision_language.py    # Vision-language models
│   │
│   ├── evaluation/               # Model evaluation
│   │   ├── evaluate_model.py     # Run evaluation metrics
│   │   └── advanced_metrics.py   # BLEU, ROUGE, BERTScore
│   │
│   ├── enrichment/               # Extra features
│   │   ├── youtube_suggester.py  # Suggest YouTube videos
│   │   ├── paper_search.py       # Find research papers
│   │   └── concept_mapper.py     # Generate concept maps
│   │
│   └── utils/                    # Utilities
│       ├── config.py             # Config loader
│       ├── helpers.py            # Helper functions
│       └── text_preprocessing.py # Text cleaning
│
├── models/                       # Trained models
│   └── fine_tuned/               # Your trained model (created after training)
│       ├── adapter_model.bin     # LoRA weights (~500MB)
│       ├── adapter_config.json   # LoRA configuration
│       └── logs/                 # TensorBoard logs
│
├── docs/                         # Documentation
│   ├── QUICKSTART.md             # 10-minute quick start
│   ├── COMPLETE_WORKFLOW_AND_FIXES.md  # Anti-hallucination guide
│   ├── LLM_AUGMENTATION_GUIDE.md       # LLM dataset generation
│   ├── PROJECT_EXPLANATION.md          # System architecture
│   ├── WORKFLOW_VISUAL.md              # Visual flowcharts
│   └── MULTIMEDIA_COMPLETE.md          # Image/diagram guide
│
├── diagnose.py                   # Data quality checker ⚠️ IMPORTANT
├── test_improvements.py          # Test suite
├── optimize_memory.py            # Memory optimization tool
│
├── *.bat                         # Windows batch scripts
│   ├── run_all.bat               # Complete pipeline
│   ├── process_data.bat          # Extract + dataset
│   ├── train.bat                 # Training only
│   └── evaluate.bat              # Evaluation only
│
├── requirements.txt              # Python dependencies
├── setup.py                      # Package setup
└── README.md                     # This file
```

---

## 📚 Documentation Guide

| Guide | When to Read | Description |
|-------|--------------|-------------|
| **[README.md](README.md)** | 🟢 **Start here** | Complete step-by-step workflow (this file) |
| **[QUICKSTART.md](QUICKSTART.md)** | 🟢 After README | Ultra-condensed commands (10 min) |
| **[diagnose.py](diagnose.py)** | 🟡 Before training | Check data quality (prevents hallucinations) |
| **[COMPLETE_WORKFLOW_AND_FIXES.md](docs/COMPLETE_WORKFLOW_AND_FIXES.md)** | 🔴 If model hallucinates | Deep-dive on fixing hallucination issues |
| **[LLM_AUGMENTATION_GUIDE.md](docs/LLM_AUGMENTATION_GUIDE.md)** | 🟡 For high quality | How to use LLM-enhanced dataset generation |
| **[PROJECT_EXPLANATION.md](docs/PROJECT_EXPLANATION.md)** | 🔵 For understanding | System architecture and technical details |
| **[WORKFLOW_VISUAL.md](docs/WORKFLOW_VISUAL.md)** | 🔵 Visual learners | Flowcharts and diagrams of pipeline |
| **[MULTIMEDIA_COMPLETE.md](docs/MULTIMEDIA_COMPLETE.md)** | 🟡 For diagrams | Extract and process images from PDFs |
| **[IMPROVEMENTS_GUIDE.md](docs/IMPROVEMENTS_GUIDE.md)** | 🔵 Advanced users | Summary of 11 major enhancements |

**Reading order:**
1. README.md (this file) → Set up and train your first model
2. diagnose.py → Check if your data is good
3. QUICKSTART.md → Quick reference for commands
4. COMPLETE_WORKFLOW_AND_FIXES.md → If you encounter issues

---

## 🚀 Quick Command Reference

```bash
# 1. Setup (one-time)
git clone <your-repo-url>
cd p1
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt

# 2. Add your materials
copy "C:\path\to\pdfs\*.pdf" data\raw\pdfs\
copy "C:\path\to\slides\*.pptx" data\raw\slides\

# 3. Extract content
python src/data_processing/extract_pdfs.py
python src/data_processing/extract_slides.py

# 4. Generate dataset (choose one)
python src/data_processing/create_dataset.py                     # Fast (15 min)
python src/data_processing/create_dataset_llm.py --test          # Test LLM (2 min)
python src/data_processing/create_dataset_llm.py                 # LLM full (60 min)

# 5. Check quality ⚠️ CRITICAL
python diagnose.py

# 6. Train model
python src/training/fine_tune.py

# 7. Build vector database
python src/data_processing/build_vectordb.py

# 8. Test model
python src/inference/query_processor.py

# 9. Evaluate
python src/evaluation/evaluate_model.py
```

---

## 🎓 Usage Examples

### Query the Model
```python
from src.inference.query_processor import QueryProcessor

processor = QueryProcessor()
answer = processor.process("What causes deadlock?")
print(answer)
```

### Generate Diagrams
```python
from src.inference.figure_generator import FigureGenerator

gen = FigureGenerator()
diagram = gen.generate_ascii_art("TCP handshake")
print(diagram['art'])
```

### Extract Images
```bash
python src/data_processing/extract_images.py --input-dir data/raw/slides
```

---

## 🧪 Testing

```bash
python test_improvements.py   # Test all 11 improvements
test_multimedia.bat           # Test multimedia features
```

---

## 🔧 Batch Scripts

Convenient Windows batch files for quick execution:

| Script | Description |
|--------|-------------|
| `run_all.bat` | Complete pipeline (extract → dataset → train) |
| `process_data.bat` | Extract PDFs/slides + create dataset |
| `train.bat` | Train model only |
| `test.bat` | Run evaluation |
| `test_multimedia.bat` | Test image extraction |
| `evaluate.bat` | Full evaluation suite |

---

## 📊 Project Improvements

This project includes 11 major enhancements:

1. **Hybrid Retrieval**: Dense (embeddings) + sparse (BM25) → Better context
2. **Smart Caching**: Redis-backed query cache → 50x faster responses
3. **Image Extraction**: OCR + vision models → Diagram understanding
4. **Concept Mapping**: Auto-generate mind maps from content
5. **YouTube Integration**: Suggest relevant video tutorials
6. **Paper Search**: Find research papers on topics (Semantic Scholar API)
7. **Refusal Training**: Say "I don't know" → Prevent hallucinations
8. **Advanced Metrics**: BLEU, ROUGE, BERTScore, Faithfulness
9. **Memory Optimization**: 4-bit quantization, Flash Attention 2
10. **LLM Augmentation**: High-quality Q&A generation (85-95/100)
11. **Comprehensive Diagnostics**: Quality scoring and fix suggestions

See [IMPROVEMENTS_GUIDE.md](docs/IMPROVEMENTS_GUIDE.md) for details.

---

## 💡 Tips for Success

1. **Always check data quality** with `diagnose.py` before training
2. **Use LLM-enhanced datasets** (Option B in Step 3) for production models
3. **Monitor training logs** - watch for overfitting (eval loss increases)
4. **Target faithfulness > 90%** - most important metric
5. **Test with out-of-scope questions** - ensure model refuses properly
6. **Use batch scripts** for convenience: `process_data.bat`, `train.bat`

---

## 🤝 Support & Contribution

**Having issues?**
1. Check [Troubleshooting Guide](#-troubleshooting-guide) above
2. Run `python diagnose.py` for data quality issues
3. See [COMPLETE_WORKFLOW_AND_FIXES.md](docs/COMPLETE_WORKFLOW_AND_FIXES.md) for hallucination fixes

**Want to contribute?**
- Report issues with training logs and diagnose.py output
- Share improvements to dataset generation
- Add support for new file formats (DOCX, HTML, etc.)

---

## 📦 Requirements

**Software:**
- Python 3.8-3.11
- CUDA 11.8+ (for GPU)
- PyTorch 2.1.0+
- Transformers 4.36.0+

**Optional:**
- Ollama (for LLM-enhanced datasets)
- Tesseract OCR (for diagram extraction)
- Redis (for query caching)

**Install all dependencies:**
```bash
pip install -r requirements.txt
```

---

## 📜 License

This project is for educational purposes. Ensure you have rights to use the training materials (PDFs, slides).

---

**🎉 Ready to start? Follow [Step 1](#step-1-prepare-your-training-materials-5-minutes) above!**

**⚡ In a hurry? See [QUICKSTART.md](QUICKSTART.md) for condensed commands.**
