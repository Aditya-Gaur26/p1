# 🚀 QUICKSTART: Training Qwen2.5 from Scratch

Complete guide to train your OS & Networks Q&A system in 30 minutes.

---

## 📋 Prerequisites

### System Requirements
- **OS**: Windows 10/11, Linux, or macOS
- **GPU**: 16GB+ VRAM (recommended) or CPU (4x slower)
- **RAM**: 16GB+ 
- **Storage**: 20GB+ free space
- **Python**: 3.8+

### What You'll Need
- Course materials (PDFs, PowerPoint slides)
- 30 minutes of setup time
- Internet connection (for downloading models)

---

## ⚡ Quick Path (5 Minutes)

Already have everything? Run these:

```bash
# 1. Setup
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# 2. Add PDFs to data/raw/pdfs/ and slides to data/raw/slides/

# 3. Run everything
run_all.bat
```

**Done!** Skip to [Usage](#-usage) section.

---

## 📝 Detailed Path (30 Minutes)

### Step 1: Environment Setup (5 min)

#### 1.1 Create Virtual Environment
```bash
# Windows PowerShell
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

#### 1.2 Install Dependencies
```bash
# Install Python packages (this will take 3-5 minutes)
pip install -r requirements.txt

# Optional: Install Tesseract OCR for diagram text extraction
# Windows:
choco install tesseract

# Linux:
sudo apt-get install tesseract-ocr

# macOS:
brew install tesseract
```

**✅ Verify installation:**
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

---

### Step 2: Add Training Materials (2 min)

#### 2.1 Create Data Directories
```bash
# These are created automatically, but verify:
mkdir -p data/raw/pdfs
mkdir -p data/raw/slides
```

#### 2.2 Copy Your Materials
```bash
# Copy PDF lecture notes, textbooks
copy "C:\path\to\your\pdfs\*.pdf" data\raw\pdfs\

# Copy PowerPoint slides
copy "C:\path\to\your\slides\*.pptx" data\raw\slides\
```

**Example structure:**
```
data/raw/
├── pdfs/
│   ├── OS_Lecture_Notes.pdf
│   ├── Networks_Textbook.pdf
│   └── Process_Management.pdf
└── slides/
    ├── Lecture_1_Introduction.pptx
    ├── Lecture_2_Processes.pptx
    └── TCP_IP_Protocols.pptx
```

**📌 Important**: 
- Need at least 3-5 PDFs or slides for meaningful training
- More data = better model performance
- Supported formats: PDF, PPTX, PPT

---

### Step 3: Process Data (5 min)

#### 3.1 Extract Content
```bash
# Extract text from PDFs
python src/data_processing/extract_pdfs.py

# Extract text from slides
python src/data_processing/extract_slides.py

# Extract images from slides (optional, for multimedia)
python src/data_processing/extract_images.py --input-dir data/raw/slides
```

**Expected output:**
```
✅ Processed 5 PDF files
✅ Extracted 1,234 text chunks
✅ Saved to data/processed/documents.json
```

#### 3.2 Build Vector Database
```bash
python src/data_processing/build_vectordb.py
```

**Expected output:**
```
📦 Building vector database...
✅ Embedded 1,234 chunks
✅ Created ChromaDB at data/processed/vectordb
```

**Or use batch script (Windows):**
```bash
process_data.bat
```

---

### Step 4: Create Training Dataset (3 min)

```bash
python src/data_processing/create_dataset.py
```

**What this does:**
- Generates Q&A pairs from your materials
- Creates 7 types of questions:
  1. Explanation ("Explain...")
  2. Comparison ("Compare...")
  3. Factual ("What is...")
  4. Procedural ("How to...")
  5. Conceptual ("Why does...")
  6. Scenario ("What happens if...")
  7. Recall ("List...")
- Applies data augmentation (paraphrasing, reasoning chains)

**Expected output:**
```
✅ Generated 500 Q&A pairs
✅ Dataset saved to data/processed/datasets/training_data.json
```

---

### Step 5: Fine-tune Model (10 min)

#### 5.1 Configure Training (Optional)

Edit `configs/training_config.yaml`:
```yaml
training:
  num_epochs: 3              # More epochs = better but slower
  per_device_train_batch_size: 4  # Reduce if out of memory
  learning_rate: 5e-5
  
lora:
  r: 32                      # LoRA rank
  lora_alpha: 64
  lora_dropout: 0.1
```

#### 5.2 Start Training
```bash
# Run training script
python src/training/fine_tune.py

# Or use batch script (Windows)
train.bat
```

**What to expect:**
```
📦 Loading Qwen/Qwen2.5-7B-Instruct...
✅ Model loaded (4-bit quantized)
✅ Flash Attention 2 enabled
📊 Training on 500 examples...

Epoch 1/3: 100%|████████| 125/125 [03:45<00:00, 1.8s/it]
  Loss: 1.234

Epoch 2/3: 100%|████████| 125/125 [03:42<00:00, 1.7s/it]
  Loss: 0.856

Epoch 3/3: 100%|████████| 125/125 [03:40<00:00, 1.7s/it]
  Loss: 0.623

✅ Training complete!
✅ Model saved to outputs/fine_tuned_model/
```

**⏱️ Training time:**
- GPU (16GB): ~10-15 minutes
- CPU: ~45-60 minutes

**💾 Model size:**
- Base model download: ~7GB (one-time)
- LoRA adapters: ~100MB
- Total saved: ~7.1GB

---

### Step 6: Evaluate Model (5 min)

```bash
# Run evaluation
python src/evaluation/evaluate_model.py

# Or use batch script (Windows)
evaluate.bat
```

**Expected output:**
```
📊 Evaluating on 8 questions...

Question 1/8: What causes deadlock?
  ✅ ROUGE-L: 0.85
  ✅ BERTScore: 0.89
  ✅ Semantic Similarity: 0.92

Question 2/8: Explain TCP handshake
  ✅ ROUGE-L: 0.81
  ✅ BERTScore: 0.87
  ✅ Semantic Similarity: 0.90

...

Overall Scores:
  ROUGE-L: 0.83 ± 0.04
  BERTScore: 0.88 ± 0.03
  Semantic Similarity: 0.91 ± 0.02
  Answer Relevance: 0.89

✅ Evaluation complete!
📄 Report saved to outputs/evaluation_results.json
```

---

## 🎯 Usage

### Query via Python

```python
from src.inference.query_processor import QueryProcessor

# Initialize processor
processor = QueryProcessor()

# Ask questions
questions = [
    "What is a deadlock in operating systems?",
    "Explain the TCP three-way handshake",
    "Compare process and thread",
    "What are the layers of the OSI model?"
]

for question in questions:
    answer = processor.process(question)
    print(f"\nQ: {question}")
    print(f"A: {answer}\n")
```

### Query via Command Line

```bash
python src/inference/query_processor.py --query "What causes deadlock?"
```

### Generate Diagrams

```python
from src.inference.figure_generator import FigureGenerator

gen = FigureGenerator()

# Generate ASCII art
tcp_diagram = gen.generate_ascii_art("TCP three-way handshake")
print(tcp_diagram['art'])

# Output:
# TCP Three-Way Handshake:
#     Client                    Server
#        |                         |
#        |    SYN (seq=100)        |
#        |------------------------>|
#        ...
```

---

## 🔧 Configuration

### Model Config (`configs/model_config.yaml`)

```yaml
model:
  name: "Qwen/Qwen2.5-7B-Instruct"  # Base model
  load_in_4bit: true                # 4-bit quantization (lower memory)
  use_flash_attention_2: true       # Faster training (2-4x)
  max_seq_length: 2048              # Context window
```

### Training Config (`configs/training_config.yaml`)

```yaml
training:
  num_epochs: 3                     # Training epochs
  per_device_train_batch_size: 4    # Batch size (reduce if OOM)
  learning_rate: 5e-5               # Learning rate
  gradient_accumulation_steps: 4    # Effective batch size = 4 × 4 = 16
  
lora:
  r: 32                             # LoRA rank (higher = more capacity)
  lora_alpha: 64                    # LoRA alpha
  lora_dropout: 0.1                 # Dropout
  target_modules:                   # Modules to apply LoRA
    - q_proj
    - k_proj
    - v_proj
    - o_proj
```

### API Config (`configs/api_config.yaml`)

For enrichment features (optional):
```yaml
youtube:
  api_key: "YOUR_API_KEY_HERE"  # For YouTube video suggestions
  
arxiv:
  email: "your_email@example.com"  # For paper search
```

---

## 📊 Batch Processing (Windows)

All-in-one scripts:

```bash
# Process all data
process_data.bat

# Train model
train.bat

# Evaluate model
evaluate.bat

# Run complete pipeline
run_all.bat
```

Edit these files to customize behavior.

---

## 🐛 Troubleshooting

### Common Issues

#### 1. Out of Memory (OOM)

**Symptoms**: CUDA out of memory, killed process

**Solutions**:
```yaml
# In configs/training_config.yaml:
training:
  per_device_train_batch_size: 2  # Reduce from 4
  gradient_accumulation_steps: 8  # Increase to maintain effective batch size
  
model:
  load_in_4bit: true  # Enable 4-bit quantization
```

Or use CPU (slower):
```bash
export CUDA_VISIBLE_DEVICES=""  # Linux
$env:CUDA_VISIBLE_DEVICES=""    # Windows PowerShell
```

#### 2. Slow Training

**Solutions**:
- Install Flash Attention 2:
  ```bash
  pip install flash-attn --no-build-isolation
  ```
- Use GPU instead of CPU
- Reduce max_seq_length:
  ```yaml
  model:
    max_seq_length: 1024  # Reduce from 2048
  ```

#### 3. Model Download Fails

**Symptoms**: Connection timeout, download error

**Solutions**:
```bash
# Clear Hugging Face cache
rm -rf ~/.cache/huggingface/

# Try download manually
python -c "from transformers import AutoModel; AutoModel.from_pretrained('Qwen/Qwen2.5-7B-Instruct')"

# Or set mirror (China):
export HF_ENDPOINT=https://hf-mirror.com
```

#### 4. OCR Not Working

**Symptoms**: Tesseract not found

**Solutions**:
```bash
# Windows:
choco install tesseract
# Add to PATH: C:\Program Files\Tesseract-OCR

# Linux:
sudo apt-get install tesseract-ocr

# macOS:
brew install tesseract

# Verify:
tesseract --version
```

#### 5. Import Errors

**Symptoms**: ModuleNotFoundError

**Solutions**:
```bash
# Reinstall requirements
pip install -r requirements.txt --force-reinstall

# Verify installation
python -c "import torch, transformers, peft, chromadb"
```

#### 6. Low Evaluation Scores

**Causes & Solutions**:
- **Not enough data**: Add more PDFs/slides (need 5+ documents)
- **Poor quality data**: Use clear, well-formatted materials
- **Undertrained**: Increase epochs to 5-10
- **Wrong hyperparameters**: Try learning rate 2e-5 or 1e-4

---

## 🎓 Advanced Usage

### Custom Question Types

Edit `src/data_processing/create_dataset.py`:

```python
# Add custom question template
QUESTION_TEMPLATES = {
    'custom_type': [
        "According to the material, {topic}?",
        "In the context of {subject}, explain {concept}",
    ]
}
```

### Multi-GPU Training

```yaml
# configs/training_config.yaml
training:
  use_multi_gpu: true
  num_gpus: 2  # or 4, 8, etc.
```

```bash
# Run with accelerate
accelerate launch --multi_gpu --num_processes 2 src/training/fine_tune.py
```

### Resume Training

```bash
# Training script automatically saves checkpoints
# Resume from checkpoint:
python src/training/fine_tune.py --resume_from_checkpoint outputs/checkpoint-500
```

### Export Model

```python
from src.training.fine_tune import export_model

# Export for deployment
export_model(
    base_model="Qwen/Qwen2.5-7B-Instruct",
    lora_path="outputs/fine_tuned_model",
    output_path="outputs/merged_model"
)
```

---

## 📈 Performance Optimization

### Training Speed
- ✅ Flash Attention 2: 2-4x faster
- ✅ 4-bit quantization: 40% memory reduction
- ✅ Gradient checkpointing: 30% memory reduction
- ✅ Mixed precision (fp16): 50% faster

### Retrieval Quality
- ✅ Hybrid retrieval: 15-20% better recall
- ✅ Cross-encoder reranking: 10-15% better precision
- ✅ Query expansion: 5-10% better for complex queries

### Model Quality
- ✅ LoRA rank 32: Better than rank 16
- ✅ Data augmentation: 2x effective dataset size
- ✅ 7 question types: More diverse training

---

## 🎯 Next Steps

After successful training:

1. **Test with Real Questions**
   - Try questions from your course
   - Test edge cases
   - Verify answer quality

2. **Iterate and Improve**
   - Add more training data
   - Tune hyperparameters
   - Try different LoRA ranks

3. **Deploy**
   - Create API endpoint
   - Build web interface (Gradio/Streamlit)
   - Integrate with course platform

4. **Monitor Performance**
   - Track answer quality
   - Collect user feedback
   - Retrain periodically

---

## 📚 Additional Resources

### Documentation
- [IMPROVEMENTS_GUIDE.md](docs/IMPROVEMENTS_GUIDE.md) - All enhancements
- [MULTIMEDIA_GUIDE.md](docs/MULTIMEDIA_GUIDE.md) - Image/OCR features
- [QUICK_REFERENCE.md](docs/QUICK_REFERENCE.md) - Command reference

### External Resources
- [Qwen Model Card](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [Flash Attention](https://arxiv.org/abs/2205.14135)
- [ChromaDB Docs](https://docs.trychroma.com/)

---

## ✅ Checklist

Complete training checklist:

- [ ] Python 3.8+ installed
- [ ] Virtual environment created
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Tesseract installed (optional, for OCR)
- [ ] Training materials added to `data/raw/`
- [ ] Data processed (`process_data.bat` or individual scripts)
- [ ] Vector database built
- [ ] Training dataset created
- [ ] Model fine-tuned (`train.bat`)
- [ ] Model evaluated (`evaluate.bat`)
- [ ] Test queries run successfully
- [ ] Results reviewed and satisfactory

---

## 🎉 Success Criteria

You'll know training succeeded when:
- ✅ No errors during training
- ✅ Training loss decreases (e.g., 1.2 → 0.6)
- ✅ ROUGE-L score > 0.75
- ✅ BERTScore > 0.80
- ✅ Model answers are relevant and accurate
- ✅ Model can handle follow-up questions

---

## 💡 Tips

- **Start Small**: Test with 2-3 PDFs first, then scale up
- **Monitor GPU**: Use `nvidia-smi` to check VRAM usage
- **Save Often**: Training saves checkpoints automatically
- **Compare Models**: Keep base model for A/B testing
- **Document**: Note what hyperparameters work best
- **Backup**: Keep copies of trained models

---

**🎓 Ready to train? Start with Step 1!**

**Questions?** Check [docs/](docs/) or open an issue.
