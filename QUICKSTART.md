# Operating Systems & Networks - Quick Start Guide

## 🚀 Quick Start (TL;DR)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Setup project
python setup.py

# 3. Add your course materials
#    - Slides → data/raw/slides/
#    - Books → data/raw/books/

# 4. Process data
python src/data_processing/extract_slides.py
python src/data_processing/extract_pdfs.py
python src/data_processing/create_dataset.py
python src/data_processing/build_vectordb.py

# 5. Fine-tune model (choose model size based on your GPU)
python src/training/fine_tune.py  # Default: Qwen2.5-7B

# 6. Test it!
python src/inference/query_processor.py --interactive

# 7. Evaluate
python src/evaluation/evaluate_model.py
```

## 📋 Detailed Workflow

### Step 1: Installation & Setup

```bash
# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run setup
python setup.py

# Configure API keys (optional but recommended)
# Edit .env file with your YouTube API key
```

### Step 2: Add Course Materials

Place your course materials:
- **Slides**: `data/raw/slides/` (.pptx, .ppt)
- **Books**: `data/raw/books/` (.pdf)
- **Notes**: `data/raw/notes/` (.txt, .md)

### Step 3: Data Processing

```bash
# Extract from slides
python src/data_processing/extract_slides.py

# Extract from PDFs
python src/data_processing/extract_pdfs.py

# Create training dataset
python src/data_processing/create_dataset.py

# Build vector database for RAG
python src/data_processing/build_vectordb.py
```

### Step 4: Fine-tuning

```bash
# Default (Qwen2.5-7B with LoRA)
python src/training/fine_tune.py

# Use smaller model (for limited GPU)
# Edit configs/training_config.yaml
# Change model.name to "Qwen/Qwen2.5-1.5B-Instruct"

# Monitor training
tensorboard --logdir models/fine_tuned/logs
```

**Model Selection Guide:**
- **4-6 GB VRAM**: Use Qwen2.5-0.6B or 1.5B
- **8-12 GB VRAM**: Use Qwen2.5-3B
- **16+ GB VRAM**: Use Qwen2.5-7B (recommended)
- **32+ GB VRAM**: Use Qwen2.5-14B

### Step 5: Testing

```bash
# Interactive mode
python src/inference/query_processor.py --interactive

# Single question
python src/inference/query_processor.py --question "What is process scheduling?"

# Without enrichment features
python src/inference/query_processor.py --no-enrichment
```

### Step 6: Evaluation

```bash
# Evaluate on endsem questions
python src/evaluation/evaluate_model.py --test-file data/evaluation/endsem_questions.json

# Results will be saved to outputs/results/
```

## 🎯 Key Features

### 1. RAG (Retrieval-Augmented Generation)
- Retrieves relevant context from course materials
- Reduces hallucination
- Provides source citations

### 2. YouTube Video Suggestions
- Automatically finds relevant educational videos
- Filters by quality (views, ratings)
- Provides direct links

### 3. Research Paper Recommendations
- Searches arXiv for related papers
- Provides abstracts and links
- Categorized by relevance

### 4. Fine-tuned Model
- Parameter-efficient training with LoRA
- Optimized for educational Q&A
- Supports multiple model sizes

## 🔧 Configuration

### Training Configuration (`configs/training_config.yaml`)

Key parameters to adjust:
```yaml
model:
  name: "Qwen/Qwen2.5-7B-Instruct"  # Change model size here

lora:
  r: 16                    # LoRA rank (8-64)
  lora_alpha: 32           # Usually 2x rank

training:
  num_train_epochs: 3      # Number of epochs
  per_device_train_batch_size: 4  # Reduce if OOM
  learning_rate: 2.0e-4    # Learning rate
```

### API Keys (`.env`)

```bash
# YouTube Data API (for video suggestions)
YOUTUBE_API_KEY=your_key_here

# Hugging Face Token (for downloading models)
HF_TOKEN=your_token_here
```

Get API keys:
- **YouTube**: https://console.cloud.google.com/
- **Hugging Face**: https://huggingface.co/settings/tokens

## 📊 Example Usage

### Example 1: Basic Question
```python
from src.inference.query_processor import QueryProcessor

processor = QueryProcessor()
result = processor.answer("What is virtual memory?")

print(result['answer'])
print(result['youtube_videos'])
print(result['research_papers'])
```

### Example 2: Batch Questions
```python
questions = [
    "Explain process scheduling",
    "What is TCP/IP?",
    "How does virtual memory work?"
]

for q in questions:
    result = processor.answer(q)
    print(f"\nQ: {q}")
    print(f"A: {result['answer'][:200]}...")
```

## 🐛 Troubleshooting

### Issue: CUDA Out of Memory
**Solution**: 
```bash
# Use smaller model or reduce batch size
# Edit configs/training_config.yaml:
training:
  per_device_train_batch_size: 2  # Reduce from 4
  gradient_accumulation_steps: 8  # Increase from 4
```

### Issue: No videos/papers found
**Solution**: 
- Check API keys in `.env`
- System gracefully degrades - base model still works
- Fallback suggestions are provided

### Issue: Vector DB not found
**Solution**: 
```bash
# Rebuild vector database
python src/data_processing/build_vectordb.py
```

### Issue: Model not generating proper responses
**Solution**: 
- Ensure training dataset has enough examples (>100)
- Try training for more epochs
- Check if base model downloaded correctly

## 📈 Evaluation Metrics

The system tracks:
- **ROUGE scores**: Measure text overlap
- **F1 score**: Precision and recall
- **BLEU score**: Translation-style quality
- **Enrichment stats**: YouTube/paper suggestions
- **Processing time**: Response speed

## 🎓 Grading Features

This project includes:
- ✅ Fine-tuned model (LoRA/PEFT)
- ✅ RAG with vector database
- ✅ YouTube video suggestions
- ✅ Research paper search
- ✅ Question answering
- ✅ Source citations
- ✅ Comprehensive evaluation
- ✅ Modular architecture
- ✅ Configuration management
- ✅ Error handling

## 📝 Notes

1. **First Run**: Training takes 4-12 hours depending on:
   - Model size
   - Dataset size
   - GPU specifications

2. **Incremental Improvement**: You can:
   - Add more course materials anytime
   - Rebuild vector DB
   - Fine-tune further with new data

3. **GPU Requirements**: 
   - CPU-only works but is VERY slow
   - Recommended: NVIDIA GPU with 8+ GB VRAM

4. **Data Quality**: Better course materials = better results
   - Clear, well-structured slides
   - Comprehensive textbooks
   - Diverse examples

## 🚀 Advanced Features

### Custom Prompts
Edit `configs/model_config.yaml` to customize prompts

### Batch Processing
Process multiple questions efficiently

### Web Interface (Optional)
```bash
# Create Gradio interface
pip install gradio
# Then create a simple UI in notebooks/demo.ipynb
```

## 📞 Support

For issues:
1. Check this guide
2. Review error messages
3. Check `outputs/logs/` for training logs
4. Verify data in `data/processed/`

---

**Happy Learning! 🎓**
