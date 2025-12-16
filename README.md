# Qwen2.5 Fine-tuning for OS & Networks RAG System

Advanced RAG system with fine-tuned Qwen2.5-7B-Instruct for Operating Systems and Computer Networks Q&A.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

---

## 🚀 Quick Start

**New to this project?** → [**QUICKSTART.md**](QUICKSTART.md)

```bash
# 1. Setup
python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt

# 2. Add training materials to data/raw/pdfs/ and data/raw/slides/

# 3. Run complete pipeline
run_all.bat
```

---

## 🎯 What This Does

Fine-tune Qwen2.5-7B-Instruct with:
- **RAG**: Hybrid retrieval (dense + BM25 + reranking)
- **LoRA**: Efficient fine-tuning (4-bit quantization, Flash Attention 2)
- **Multimedia**: Extract images, OCR diagrams, generate figures
- **Advanced Evaluation**: ROUGE, BERTScore, semantic similarity

---

## ⚡ Key Features

| Feature | Description |
|---------|-------------|
| **Hybrid RAG** | Dense (mpnet) + Sparse (BM25) + Cross-encoder reranking |
| **Fast Training** | Flash Attention 2 (2-4x faster) + 4-bit quantization |
| **Multimedia** | Extract images, OCR diagrams, BLIP captions, ASCII/Mermaid generation |
| **7 Question Types** | Explanation, comparison, factual, procedural, conceptual, scenario, recall |
| **Advanced Metrics** | ROUGE-L, BERTScore, semantic similarity, factual consistency |

---

## 📁 Structure

```
d:\iREL\p1\
├── configs/          # Configuration (model, training, API)
├── data/
│   ├── raw/         # Add your PDFs and slides here
│   ├── processed/   # Extracted content
│   └── evaluation/  # Test questions
├── src/
│   ├── data_processing/  # Extract & process
│   ├── training/         # Fine-tune model
│   ├── inference/        # RAG + query
│   └── evaluation/       # Metrics
├── docs/            # Documentation
└── *.bat           # Automation scripts
```

---

## 📚 Documentation

| Guide | Description |
|-------|-------------|
| **[QUICKSTART.md](QUICKSTART.md)** | **Start here** - Complete setup from scratch |
| [docs/IMPROVEMENTS_GUIDE.md](docs/IMPROVEMENTS_GUIDE.md) | All 11 improvements explained |
| [docs/MULTIMEDIA_GUIDE.md](docs/MULTIMEDIA_GUIDE.md) | Image extraction, OCR, vision models |
| [docs/QUICK_REFERENCE.md](docs/QUICK_REFERENCE.md) | Quick commands & API reference |

---

## 🎓 Usage

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
test_improvements.py   # Test all 11 improvements
test_multimedia.bat    # Test multimedia features
```

---

## 🔧 Batch Scripts

```bash
process_data.bat   # Extract PDFs, slides, build vector DB
train.bat         # Fine-tune model
evaluate.bat      # Run evaluation
run_all.bat       # Complete pipeline
```

---

## 📊 Improvements

11 major enhancements:
1. ✅ 7 diverse question types
2. ✅ Data augmentation
3. ✅ Flash Attention 2 + 4-bit quantization
4. ✅ Hybrid retrieval (dense + BM25)
5. ✅ Advanced evaluation metrics
6. ✅ Image extraction
7. ✅ OCR with Tesseract
8. ✅ Vision models (BLIP + CLIP)
9. ✅ Diagram generation
10. ✅ Text preprocessing
11. ✅ Comprehensive docs

Details: [docs/IMPROVEMENTS_GUIDE.md](docs/IMPROVEMENTS_GUIDE.md)

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| Out of memory | Reduce batch size, enable 4-bit quantization |
| Slow training | Install Flash Attention 2, use GPU |
| OCR not working | Install Tesseract: `choco install tesseract` |
| Model download fails | Check internet, clear `~/.cache/huggingface/` |

---

## 📦 Requirements

- Python 3.8+
- GPU with 16GB+ VRAM (or CPU, slower)
- 20GB+ storage

```bash
pip install -r requirements.txt

# Optional (for OCR):
choco install tesseract  # Windows
```

---

**🚀 Get started: [QUICKSTART.md](QUICKSTART.md)**
