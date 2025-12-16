# 📁 Project File Index

## 📖 Documentation Files

| File | Purpose | When to Read |
|------|---------|-------------|
| **README.md** | Main documentation, comprehensive project overview | First - Start here |
| **QUICKSTART.md** | Quick start guide with step-by-step instructions | When you want to get started quickly |
| **EXPLANATION.md** | Technical deep dive into architecture and design | When you want to understand how it works |
| **PROJECT_SUMMARY.md** | Complete project summary and next steps | After setup, before starting work |
| **GITHUB_INSTRUCTIONS.md** | Guide for uploading to GitHub | When ready to upload to GitHub |
| **FILE_INDEX.md** | This file - navigation guide | Anytime you're lost |

## ⚙️ Configuration Files

| File | Purpose |
|------|---------|
| `.env.template` | Template for environment variables |
| `.env` | Your actual API keys (auto-created, don't commit) |
| `.gitignore` | Git ignore patterns |
| `requirements.txt` | Python package dependencies |
| `setup.py` | Project initialization script |
| `configs/training_config.yaml` | Training parameters |
| `configs/model_config.yaml` | Model and inference settings |
| `configs/api_config.yaml` | API configurations |

## 🚀 Quick Start Scripts (Windows)

| Script | Purpose | When to Run |
|--------|---------|------------|
| `run_all.bat` | Complete setup (venv + dependencies + directories) | **First** - Initial setup |
| `process_data.bat` | Process all course materials | **Second** - After adding materials |
| `train.bat` | Fine-tune the model | **Third** - After data processing |
| `test.bat` | Interactive testing | **Fourth** - After training |
| `evaluate.bat` | Evaluate on test questions | **Fifth** - Final evaluation |

## 💻 Source Code Structure

### Core Utilities (`src/utils/`)
- `config.py` - Configuration management
- `helpers.py` - Utility functions

### Data Processing (`src/data_processing/`)
- `extract_slides.py` - Extract from PowerPoint
- `extract_pdfs.py` - Extract from PDF books
- `create_dataset.py` - Create training dataset
- `build_vectordb.py` - Build vector database for RAG

### Training (`src/training/`)
- `fine_tune.py` - Main fine-tuning script with LoRA

### Inference (`src/inference/`)
- `model_loader.py` - Load fine-tuned model
- `rag_system.py` - RAG implementation
- `query_processor.py` - Main query processing (use this!)

### Enrichment (`src/enrichment/`)
- `youtube_suggester.py` - YouTube video suggestions
- `paper_search.py` - arXiv paper search
- `concept_mapper.py` - Concept relationships

### Evaluation (`src/evaluation/`)
- `evaluate_model.py` - Model evaluation with metrics

## 📊 Data Directories

### Input Data (`data/raw/`)
**You add files here:**
- `slides/` - PowerPoint presentations (.pptx, .ppt)
- `books/` - PDF textbooks (.pdf)
- `notes/` - Additional notes (.txt, .md)

### Processed Data (`data/processed/`)
**Auto-generated:**
- `slides/` - Extracted slide content
- `books/` - Extracted book content
- `train.jsonl` - Training dataset
- `val.jsonl` - Validation dataset

### Evaluation (`data/evaluation/`)
- `endsem_questions.json` - Sample test questions (provided)

## 🤖 Model & Output Directories

### Models (`models/`)
- `base/` - Base Qwen3 model (auto-downloaded)
- `fine_tuned/` - Your fine-tuned model (created during training)

### Vector Database (`vectordb/`)
- `course_materials/` - ChromaDB vector database (auto-created)

### Outputs (`outputs/`)
- `logs/` - Training logs
- `results/` - Evaluation results
- `responses/` - Sample responses

## 🎯 Workflow Guide

### Phase 1: Setup
1. Read **README.md**
2. Run `run_all.bat`
3. Edit `.env` with API keys (optional)

### Phase 2: Data Preparation
1. Add course materials to `data/raw/`
2. Run `process_data.bat`
3. Check `data/processed/` for output

### Phase 3: Training
1. Review `configs/training_config.yaml`
2. Run `train.bat`
3. Monitor with TensorBoard

### Phase 4: Testing
1. Run `test.bat` for interactive mode
2. Or use: `python src/inference/query_processor.py --question "your question"`

### Phase 5: Evaluation
1. Run `evaluate.bat`
2. Check `outputs/results/` for metrics

### Phase 6: GitHub Upload
1. Read **GITHUB_INSTRUCTIONS.md**
2. Initialize git and push
3. Submit repository URL

## 🔍 Quick Reference

### Need Help With... | Check File...
- Getting started → **QUICKSTART.md**
- Understanding architecture → **EXPLANATION.md**
- Setting up → **README.md**
- Uploading to GitHub → **GITHUB_INSTRUCTIONS.md**
- Finding files → **FILE_INDEX.md** (this file)

### Want to...  | Run Command...
- Setup project → `run_all.bat` or `python setup.py`
- Process data → `process_data.bat`
- Train model → `train.bat` or `python src/training/fine_tune.py`
- Test system → `test.bat` or `python src/inference/query_processor.py -i`
- Evaluate → `evaluate.bat` or `python src/evaluation/evaluate_model.py`

### Troubleshooting | Check...
- Error messages → Console output
- Training issues → `outputs/logs/`
- Data issues → `data/processed/`
- Model issues → `models/fine_tuned/`

## 📚 Key Configuration Files

### For Training
Edit `configs/training_config.yaml`:
- Model size
- Batch size
- Learning rate
- Number of epochs

### For Inference
Edit `configs/model_config.yaml`:
- Temperature
- Max tokens
- Prompt templates

### For APIs
Edit `.env`:
- YouTube API key
- Hugging Face token

## ✅ File Checklist

Before starting, ensure you have:
- [x] All Python files in `src/`
- [x] All config files in `configs/`
- [x] Documentation files in root
- [x] `.env.template` file
- [x] `.gitignore` file
- [x] `requirements.txt`
- [x] Sample questions in `data/evaluation/`

## 🎓 For BTP Submission

Include in your submission:
1. **GitHub repository URL**
2. **README.md** (already excellent)
3. **Evaluation results** from `outputs/results/`
4. **Sample outputs** (optional screenshots)

## 💡 Pro Tips

1. **Always activate venv** before running Python scripts
2. **Check logs** in `outputs/logs/` if something fails
3. **Start small** - use a smaller model (1.5B) for testing
4. **Monitor memory** during training
5. **Save work** - commit to git regularly

## 📞 Getting Help

1. Check relevant documentation file
2. Review error messages carefully
3. Check `outputs/logs/` for training logs
4. Verify data in `data/processed/`
5. Ensure API keys are correct in `.env`

---

**Navigation Tips:**
- 📄 = Documentation
- ⚙️ = Configuration
- 💻 = Source code
- 🚀 = Quick start scripts
- 📊 = Data
- 🤖 = Models/Outputs

**Start Here:** README.md → run_all.bat → Add materials → process_data.bat → train.bat
