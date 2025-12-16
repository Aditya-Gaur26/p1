# 🚀 GET STARTED - Your First Steps

Welcome! This is your **fastest path** to getting the project running.

## ⚡ Super Quick Start (5 minutes)

If you just want to see what this project does:

### Option 1: Windows Users
```batch
REM Double-click or run:
run_all.bat
```

### Option 2: Command Line
```bash
# Install dependencies
pip install -r requirements.txt

# Setup project
python setup.py
```

That's it! The project structure is now ready.

## 📋 What You Need to Provide

The project is **90% complete**. You just need to add:

1. **Your course materials** (slides and books)
   - PowerPoint slides → `data/raw/slides/`
   - PDF books → `data/raw/books/`

2. **API keys** (optional but recommended)
   - YouTube API key for video suggestions
   - Edit the `.env` file

## 🎯 Complete Workflow (Step-by-Step)

### Step 1: Setup (5 minutes)
```bash
# Already done if you ran run_all.bat!
pip install -r requirements.txt
python setup.py
```

### Step 2: Add Course Materials (You do this)
Place your Operating Systems & Networks materials:
- **Slides**: Copy `.pptx` files to `data/raw/slides/`
- **Books**: Copy `.pdf` files to `data/raw/books/`

Example:
```
data/raw/slides/
  ├── OS_Lecture_01.pptx
  ├── OS_Lecture_02.pptx
  └── Networks_Introduction.pptx

data/raw/books/
  ├── Operating_Systems_Concepts.pdf
  └── Computer_Networks_Tanenbaum.pdf
```

### Step 3: Process Data (10-30 minutes)

**Windows:**
```batch
process_data.bat
```

**Command Line:**
```bash
python src/data_processing/extract_slides.py
python src/data_processing/extract_pdfs.py
python src/data_processing/create_dataset.py
python src/data_processing/build_vectordb.py
```

**What this does:**
- Extracts text from your slides and books
- Creates training dataset
- Builds vector database for RAG

### Step 4: Configure API Keys (Optional, 2 minutes)

Edit `.env` file:
```env
YOUTUBE_API_KEY=your_actual_api_key_here
HF_TOKEN=your_huggingface_token_here
```

**Get API keys:**
- YouTube: https://console.cloud.google.com/ (free)
- Hugging Face: https://huggingface.co/settings/tokens (free)

**Note:** System works without API keys, but with reduced features

### Step 5: Fine-tune Model (4-12 hours)

**Choose model size based on your GPU:**

| GPU VRAM | Recommended Model | Config Change |
|----------|------------------|---------------|
| 4-6 GB | Qwen2.5-1.5B | Edit `configs/training_config.yaml` |
| 8-12 GB | Qwen2.5-3B | Edit `configs/training_config.yaml` |
| 16+ GB | Qwen2.5-7B | Default (no change needed) |

**Run training:**

Windows:
```batch
train.bat
```

Command Line:
```bash
python src/training/fine_tune.py
```

**Monitor progress:**
```bash
# In another terminal
tensorboard --logdir models/fine_tuned/logs
# Open http://localhost:6006
```

### Step 6: Test the System (Immediately after training)

**Windows:**
```batch
test.bat
```

**Command Line:**
```bash
python src/inference/query_processor.py --interactive
```

**Try questions like:**
- "What is process scheduling?"
- "Explain the TCP three-way handshake"
- "What is virtual memory?"

### Step 7: Evaluate (5-10 minutes)

**Windows:**
```batch
evaluate.bat
```

**Command Line:**
```bash
python src/evaluation/evaluate_model.py
```

Results will be in `outputs/results/`

### Step 8: Upload to GitHub

Follow: **GITHUB_INSTRUCTIONS.md**

## 🎓 For Your BTP Submission

### What to Submit:
1. **GitHub Repository URL**
2. **Evaluation Results** (from `outputs/results/`)
3. **README.md** (already perfect)
4. **Demo video** (optional but impressive)

### Timeline Suggestion:
- **Day 1**: Setup + Add materials + Process data (1-2 hours)
- **Day 2-3**: Fine-tune model (runs overnight, 8-16 hours)
- **Day 4**: Test and evaluate (1-2 hours)
- **Day 5**: Upload to GitHub + documentation (1 hour)

## 🆘 Help! Something's Wrong

### Issue: "No module named 'transformers'"
**Solution:** Run `pip install -r requirements.txt`

### Issue: "CUDA out of memory"
**Solution:** 
1. Edit `configs/training_config.yaml`
2. Change `per_device_train_batch_size: 2` (from 4)
3. Or use smaller model (1.5B instead of 7B)

### Issue: "No course materials found"
**Solution:** Add `.pptx` and `.pdf` files to `data/raw/`

### Issue: "YouTube API error"
**Solution:** Either:
- Add API key to `.env`
- Or ignore - system works without it (uses fallback)

### Issue: Training is very slow
**Solution:** 
- Use smaller model (edit `configs/training_config.yaml`)
- Or be patient - it's normal (4-12 hours)
- Check if GPU is being used

## 📊 What to Expect

### After Processing Data:
- Files in `data/processed/`
- Training dataset with 100-1000+ examples
- Vector database in `vectordb/`

### After Training:
- Model in `models/fine_tuned/`
- Training logs in `outputs/logs/`
- Takes 4-12 hours depending on GPU

### After Testing:
- Interactive Q&A system
- Answers with sources
- YouTube video suggestions
- Research paper recommendations

### After Evaluation:
- ROUGE, BLEU, F1 scores
- Performance metrics
- Detailed report in `outputs/results/`

## ✅ Quick Checklist

Before you start:
- [ ] Python 3.8+ installed
- [ ] pip working
- [ ] 20+ GB free disk space
- [ ] (Optional) NVIDIA GPU with 8+ GB VRAM

Ready to go:
- [ ] Ran `run_all.bat` or `setup.py`
- [ ] Added course materials to `data/raw/`
- [ ] Configured `.env` (optional)

## 🎯 Success Criteria

You'll know it's working when:
1. ✅ Setup creates all directories
2. ✅ Data processing generates files in `data/processed/`
3. ✅ Training runs without errors
4. ✅ Testing gives relevant answers
5. ✅ Evaluation produces metrics

## 💡 Pro Tips

1. **Start small**: Test with 1-2 slides first
2. **Use small model**: Try 1.5B before 7B
3. **Save checkpoints**: Training saves every 100 steps
4. **Monitor memory**: Watch GPU/RAM usage
5. **Read errors**: Error messages are helpful

## 🎓 Understanding the Output

When you ask a question, you get:
```
📝 ANSWER: [Detailed explanation from fine-tuned model]

📚 SOURCES: [Where the information came from]
  1. Lecture_03.pptx - Slide 5
  2. OS_Book.pdf - Section 12

🎥 RELATED VIDEOS: [Educational YouTube videos]
  1. Process Scheduling Explained - Neso Academy
     https://youtube.com/...

📄 RESEARCH PAPERS: [Academic papers from arXiv]
  1. Modern CPU Scheduling Techniques
     https://arxiv.org/...

⏱️ Processing time: 2.3s
```

## 🚀 Advanced Usage

Once basic system works:
1. Customize prompts in `configs/model_config.yaml`
2. Add more training data
3. Fine-tune further
4. Extend enrichment features
5. Create web UI (Gradio)

## 📞 Need More Help?

Read these in order:
1. **This file** - Quick start
2. **README.md** - Complete documentation
3. **QUICKSTART.md** - Detailed workflow
4. **EXPLANATION.md** - How it works
5. **FILE_INDEX.md** - Find any file

## 🎉 Ready to Start!

```bash
# The only command you need right now:
run_all.bat

# Or:
python setup.py

# Then add your course materials and continue!
```

---

**You've got this! 🚀**

The project is well-structured, well-documented, and ready to use. Just follow the steps above, and you'll have a working fine-tuned model in no time!

Questions? Check the documentation files or look at the code comments - everything is explained.

**Good luck with your BTP! 🎓✨**
