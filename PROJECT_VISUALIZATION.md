# 📊 Complete Project Visualization

## 🎯 System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         USER INTERACTION                             │
│                   "What is process scheduling?"                      │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     QUERY PROCESSOR                                  │
│              (src/inference/query_processor.py)                      │
└─────┬──────────────────────┬─────────────────────┬──────────────────┘
      │                      │                     │
      │                      │                     │
      ▼                      ▼                     ▼
┌──────────────┐    ┌─────────────────┐    ┌──────────────────┐
│ RAG SYSTEM   │    │ FINE-TUNED      │    │ ENRICHMENT       │
│              │    │ QWEN3 MODEL     │    │ FEATURES         │
│ ChromaDB     │    │                 │    │                  │
│ Vector DB    │    │ LoRA Adapters   │    │ • YouTube API    │
│              │    │ 7B Parameters   │    │ • arXiv Search   │
│              │    │                 │    │ • Concept Map    │
└──────┬───────┘    └────────┬────────┘    └────────┬─────────┘
       │                     │                      │
       │                     │                      │
       └─────────────────────┼──────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       FORMATTED RESPONSE                             │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ 📝 ANSWER: Process scheduling is the method by which...       │ │
│  │                                                                │ │
│  │ 📚 SOURCES:                                                   │ │
│  │   • Lecture_05.pptx - Slide 12                               │ │
│  │   • Operating_Systems.pdf - Chapter 5                        │ │
│  │                                                                │ │
│  │ 🎥 VIDEOS:                                                    │ │
│  │   • CPU Scheduling - Neso Academy                            │ │
│  │   • Process Scheduling Explained - Gate Smashers             │ │
│  │                                                                │ │
│  │ 📄 PAPERS:                                                    │ │
│  │   • Modern Scheduling Algorithms (arXiv)                     │ │
│  │   • Real-time Scheduling Techniques (IEEE)                   │ │
│  └────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure (Complete)

```
btp_selection/
│
├── 📄 Documentation (Start Here!)
│   ├── README.md ⭐ (Main documentation)
│   ├── GET_STARTED.md ⭐ (Fastest start)
│   ├── QUICKSTART.md (Step-by-step guide)
│   ├── EXPLANATION.md (Technical deep-dive)
│   ├── PROJECT_SUMMARY.md (Complete overview)
│   ├── FILE_INDEX.md (Navigation guide)
│   └── GITHUB_INSTRUCTIONS.md (Upload guide)
│
├── ⚙️ Configuration
│   ├── .env.template (API keys template)
│   ├── .env (Your API keys - created)
│   ├── .gitignore (Git ignore rules)
│   ├── requirements.txt (Python dependencies)
│   └── setup.py (Initialization script)
│
├── 🎛️ Config Files
│   └── configs/
│       ├── training_config.yaml (Training parameters)
│       ├── model_config.yaml (Model settings)
│       └── api_config.yaml (API configurations)
│
├── 🚀 Quick Start Scripts (Windows)
│   ├── run_all.bat ⭐ (Complete setup)
│   ├── process_data.bat (Process materials)
│   ├── train.bat (Fine-tune model)
│   ├── test.bat (Interactive testing)
│   └── evaluate.bat (Run evaluation)
│
├── 💻 Source Code
│   └── src/
│       ├── utils/ (Configuration & helpers)
│       │   ├── config.py
│       │   └── helpers.py
│       │
│       ├── data_processing/ (Data pipeline)
│       │   ├── extract_slides.py
│       │   ├── extract_pdfs.py
│       │   ├── create_dataset.py
│       │   └── build_vectordb.py
│       │
│       ├── training/ (Model training)
│       │   └── fine_tune.py
│       │
│       ├── inference/ (Using the model)
│       │   ├── model_loader.py
│       │   ├── rag_system.py
│       │   └── query_processor.py ⭐ (Main interface)
│       │
│       ├── enrichment/ (Extra features)
│       │   ├── youtube_suggester.py
│       │   ├── paper_search.py
│       │   └── concept_mapper.py
│       │
│       └── evaluation/ (Testing)
│           └── evaluate_model.py
│
├── 📊 Data Directories
│   └── data/
│       ├── raw/ (YOU ADD FILES HERE!)
│       │   ├── slides/ ← Add .pptx files
│       │   ├── books/ ← Add .pdf files
│       │   └── notes/ ← Add .txt/.md files
│       │
│       ├── processed/ (Auto-generated)
│       │   ├── slides/ (Extracted)
│       │   ├── books/ (Extracted)
│       │   ├── train.jsonl (Training data)
│       │   └── val.jsonl (Validation data)
│       │
│       └── evaluation/ (Test questions)
│           └── endsem_questions.json ✓ (Provided)
│
├── 🤖 Model & Database
│   ├── models/
│   │   ├── base/ (Downloaded model)
│   │   └── fine_tuned/ (Your trained model)
│   │
│   └── vectordb/
│       └── course_materials/ (ChromaDB)
│
└── 📈 Outputs
    └── outputs/
        ├── logs/ (Training logs)
        ├── results/ (Evaluation results)
        └── responses/ (Sample outputs)
```

## 🔄 Complete Workflow Visualization

```
┌─────────────────────────────────────────────────────────────────────┐
│                          PHASE 1: SETUP                              │
│                                                                       │
│  1. Run: run_all.bat                                                │
│     ├─→ Creates virtual environment                                 │
│     ├─→ Installs dependencies                                       │
│     ├─→ Creates directories                                         │
│     └─→ Sets up .env file                                          │
│                                                                       │
│  2. Edit .env with API keys (optional)                              │
└─────────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   PHASE 2: DATA PREPARATION                          │
│                                                                       │
│  1. Add your materials:                                              │
│     • Copy slides → data/raw/slides/                                │
│     • Copy books → data/raw/books/                                  │
│                                                                       │
│  2. Run: process_data.bat                                           │
│     ├─→ extract_slides.py    → JSON files                          │
│     ├─→ extract_pdfs.py      → JSON files                          │
│     ├─→ create_dataset.py    → train.jsonl, val.jsonl              │
│     └─→ build_vectordb.py    → ChromaDB                            │
└─────────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      PHASE 3: TRAINING                               │
│                                                                       │
│  1. (Optional) Edit configs/training_config.yaml                    │
│     • Choose model size (1.5B, 3B, 7B)                             │
│     • Adjust batch size for your GPU                                │
│                                                                       │
│  2. Run: train.bat                                                  │
│     ├─→ Downloads base Qwen3 model                                 │
│     ├─→ Applies LoRA adapters                                      │
│     ├─→ Trains on your data (4-12 hours)                          │
│     └─→ Saves to models/fine_tuned/                                │
│                                                                       │
│  3. Monitor: tensorboard --logdir models/fine_tuned/logs           │
└─────────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      PHASE 4: TESTING                                │
│                                                                       │
│  1. Run: test.bat (interactive mode)                                │
│                                                                       │
│  2. Ask questions:                                                   │
│     • "What is virtual memory?"                                     │
│     • "Explain TCP congestion control"                              │
│     • "What is process synchronization?"                            │
│                                                                       │
│  3. Review responses:                                                │
│     ✓ Answer from fine-tuned model                                 │
│     ✓ Source citations                                              │
│     ✓ YouTube video suggestions                                     │
│     ✓ Research paper recommendations                                │
└─────────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     PHASE 5: EVALUATION                              │
│                                                                       │
│  1. Run: evaluate.bat                                               │
│     ├─→ Tests on endsem_questions.json                             │
│     ├─→ Calculates ROUGE, BLEU, F1 scores                          │
│     ├─→ Measures enrichment coverage                                │
│     └─→ Saves report to outputs/results/                           │
│                                                                       │
│  2. Review results:                                                  │
│     • outputs/results/evaluation_TIMESTAMP.json                     │
│     • outputs/results/evaluation_TIMESTAMP_summary.txt              │
└─────────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   PHASE 6: GITHUB UPLOAD                             │
│                                                                       │
│  1. Follow GITHUB_INSTRUCTIONS.md                                   │
│  2. git init → git add → git commit → git push                     │
│  3. Submit repository URL for grading                               │
└─────────────────────────────────────────────────────────────────────┘
```

## 🎯 Feature Map

```
┌─────────────────────────────────────────────────────────────────────┐
│                     CORE CAPABILITIES                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  1. FINE-TUNED MODEL (src/training/)                                │
│     ├── LoRA: Parameter-efficient fine-tuning                       │
│     ├── Quantization: 8-bit for efficiency                          │
│     ├── Multi-size: 0.6B to 32B parameters                          │
│     └── Domain-specific: Adapted to OS/Networks                     │
│                                                                       │
│  2. RAG SYSTEM (src/inference/rag_system.py)                        │
│     ├── ChromaDB: Vector database                                   │
│     ├── Embeddings: Semantic search                                 │
│     ├── Context: Relevant course material                           │
│     └── Citations: Source attribution                               │
│                                                                       │
│  3. DATA PROCESSING (src/data_processing/)                          │
│     ├── Multi-format: PPT, PDF, TXT                                │
│     ├── Chunking: Optimal size splits                               │
│     ├── Cleaning: Text normalization                                │
│     └── Dataset: Instruction-response pairs                         │
│                                                                       │
│  4. EVALUATION (src/evaluation/)                                    │
│     ├── ROUGE: Text overlap metrics                                 │
│     ├── BLEU: Quality measurement                                   │
│     ├── F1: Precision-recall balance                                │
│     └── Custom: Enrichment coverage                                 │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                   ENRICHMENT FEATURES                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  5. YOUTUBE INTEGRATION (src/enrichment/youtube_suggester.py)       │
│     ├── API: YouTube Data API v3                                   │
│     ├── Filtering: Quality metrics                                  │
│     ├── Ranking: Relevance scoring                                  │
│     └── Fallback: Curated suggestions                               │
│                                                                       │
│  6. RESEARCH PAPERS (src/enrichment/paper_search.py)                │
│     ├── arXiv: Academic paper search                                │
│     ├── Categories: cs.OS, cs.NI, cs.DC                            │
│     ├── Abstracts: Paper summaries                                  │
│     └── Links: Direct access                                        │
│                                                                       │
│  7. CONCEPT MAPPING (src/enrichment/concept_mapper.py)              │
│     ├── Relations: Connected topics                                 │
│     ├── Prerequisites: Learning path                                │
│     ├── Subtopics: Detailed breakdown                               │
│     └── Categories: Topic organization                              │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

## 🏆 Grading Criteria Coverage

```
✅ FINE-TUNING
   ├── ✓ Modern technique (LoRA)
   ├── ✓ Efficient (8-bit quantization)
   ├── ✓ Configurable (YAML configs)
   └── ✓ Scalable (0.6B-32B models)

✅ COURSE MATERIALS
   ├── ✓ Multi-format support (PPT, PDF)
   ├── ✓ Automated processing
   ├── ✓ Quality extraction
   └── ✓ Dataset generation

✅ VECTOR DATABASE
   ├── ✓ ChromaDB implementation
   ├── ✓ Semantic search
   ├── ✓ RAG integration
   └── ✓ Source tracking

✅ ENRICHMENT FEATURES
   ├── ✓ YouTube suggestions
   ├── ✓ Research papers
   ├── ✓ Concept mapping
   └── ✓ Multi-source information

✅ EVALUATION
   ├── ✓ Multiple metrics
   ├── ✓ Comprehensive reporting
   ├── ✓ Test dataset
   └── ✓ Performance tracking

✅ SOFTWARE ENGINEERING
   ├── ✓ Modular architecture
   ├── ✓ Configuration management
   ├── ✓ Documentation
   ├── ✓ Error handling
   └── ✓ Version control ready

✅ EXTENSIBILITY
   ├── ✓ Plugin architecture
   ├── ✓ API integration
   ├── ✓ Configurable pipelines
   └── ✓ Future-proof design
```

## 📊 Technology Stack

```
┌─────────────────────────────────────────────┐
│         MACHINE LEARNING STACK              │
├─────────────────────────────────────────────┤
│ • PyTorch 2.0+                              │
│ • Transformers (Hugging Face)               │
│ • PEFT (LoRA implementation)                │
│ • bitsandbytes (Quantization)               │
│ • sentence-transformers (Embeddings)        │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│           DATABASE & SEARCH                  │
├─────────────────────────────────────────────┤
│ • ChromaDB (Vector database)                │
│ • FAISS (Alternative search)                │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│         API & WEB SERVICES                   │
├─────────────────────────────────────────────┤
│ • YouTube Data API v3                       │
│ • arXiv API (Research papers)               │
│ • Google APIs (OAuth)                       │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│        DATA PROCESSING                       │
├─────────────────────────────────────────────┤
│ • python-pptx (PowerPoint)                  │
│ • PyPDF2, pdfplumber (PDF)                  │
│ • pandas, numpy (Data manipulation)         │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│         EVALUATION & METRICS                 │
├─────────────────────────────────────────────┤
│ • rouge-score (ROUGE metrics)               │
│ • nltk (BLEU, NLP)                          │
│ • scikit-learn (ML metrics)                 │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│      CONFIGURATION & UTILITIES               │
├─────────────────────────────────────────────┤
│ • PyYAML (Config files)                     │
│ • python-dotenv (Environment)               │
│ • rich (CLI formatting)                     │
│ • tqdm (Progress bars)                      │
└─────────────────────────────────────────────┘
```

## 🎯 Quick Commands Reference

```bash
# SETUP
run_all.bat              # Complete setup (Windows)
python setup.py          # Manual setup

# DATA PROCESSING
process_data.bat         # All-in-one (Windows)
python src/data_processing/extract_slides.py
python src/data_processing/extract_pdfs.py
python src/data_processing/create_dataset.py
python src/data_processing/build_vectordb.py

# TRAINING
train.bat                # Train model (Windows)
python src/training/fine_tune.py

# TESTING
test.bat                 # Interactive mode (Windows)
python src/inference/query_processor.py --interactive
python src/inference/query_processor.py --question "Your Q"

# EVALUATION
evaluate.bat             # Run evaluation (Windows)
python src/evaluation/evaluate_model.py

# MONITORING
tensorboard --logdir models/fine_tuned/logs
```

---

**This visualization shows the complete system at a glance.**  
**Start with GET_STARTED.md for step-by-step instructions! 🚀**
