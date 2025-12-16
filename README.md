# Operating Systems & Networks - Fine-tuned Qwen3 Model with Enriched Features

## 🎯 Project Overview

This project fine-tunes a Qwen3 language model on Operating Systems and Networks course material, creating an intelligent assistant that can:
- Answer course-related questions accurately
- Suggest relevant YouTube videos for concepts
- Recommend research papers
- Provide detailed explanations with examples
- Use Retrieval-Augmented Generation (RAG) for context-aware responses

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     User Query                              │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│              Query Processing Layer                          │
│  - Intent Classification                                     │
│  - Query Enhancement                                         │
└──────────────────┬──────────────────────────────────────────┘
                   │
        ┌──────────┴──────────┐
        │                     │
        ▼                     ▼
┌──────────────┐    ┌──────────────────┐
│ Vector DB    │    │  Fine-tuned      │
│ (ChromaDB)   │    │  Qwen3 Model     │
│ Retrieval    │    │  (LoRA)          │
└──────┬───────┘    └────────┬─────────┘
       │                     │
       └──────────┬──────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│           Enrichment Layer (Parallel)                        │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐        │
│  │   YouTube    │ │    arXiv     │ │   Course     │        │
│  │   Suggester  │ │    Papers    │ │   Context    │        │
│  └──────────────┘ └──────────────┘ └──────────────┘        │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│              Response Generation                             │
│  - Main Answer                                               │
│  - YouTube Videos (with links)                               │
│  - Research Papers (with abstracts)                          │
│  - Related Concepts                                          │
└─────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
project/
├── data/                           # Training data
│   ├── raw/                        # Original course materials
│   │   ├── slides/                 # PPT/PDF slides
│   │   ├── books/                  # PDF textbooks
│   │   └── notes/                  # Additional notes
│   ├── processed/                  # Processed training data
│   │   ├── train.jsonl             # Training dataset
│   │   └── val.jsonl               # Validation dataset
│   └── evaluation/                 # Test data
│       └── endsem_questions.json   # End-semester questions
│
├── models/                         # Model storage
│   ├── base/                       # Base Qwen3 model
│   └── fine_tuned/                 # Fine-tuned model checkpoints
│
├── vectordb/                       # ChromaDB storage
│   └── course_materials/           # Embedded course content
│
├── src/                            # Source code
│   ├── data_processing/            # Data preparation scripts
│   │   ├── extract_slides.py      # Extract text from slides
│   │   ├── extract_pdfs.py        # Extract text from PDFs
│   │   ├── create_dataset.py      # Create training dataset
│   │   └── build_vectordb.py      # Build vector database
│   │
│   ├── training/                   # Training scripts
│   │   ├── fine_tune.py           # Main fine-tuning script
│   │   └── train_config.py        # Training configurations
│   │
│   ├── inference/                  # Inference scripts
│   │   ├── model_loader.py        # Load fine-tuned model
│   │   ├── rag_system.py          # RAG implementation
│   │   └── query_processor.py     # Query processing
│   │
│   ├── enrichment/                 # Feature enrichment
│   │   ├── youtube_suggester.py   # YouTube video suggestions
│   │   ├── paper_search.py        # Research paper search
│   │   └── concept_mapper.py      # Concept relationship mapping
│   │
│   ├── evaluation/                 # Evaluation scripts
│   │   ├── evaluate_model.py      # Model evaluation
│   │   └── generate_report.py     # Generate evaluation report
│   │
│   └── utils/                      # Utility functions
│       ├── config.py              # Configuration management
│       └── helpers.py             # Helper functions
│
├── configs/                        # Configuration files
│   ├── training_config.yaml       # Training parameters
│   ├── model_config.yaml          # Model specifications
│   └── api_config.yaml            # API keys (template)
│
├── outputs/                        # Output files
│   ├── logs/                      # Training logs
│   ├── results/                   # Evaluation results
│   └── responses/                 # Sample responses
│
├── notebooks/                      # Jupyter notebooks
│   ├── data_exploration.ipynb     # Explore data
│   └── demo.ipynb                 # Demo notebook
│
├── requirements.txt               # Python dependencies
├── setup.py                       # Package setup
├── .env.template                  # Environment variables template
├── .gitignore                     # Git ignore file
└── README.md                      # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>

# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate  # On Windows
# source venv/bin/activate  # On Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Copy and configure environment variables
copy .env.template .env
# Edit .env with your API keys
```

### 2. Prepare Data

```bash
# Place your course materials in data/raw/
# - Slides in data/raw/slides/
# - Books in data/raw/books/
# - Notes in data/raw/notes/

# Extract and process data
python src/data_processing/extract_slides.py
python src/data_processing/extract_pdfs.py
python src/data_processing/create_dataset.py

# Build vector database for RAG
python src/data_processing/build_vectordb.py
```

### 3. Fine-tune the Model

```bash
# Configure training parameters in configs/training_config.yaml
# Start fine-tuning (uses LoRA for efficiency)
python src/training/fine_tune.py --config configs/training_config.yaml

# Monitor training progress
tensorboard --logdir outputs/logs/
```

### 4. Run Inference

```bash
# Interactive Q&A mode
python -m src.inference.query_processor --interactive

# Answer specific question
python -m src.inference.query_processor --question "Explain process synchronization"

# Evaluate on endsem questions
python src/evaluation/evaluate_model.py --test-file data/evaluation/endsem_questions.json
```

## 🎓 Features

### 1. **Fine-tuned Qwen3 Model**
- Uses Parameter-Efficient Fine-Tuning (PEFT) with LoRA
- Supports models from 0.6B to 32B parameters
- Optimized for educational Q&A
- Context-aware responses

### 2. **Retrieval-Augmented Generation (RAG)**
- ChromaDB vector database
- Semantic search over course materials
- Combines retrieval with generation
- Reduces hallucination

### 3. **YouTube Video Suggestions**
- Automatically suggests relevant educational videos
- Uses YouTube Data API v3
- Filters by quality metrics (views, ratings)
- Provides video titles, channels, and links

### 4. **Research Paper Recommendations**
- Searches arXiv for related papers
- Provides abstracts and links
- Categorizes by relevance
- Includes citation information

### 5. **Concept Mapping**
- Identifies related concepts
- Shows prerequisite knowledge
- Suggests learning paths
- Visual concept graphs (optional)

### 6. **Advanced Question Answering**
- Multiple answer formats (brief, detailed, example-based)
- Step-by-step explanations
- Code examples for OS/Network concepts
- Diagram descriptions

## 📊 Model Specifications

### Supported Qwen3 Models

| Model Size | Parameters | VRAM Required | Training Time (est.) | Recommended Use |
|-----------|-----------|---------------|---------------------|----------------|
| Qwen3-0.6B | 600M | ~4 GB | 2-4 hours | Quick prototyping |
| Qwen3-1.5B | 1.5B | ~8 GB | 4-6 hours | Balanced performance |
| Qwen3-3B | 3B | ~12 GB | 6-10 hours | Good quality |
| Qwen3-7B | 7B | ~20 GB | 12-20 hours | High quality (Recommended) |
| Qwen3-14B | 14B | ~40 GB | 24-36 hours | Very high quality |
| Qwen3-32B | 32B | ~80 GB | 48+ hours | Maximum quality |

### Fine-Tuning Configuration

```yaml
# Default configuration (configs/training_config.yaml)
model_name: "Qwen/Qwen2.5-7B-Instruct"  # Qwen3 equivalent
lora_config:
  r: 16                    # LoRA rank
  lora_alpha: 32           # LoRA alpha
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
  lora_dropout: 0.1
  bias: "none"

training:
  num_epochs: 3
  batch_size: 4
  gradient_accumulation_steps: 4
  learning_rate: 2e-4
  warmup_steps: 100
  max_seq_length: 2048
  
optimization:
  use_8bit: true           # 8-bit quantization
  use_gradient_checkpointing: true
  fp16: true               # Mixed precision training
```

## 🧪 Evaluation

### Metrics Tracked

1. **Accuracy Metrics**
   - Exact match score
   - F1 score (token overlap)
   - BLEU score
   - ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L)

2. **Quality Metrics**
   - Relevance score
   - Coherence score
   - Completeness score
   - Factual accuracy

3. **Enrichment Metrics**
   - YouTube suggestion relevance
   - Paper recommendation quality
   - Response time
   - User satisfaction (manual evaluation)

### Sample Evaluation Report

```
=== Evaluation Results ===
Test Set: Endsem Questions (50 questions)

Core Metrics:
- Exact Match: 34.0%
- F1 Score: 78.5%
- ROUGE-L: 0.742

Enrichment Features:
- YouTube Suggestions: 48/50 questions (96%)
- Avg. Videos per Query: 3.2
- Research Papers: 42/50 questions (84%)
- Avg. Papers per Query: 2.5

Performance:
- Avg. Response Time: 2.3s
- RAG Retrieval Accuracy: 89%
- Context Relevance: 92%
```

## 🔧 Configuration

### API Keys Required

1. **YouTube Data API** (Optional, for video suggestions)
   - Get key from: https://console.cloud.google.com/
   - Free tier: 10,000 units/day

2. **OpenAI API** (Optional, for embeddings)
   - Alternative: Use open-source embeddings (sentence-transformers)

### Environment Variables

```bash
# .env file
YOUTUBE_API_KEY=your_youtube_api_key_here
OPENAI_API_KEY=your_openai_key_here  # Optional
HF_TOKEN=your_huggingface_token_here  # For model download

# Paths
DATA_DIR=./data
MODEL_DIR=./models
VECTORDB_DIR=./vectordb
OUTPUT_DIR=./outputs
```

## 📝 Usage Examples

### Example 1: Basic Question

```python
from src.inference.query_processor import QueryProcessor

processor = QueryProcessor()
result = processor.answer("What is a deadlock?")

print(result['answer'])
# Output: "A deadlock is a situation where two or more processes..."

print(result['youtube_videos'])
# Output: [
#   {
#     'title': 'Deadlocks in Operating Systems',
#     'channel': 'Neso Academy',
#     'url': 'https://youtube.com/watch?v=...'
#   },
#   ...
# ]
```

### Example 2: Detailed Explanation

```python
result = processor.answer(
    "Explain the Banker's Algorithm with an example",
    detail_level="comprehensive"
)

print(result['answer'])
print(result['code_examples'])
print(result['research_papers'])
```

### Example 3: Batch Evaluation

```python
from src.evaluation.evaluate_model import evaluate_on_file

results = evaluate_on_file('data/evaluation/endsem_questions.json')
print(results.summary())
```

## 🎯 Grading Features Implementation

### Feature Checklist ✅

- [x] **Fine-tuned Model** - Qwen3 with LoRA fine-tuning
- [x] **RAG System** - ChromaDB vector database integration
- [x] **YouTube Integration** - Automatic video suggestions with relevance ranking
- [x] **Research Papers** - arXiv API integration with smart filtering
- [x] **Question Solving** - Multi-format answer generation
- [x] **Concept Mapping** - Related topics and prerequisites
- [x] **Code Examples** - Auto-generated code snippets for OS/Network concepts
- [x] **Interactive Mode** - CLI and programmatic interfaces
- [x] **Evaluation Framework** - Comprehensive metrics and reporting
- [x] **Extensibility** - Plugin architecture for additional features

### Advanced Features 🌟

1. **Multi-Modal Responses**
   - Text explanations
   - Code snippets
   - ASCII diagrams
   - Pseudocode

2. **Smart Context Management**
   - Conversation history
   - Follow-up questions
   - Context-aware clarifications

3. **Performance Optimization**
   - Model quantization (8-bit, 4-bit)
   - Caching mechanisms
   - Batch processing

4. **Extensibility**
   - Plugin system for new features
   - Custom enrichment modules
   - Configurable pipelines

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Use smaller model or reduce batch size
   python src/training/fine_tune.py --model Qwen3-1.5B --batch-size 2
   ```

2. **YouTube API Quota Exceeded**
   ```bash
   # The system gracefully degrades; video suggestions will be skipped
   # Or use cached results
   ```

3. **Slow Inference**
   ```bash
   # Enable quantization
   python -m src.inference.query_processor --quantize 8bit
   ```

## 📚 Course Material Format

### Expected Input Format

**Slides (PPT/PDF):**
- Place in `data/raw/slides/`
- Supported formats: .pptx, .pdf
- Auto-extracted with text and structure preservation

**Books (PDF):**
- Place in `data/raw/books/`
- Automatically chunked into sections
- Preserves chapter/section hierarchy

**Custom Notes:**
- Place in `data/raw/notes/`
- Markdown or text format
- Tagged with topics

### Training Data Format

```json
{
  "instruction": "Explain process scheduling in operating systems",
  "input": "",
  "output": "Process scheduling is the activity of the process manager...",
  "context": "From: Operating Systems Concepts, Chapter 5",
  "topics": ["process scheduling", "CPU scheduling", "operating systems"]
}
```

## 🚀 Future Enhancements

- [ ] Web interface (Gradio/Streamlit)
- [ ] Voice input/output
- [ ] Diagram generation
- [ ] Flashcard creation
- [ ] Quiz generation
- [ ] Multi-language support
- [ ] Mobile app integration
- [ ] Collaborative learning features

## 📄 License

MIT License - Feel free to use and modify for educational purposes.

## 🙏 Acknowledgments

- **Qwen Team** - For the excellent base model
- **Hugging Face** - For transformers and PEFT libraries
- **ChromaDB** - For vector database
- **YouTube & arXiv** - For API access

## 📧 Contact

For questions or issues, please open an issue on GitHub.

---

**Note**: This project is designed for educational purposes. Ensure you have proper permissions for course materials and comply with API usage terms.
