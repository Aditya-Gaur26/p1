# Multimedia Features - Quick Reference Card

## 🚀 Quick Commands

### Image Extraction
```bash
# Extract from all slides in directory
python src/data_processing/extract_images.py --input-dir data/raw/slides

# Extract from single file
python src/data_processing/extract_images.py --file lecture1.pptx

# Extract without OCR
python src/data_processing/extract_images.py --input-dir data/raw/slides --no-ocr
```

### Test Suite
```bash
# Windows
test_multimedia.bat

# Linux/macOS
python test_multimedia.py
```

---

## 📦 Installation

### Python Packages
```bash
pip install pytesseract opencv-python Pillow transformers torch
```

### System Dependencies
```bash
# Windows
choco install tesseract

# Linux
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract
```

### Optional Tools
```bash
# Mermaid CLI (for rendering)
npm install -g @mermaid-js/mermaid-cli

# PlantUML (download from plantuml.com)
```

---

## 💻 Code Examples

### 1. Extract Images
```python
from src.data_processing.extract_images import ImageExtractor
from pathlib import Path

extractor = ImageExtractor(save_images=True, use_ocr=True)
images = extractor.extract_images_from_presentation(Path("lecture.pptx"))
```

### 2. OCR Text Extraction
```python
from src.data_processing.ocr_processor import OCRProcessor
from PIL import Image

ocr = OCRProcessor(lang='eng')
image = Image.open("diagram.png")
result = ocr.extract_from_diagram(image)

print(f"Text: {result['text']}")
print(f"Type: {result['diagram_type']}")
```

### 3. Image Captioning
```python
from src.inference.vision_language import ImageCaptioner
from PIL import Image

captioner = ImageCaptioner()
image = Image.open("diagram.png")
caption = captioner.generate_caption(image)

print(f"Caption: {caption}")
```

### 4. Figure Generation
```python
from src.inference.figure_generator import FigureGenerator

generator = FigureGenerator()

# ASCII art (no dependencies)
ascii_result = generator.generate_ascii_art("TCP handshake")
print(ascii_result['art'])

# Mermaid code
mermaid_result = generator.generate_mermaid_diagram("TCP handshake", "sequence")
print(mermaid_result['code'])
```

---

## 🎯 Key Features

| Feature | Module | Dependencies |
|---------|--------|--------------|
| **Image Extraction** | `extract_images.py` | python-pptx, PIL |
| **OCR** | `ocr_processor.py` | pytesseract, opencv |
| **Captioning** | `vision_language.py` | transformers (BLIP) |
| **Classification** | `vision_language.py` | transformers (CLIP) |
| **Figure Generation** | `figure_generator.py` | None (code only) |

---

## 📊 Diagram Types Supported

### OCR Classification
1. Flowchart
2. State Diagram
3. Network Diagram
4. Sequence Diagram
5. Architecture Diagram
6. Graph/Chart
7. Timeline
8. Memory Diagram
9. Protocol Diagram

### Figure Generation
1. Sequence Diagrams (TCP handshake, etc.)
2. Flowcharts (Process flows)
3. State Diagrams (Process states)
4. Network Topologies
5. Architecture Diagrams
6. OSI Model
7. Deadlock Visualization

---

## 🔍 Built-in Templates

Quick generate common diagrams:

```python
from src.inference.figure_generator import FigureGenerator

generator = FigureGenerator()

# Automatically recognizes and generates:
templates = [
    "TCP three-way handshake",       # → Sequence diagram
    "Process state diagram",         # → State machine
    "Deadlock circular wait",        # → Graph diagram
    "OSI model layers"               # → Architecture
]

for template in templates:
    result = generator.generate_ascii_art(template)
    print(result['art'])
```

---

## 📁 Output Locations

```
data/
├── raw/
│   └── slides/                  ← Add your .pptx files here
└── processed/
    └── images/
        ├── *.png                ← Extracted images
        └── *_images.json        ← Metadata
```

---

## ⚙️ Configuration Flags

### ImageExtractor
```python
ImageExtractor(
    save_images=True,        # Save to disk
    use_ocr=True,           # Enable OCR
    use_captioning=True,    # Enable BLIP
    output_dir=Path("...")  # Custom output
)
```

### OCRProcessor
```python
OCRProcessor(
    lang='eng',             # Language (eng, fra, deu, etc.)
    min_confidence=50       # Confidence threshold
)
```

### ImageCaptioner
```python
ImageCaptioner(
    model_name="Salesforce/blip-image-captioning-base"  # Model
)
```

---

## 🐛 Troubleshooting

### OCR not working?
```bash
# Check Tesseract installation
tesseract --version

# Add to PATH if needed (Windows)
setx PATH "%PATH%;C:\Program Files\Tesseract-OCR"
```

### Models not downloading?
```bash
# Check internet connection
# Clear cache if corrupted
rm -rf ~/.cache/huggingface/

# Re-download
python -c "from transformers import BlipProcessor; BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base')"
```

### Out of memory?
```python
# Use CPU instead of GPU
import torch
torch.cuda.is_available = lambda: False

# Or reduce batch size
captioner.batch_caption(images[:10])  # Process 10 at a time
```

---

## 📖 Documentation

- **Complete Guide**: `MULTIMEDIA_GUIDE.md` (24 pages)
- **Implementation Summary**: `MULTIMEDIA_IMPLEMENTATION.md`
- **Quick Reference**: `MULTIMEDIA_QUICKREF.md` (this file)
- **Test Suite**: `test_multimedia.py`

---

## 🎓 Common Workflows

### Workflow 1: Add Training Materials
```bash
1. mkdir data\raw\slides
2. copy lectures\*.pptx data\raw\slides\
3. python src\data_processing\extract_images.py --input-dir data\raw\slides
4. python src\data_processing\extract_slides.py
5. python src\data_processing\build_vectordb.py
```

### Workflow 2: Process Existing Images
```python
from src.data_processing.ocr_processor import OCRProcessor
from src.inference.vision_language import VisionLanguageProcessor
from PIL import Image
from pathlib import Path

ocr = OCRProcessor()
vlp = VisionLanguageProcessor()

for img_path in Path("data/processed/images").glob("*.png"):
    image = Image.open(img_path)
    
    # OCR
    ocr_result = ocr.extract_from_diagram(image)
    
    # Caption
    vlp_result = vlp.process_image(image)
    
    print(f"{img_path.name}:")
    print(f"  Text: {ocr_result['text'][:50]}")
    print(f"  Caption: {vlp_result['caption']}")
    print(f"  Type: {ocr_result['diagram_type']}")
```

### Workflow 3: Generate Diagrams in Responses
```python
from src.inference.figure_generator import generate_diagram_for_answer

question = "Explain the TCP three-way handshake"
answer = "TCP uses SYN, SYN-ACK, and ACK packets..."

# Auto-generate diagram
diagram = generate_diagram_for_answer(question, answer)

if diagram:
    # Include ASCII art in response
    for d in diagram['diagrams']:
        if d['type'] == 'ascii':
            final_answer = f"{answer}\n\n{d['art']}"
```

---

## 💡 Pro Tips

1. **Batch Processing**: Process multiple files at once for efficiency
   ```bash
   python src/data_processing/extract_images.py --input-dir data/raw/slides
   ```

2. **OCR Quality**: Preprocess images for better accuracy
   ```python
   preprocessed = ocr.preprocess_image(image)
   result = ocr.extract_text(preprocessed)
   ```

3. **Caption Quality**: Use conditional generation for diagrams
   ```python
   caption = captioner.generate_conditional_caption(image, "a diagram showing")
   ```

4. **Figure Format**: ASCII art needs no dependencies, perfect for quick responses
   ```python
   ascii_result = generator.generate_ascii_art(description)
   ```

5. **Memory Management**: Process in batches to avoid OOM
   ```python
   for batch in chunks(images, 10):
       results = captioner.batch_caption(batch)
   ```

---

## 🔗 Quick Links

- **GitHub Issues**: Report bugs or request features
- **Hugging Face Models**: 
  - [BLIP](https://huggingface.co/Salesforce/blip-image-captioning-base)
  - [CLIP](https://huggingface.co/openai/clip-vit-base-patch32)
- **Tesseract**: [GitHub](https://github.com/tesseract-ocr/tesseract)
- **Mermaid**: [Documentation](https://mermaid.js.org/)

---

## ✅ Checklist

Before using multimedia features:

- [ ] Install Python packages (`pip install ...`)
- [ ] Install Tesseract OCR (system dependency)
- [ ] Create `data/raw/slides/` directory
- [ ] Add training materials (.pptx files)
- [ ] Run test suite (`test_multimedia.bat`)
- [ ] Check all tests pass
- [ ] Extract images from slides
- [ ] Review extracted images
- [ ] Rebuild vector database with image content

---

*For detailed documentation, see MULTIMEDIA_GUIDE.md*
