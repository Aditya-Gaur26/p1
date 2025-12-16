"""
Test Multimedia Features
Tests image extraction, OCR, vision-language, and figure generation
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from PIL import Image, ImageDraw, ImageFont
import json


def test_image_extraction():
    """Test image extraction from PowerPoint"""
    print("\n" + "=" * 70)
    print("TEST 1: Image Extraction".center(70))
    print("=" * 70)
    
    try:
        from src.data_processing.extract_images import ImageExtractor
        
        # Check if test slides exist
        slides_dir = Path("data/raw/slides")
        if not slides_dir.exists():
            print(f"⚠️  Slides directory not found: {slides_dir}")
            print("   Creating directory and skipping test...")
            slides_dir.mkdir(parents=True, exist_ok=True)
            print("   ✓ Add your .pptx files to data/raw/slides/ to test extraction")
            return False
        
        pptx_files = list(slides_dir.glob("*.pptx")) + list(slides_dir.glob("*.ppt"))
        
        if not pptx_files:
            print(f"⚠️  No PowerPoint files found in {slides_dir}")
            print("   Add .pptx files to test extraction")
            return False
        
        # Test with first file
        test_file = pptx_files[0]
        print(f"\n📄 Testing with: {test_file.name}")
        
        extractor = ImageExtractor(save_images=True)
        images = extractor.extract_images_from_presentation(test_file)
        
        print(f"\n✅ Extracted {len(images)} images")
        
        if images:
            # Show sample
            sample = images[0]
            print(f"\n📊 Sample image:")
            print(f"   Slide: {sample.get('slide_number')}")
            print(f"   Format: {sample.get('image_format')}")
            print(f"   Size: {sample.get('image_size')}")
            print(f"   Saved: {sample.get('saved_path')}")
            
            # Get statistics
            stats = extractor.get_statistics(images)
            print(f"\n📈 Statistics:")
            for key, value in stats.items():
                print(f"   {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ocr():
    """Test OCR text extraction"""
    print("\n" + "=" * 70)
    print("TEST 2: OCR Text Extraction".center(70))
    print("=" * 70)
    
    try:
        from src.data_processing.ocr_processor import OCRProcessor, DiagramClassifier
        
        # Create test image with text
        print("\n📷 Creating test image with text...")
        img = Image.new('RGB', (400, 200), color='white')
        draw = ImageDraw.Draw(img)
        
        # Draw text
        text = "TCP/IP Protocol Stack"
        draw.text((50, 50), text, fill='black')
        draw.text((50, 100), "Application Layer", fill='black')
        draw.text((50, 130), "Transport Layer", fill='black')
        
        # Save test image
        test_path = Path("data/processed/images/test_ocr.png")
        test_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(test_path)
        print(f"   ✓ Saved to: {test_path}")
        
        # Test OCR
        print("\n🔍 Running OCR...")
        ocr = OCRProcessor()
        
        if not ocr.tesseract_available:
            print("⚠️  Tesseract not installed")
            print("   Install: choco install tesseract (Windows)")
            print("   Install: sudo apt-get install tesseract-ocr (Linux)")
            print("   Install: brew install tesseract (macOS)")
            return False
        
        result = ocr.extract_text(img)
        
        print(f"\n✅ OCR Results:")
        print(f"   Text: {result['text'][:100]}")
        print(f"   Confidence: {result['confidence']:.2f}%")
        print(f"   Word count: {result['word_count']}")
        
        # Test diagram classification
        print("\n🎯 Classifying diagram type...")
        classifier = DiagramClassifier()
        diagram_type, keywords = classifier.classify(img)
        
        print(f"   Type: {diagram_type}")
        print(f"   Keywords: {keywords[:5]}")
        
        # Test preprocessing
        print("\n🔧 Testing image preprocessing...")
        preprocessed = ocr.preprocess_image(img)
        print(f"   Original size: {img.size}")
        print(f"   Preprocessed size: {preprocessed.size}")
        print(f"   Preprocessed mode: {preprocessed.mode}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_vision_language():
    """Test vision-language models"""
    print("\n" + "=" * 70)
    print("TEST 3: Vision-Language Integration".center(70))
    print("=" * 70)
    
    try:
        from src.inference.vision_language import VisionLanguageProcessor, BLIP_AVAILABLE, CLIP_AVAILABLE
        
        print(f"\n📦 Availability Check:")
        print(f"   BLIP (captioning): {BLIP_AVAILABLE}")
        print(f"   CLIP (classification): {CLIP_AVAILABLE}")
        
        if not BLIP_AVAILABLE and not CLIP_AVAILABLE:
            print("\n⚠️  Vision models not installed")
            print("   Install: pip install transformers torch")
            print("   Note: This will download ~2GB of models")
            return False
        
        # Create test diagram
        print("\n📷 Creating test diagram...")
        img = Image.new('RGB', (400, 300), color='white')
        draw = ImageDraw.Draw(img)
        
        # Draw flowchart
        draw.rectangle([(100, 50), (300, 100)], outline='black', width=2)
        draw.text((160, 65), "Start", fill='black')
        
        draw.rectangle([(100, 130), (300, 180)], outline='black', width=2)
        draw.text((140, 145), "Process", fill='black')
        
        draw.rectangle([(100, 210), (300, 260)], outline='black', width=2)
        draw.text((160, 225), "End", fill='black')
        
        # Arrows
        draw.line([(200, 100), (200, 130)], fill='black', width=2)
        draw.line([(200, 180), (200, 210)], fill='black', width=2)
        
        # Process image
        print("\n🔍 Processing with vision models...")
        processor = VisionLanguageProcessor()
        result = processor.process_image(img)
        
        print(f"\n✅ Results:")
        print(f"   Caption: {result.get('caption', 'N/A')}")
        print(f"   Diagram description: {result.get('diagram_description', 'N/A')}")
        print(f"   Is educational: {result.get('is_educational', 'N/A')}")
        
        if result.get('diagram_type'):
            print(f"\n   Diagram type classification:")
            sorted_types = sorted(result['diagram_type'].items(), 
                                key=lambda x: x[1], reverse=True)[:3]
            for dtype, score in sorted_types:
                print(f"      {dtype}: {score:.3f}")
        
        # Test text alternative
        alt_text = processor.extract_text_alternative(img)
        print(f"\n   Text alternative: {alt_text}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_figure_generation():
    """Test figure generation"""
    print("\n" + "=" * 70)
    print("TEST 4: Figure Generation".center(70))
    print("=" * 70)
    
    try:
        from src.inference.figure_generator import FigureGenerator, generate_diagram_for_answer
        
        generator = FigureGenerator()
        
        print(f"\n🛠️  Tool Availability:")
        print(f"   Mermaid CLI: {generator.mermaid_available}")
        print(f"   PlantUML: {generator.plantuml_available}")
        print(f"   Graphviz: {generator.graphviz_available}")
        
        # Test ASCII art (no dependencies)
        print("\n🎨 Test 1: ASCII Art Generation")
        ascii_result = generator.generate_ascii_art("TCP three-way handshake")
        print(ascii_result['art'])
        
        # Test Mermaid code generation
        print("\n🎨 Test 2: Mermaid Code Generation")
        mermaid_result = generator.generate_mermaid_diagram(
            "TCP three-way handshake", 
            "sequence"
        )
        print("Generated Mermaid code:")
        print(mermaid_result['code'][:300] + "...")
        print(f"Can render to image: {mermaid_result['can_render']}")
        
        # Test diagram detection
        print("\n🎨 Test 3: Diagram Need Detection")
        test_questions = [
            "Explain TCP handshake",
            "Draw a flowchart of the process",
            "What is deadlock?",
            "Show me the OSI model"
        ]
        
        for question in test_questions:
            needs_diagram = generator.detect_diagram_need(question, "")
            diagram_type = generator.suggest_diagram_type(question)
            print(f"   '{question[:40]}'")
            print(f"      Needs diagram: {needs_diagram}, Type: {diagram_type}")
        
        # Test automatic generation
        print("\n🎨 Test 4: Automatic Diagram Generation")
        result = generate_diagram_for_answer(
            "Draw the process state diagram",
            "Processes go through various states..."
        )
        
        if result:
            print(f"   Generated {len(result['diagrams'])} diagrams")
            print(f"   Suggested type: {result['diagram_type']}")
        
        # Show all built-in templates
        print("\n🎨 Test 5: Built-in Templates")
        templates = [
            "TCP three-way handshake",
            "Process state diagram",
            "Deadlock circular wait",
            "OSI model layers"
        ]
        
        for template in templates:
            ascii_result = generator.generate_ascii_art(template)
            print(f"\n   {template}:")
            print(ascii_result['art'][:200] + "..." if len(ascii_result['art']) > 200 else ascii_result['art'])
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration():
    """Test integration of all features"""
    print("\n" + "=" * 70)
    print("TEST 5: Integration Test".center(70))
    print("=" * 70)
    
    try:
        # Test workflow: Extract -> OCR -> Caption -> Generate
        print("\n🔄 Testing complete workflow...")
        
        # 1. Create test image
        print("\n1️⃣ Creating test diagram...")
        img = Image.new('RGB', (400, 300), color='white')
        draw = ImageDraw.Draw(img)
        draw.rectangle([(50, 50), (350, 250)], outline='black', width=2)
        draw.text((100, 120), "Network Topology", fill='black')
        
        test_path = Path("data/processed/images/test_integration.png")
        img.save(test_path)
        print(f"   ✓ Saved to: {test_path}")
        
        # 2. Run OCR
        print("\n2️⃣ Extracting text with OCR...")
        from src.data_processing.ocr_processor import OCRProcessor
        
        ocr = OCRProcessor()
        if ocr.tesseract_available:
            ocr_result = ocr.extract_text(img)
            print(f"   ✓ Extracted: {ocr_result['text'][:50]}")
        else:
            print("   ⚠️  OCR not available")
        
        # 3. Generate caption
        print("\n3️⃣ Generating image caption...")
        from src.inference.vision_language import ImageCaptioner, BLIP_AVAILABLE
        
        if BLIP_AVAILABLE:
            captioner = ImageCaptioner()
            caption = captioner.generate_caption(img)
            print(f"   ✓ Caption: {caption}")
        else:
            print("   ⚠️  BLIP not available")
        
        # 4. Generate similar diagram
        print("\n4️⃣ Generating diagram from description...")
        from src.inference.figure_generator import FigureGenerator
        
        generator = FigureGenerator()
        ascii_diagram = generator.generate_ascii_art("network topology")
        print("   ✓ Generated ASCII diagram")
        print(ascii_diagram['art'][:200])
        
        print("\n✅ Integration test complete!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("=" * 70)
    print("MULTIMEDIA FEATURES TEST SUITE".center(70))
    print("=" * 70)
    
    results = {}
    
    # Run tests
    results['image_extraction'] = test_image_extraction()
    results['ocr'] = test_ocr()
    results['vision_language'] = test_vision_language()
    results['figure_generation'] = test_figure_generation()
    results['integration'] = test_integration()
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY".center(70))
    print("=" * 70)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name.replace('_', ' ').title():.<50} {status}")
    
    total = len(results)
    passed = sum(results.values())
    
    print(f"\n{'Total:':.<50} {passed}/{total} passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
    
    # Installation instructions
    print("\n" + "=" * 70)
    print("INSTALLATION NOTES".center(70))
    print("=" * 70)
    
    print("\n📦 Required packages:")
    print("   pip install pytesseract opencv-python Pillow transformers torch")
    
    print("\n🔧 System dependencies:")
    print("   Tesseract OCR:")
    print("      Windows: choco install tesseract")
    print("      Linux:   sudo apt-get install tesseract-ocr")
    print("      macOS:   brew install tesseract")
    
    print("\n📚 See MULTIMEDIA_GUIDE.md for detailed documentation")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
