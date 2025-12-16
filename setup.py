#!/usr/bin/env python
"""
Quick setup script for the project
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.config import config


def main():
    """Run initial setup"""
    
    print("=" * 60)
    print("BTP Selection - Project Setup".center(60))
    print("Operating Systems & Networks Fine-tuning".center(60))
    print("=" * 60)
    
    # Create directories
    print("\n1. Creating directory structure...")
    config.ensure_directories()
    
    # Check for .env file
    print("\n2. Checking environment configuration...")
    env_file = Path(".env")
    if not env_file.exists():
        print("   Creating .env file from template...")
        template = Path(".env.template")
        if template.exists():
            import shutil
            shutil.copy(template, env_file)
            print("   ✓ .env file created")
            print("   ⚠️  Please edit .env with your API keys!")
        else:
            print("   ⚠️  .env.template not found")
    else:
        print("   ✓ .env file exists")
    
    # Check for data
    print("\n3. Checking for course materials...")
    slides_dir = config.data_dir / "raw" / "slides"
    books_dir = config.data_dir / "raw" / "books"
    
    slides_count = len(list(slides_dir.glob("*.ppt*")))
    books_count = len(list(books_dir.glob("*.pdf")))
    
    print(f"   Slides found: {slides_count}")
    print(f"   Books found: {books_count}")
    
    if slides_count == 0 and books_count == 0:
        print("\n   ⚠️  No course materials found!")
        print(f"   Please add:")
        print(f"   - PowerPoint slides to: {slides_dir}")
        print(f"   - PDF books to: {books_dir}")
    
    # Next steps
    print("\n" + "=" * 60)
    print("Setup Complete!".center(60))
    print("=" * 60)
    
    print("\n📋 Next Steps:")
    print("\n1. Add course materials:")
    print(f"   - Slides (.pptx) → {slides_dir}")
    print(f"   - Books (.pdf) → {books_dir}")
    
    print("\n2. Configure API keys (optional):")
    print("   - Edit .env file with YouTube and other API keys")
    
    print("\n3. Process data:")
    print("   python src/data_processing/extract_slides.py")
    print("   python src/data_processing/extract_pdfs.py")
    print("   python src/data_processing/create_dataset.py")
    print("   python src/data_processing/build_vectordb.py")
    
    print("\n4. Fine-tune model:")
    print("   python src/training/fine_tune.py")
    
    print("\n5. Test the system:")
    print("   python src/inference/query_processor.py --interactive")
    
    print("\n6. Evaluate:")
    print("   python src/evaluation/evaluate_model.py")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
