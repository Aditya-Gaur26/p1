"""
Extract text from PDF files (books and documents) with enhanced preprocessing
"""

import sys
from pathlib import Path
from typing import List, Dict
import PyPDF2
import pdfplumber

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.config import config
from src.utils.helpers import save_json, clean_text, chunk_text
from src.utils.text_preprocessing import AdvancedTextCleaner, clean_pdf_text


def extract_with_pypdf2(filepath: Path) -> str:
    """Extract text using PyPDF2"""
    text = ""
    try:
        with open(filepath, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
    except Exception as e:
        print(f"  ⚠️  PyPDF2 extraction failed: {str(e)}")
    
    return text


def extract_with_pdfplumber(filepath: Path) -> str:
    """Extract text using pdfplumber (better for complex PDFs)"""
    text = ""
    try:
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
    except Exception as e:
        print(f"  ⚠️  pdfplumber extraction failed: {str(e)}")
    
    return text


def extract_from_pdf(filepath: Path) -> Dict:
    """Extract content from PDF file with enhanced cleaning"""
    print(f"Processing: {filepath.name}...")
    
    # Try pdfplumber first (better quality), fallback to PyPDF2
    text = extract_with_pdfplumber(filepath)
    
    if not text or len(text.strip()) < 100:
        print("  → Trying alternative extraction method...")
        text = extract_with_pypdf2(filepath)
    
    if not text or len(text.strip()) < 100:
        print(f"  ❌ Failed to extract meaningful text from {filepath.name}")
        return None
    
    # Enhanced cleaning for PDF text
    cleaned_text = clean_pdf_text(text)
    
    # Extract structured content
    cleaner = AdvancedTextCleaner()
    structured_content = cleaner.extract_structured_content(cleaned_text)
    
    # Chunk the text with better boundaries
    chunks = chunk_text(cleaned_text, chunk_size=512, overlap=50)
    
    return {
        "filename": filepath.name,
        "text": cleaned_text,
        "chunks": chunks,
        "num_chunks": len(chunks),
        "char_count": len(cleaned_text),
        "headers": structured_content.get('headers', []),
        "has_code": len(structured_content.get('code_blocks', [])) > 0
    }


def process_all_pdfs(input_dir: Path, output_dir: Path):
    """Process all PDF files in directory"""
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all PDF files
    pdf_files = list(input_dir.glob("*.pdf"))
    
    if not pdf_files:
        print(f"⚠️  No PDF files found in {input_dir}")
        print(f"Please add PDF files to {input_dir}")
        return
    
    print(f"📚 Found {len(pdf_files)} PDF files")
    
    all_extracted = []
    
    for pdf_file in pdf_files:
        extracted = extract_from_pdf(pdf_file)
        
        if extracted:
            all_extracted.append(extracted)
            
            # Save individual file
            output_file = output_dir / f"{pdf_file.stem}_extracted.json"
            save_json(extracted, output_file)
            print(f"  ✓ Extracted {extracted['num_chunks']} chunks ({extracted['char_count']} chars)")
    
    # Save combined output
    if all_extracted:
        combined_output = output_dir / "all_pdfs_combined.json"
        save_json({"files": all_extracted, "total_files": len(all_extracted)}, combined_output)
        print(f"\n✅ Successfully processed {len(all_extracted)} files")
        print(f"📁 Output saved to: {output_dir}")
    else:
        print("❌ No files were successfully processed")


def main():
    """Main execution function"""
    input_dir = config.data_dir / "raw" / "books"
    output_dir = config.data_dir / "processed" / "books"
    
    print("=" * 60)
    print("PDF Text Extraction".center(60))
    print("=" * 60)
    
    process_all_pdfs(input_dir, output_dir)


if __name__ == "__main__":
    main()
