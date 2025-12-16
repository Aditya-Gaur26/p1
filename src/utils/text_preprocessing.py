"""
Enhanced text preprocessing and cleaning utilities
"""

import re
from typing import List, Dict, Any
from pathlib import Path


class AdvancedTextCleaner:
    """Advanced text cleaning and preprocessing"""
    
    def __init__(self):
        # Common OCR errors and fixes
        self.ocr_fixes = {
            r'\bl\s+': '1 ',  # lowercase L before space -> 1
            r'\s+l\b': ' 1',  # lowercase L after space -> 1
            r'\bO\s+': '0 ',  # uppercase O before space -> 0
            r'\|\|': 'll',     # double pipe -> ll
            r'\|\/\|': 'M',    # |/| -> M
            r'rn': 'm',        # rn -> m (common OCR error)
        }
        
        # Technical term patterns to preserve
        self.preserve_patterns = [
            r'\b[A-Z]{2,}\b',  # Acronyms (TCP, UDP, etc.)
            r'\b[A-Za-z]+\d+\b',  # Alphanumeric (IPv4, HTTP2, etc.)
            r'\d+\.\d+\.\d+\.\d+',  # IP addresses
            r'0x[0-9a-fA-F]+',  # Hex numbers
        ]
    
    def clean_text(self, text: str, aggressive: bool = False) -> str:
        """
        Clean and normalize text
        
        Args:
            text: Input text
            aggressive: Apply more aggressive cleaning
        
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+])+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+\.\S+', '', text)
        
        # Fix common OCR errors
        for pattern, replacement in self.ocr_fixes.items():
            text = re.sub(pattern, replacement, text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix sentence boundaries
        text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)
        
        # Remove repeated characters (hellllo -> hello)
        if aggressive:
            text = re.sub(r'(.)\1{2,}', r'\1\1', text)
        
        # Fix common punctuation issues
        text = re.sub(r'\s+([,.!?;:])', r'\1', text)  # Remove space before punctuation
        text = re.sub(r'([,.!?;:])\s*([,.!?;:])', r'\1 \2', text)  # Fix double punctuation
        
        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        # Remove control characters
        text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t')
        
        return text.strip()
    
    def extract_code_blocks(self, text: str) -> List[Dict[str, str]]:
        """
        Extract code blocks from text
        
        Returns:
            List of dictionaries with 'code' and 'language'
        """
        code_blocks = []
        
        # Markdown code blocks
        pattern = r'```(\w+)?\n(.*?)```'
        matches = re.finditer(pattern, text, re.DOTALL)
        
        for match in matches:
            language = match.group(1) or 'unknown'
            code = match.group(2).strip()
            code_blocks.append({
                'language': language,
                'code': code
            })
        
        return code_blocks
    
    def is_code_block(self, text: str) -> bool:
        """Check if text is likely a code block"""
        # Heuristics for code detection
        code_indicators = [
            r'\bint\s+\w+\s*=',  # Variable declarations
            r'\bdef\s+\w+\s*\(',  # Python function
            r'\bclass\s+\w+',  # Class definition
            r'[{};]\s*$',  # Code punctuation at end
            r'^\s*#include',  # C/C++ include
            r'^\s*import\s+',  # Import statements
        ]
        
        for pattern in code_indicators:
            if re.search(pattern, text, re.MULTILINE):
                return True
        
        # Check for high density of special characters
        special_chars = sum(1 for c in text if c in '{}[]();=<>+-*/')
        if len(text) > 0 and special_chars / len(text) > 0.1:
            return True
        
        return False
    
    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences intelligently"""
        # Handle common abbreviations
        text = re.sub(r'\bDr\.', 'Dr', text)
        text = re.sub(r'\bMr\.', 'Mr', text)
        text = re.sub(r'\bMrs\.', 'Mrs', text)
        text = re.sub(r'\be\.g\.', 'eg', text)
        text = re.sub(r'\bi\.e\.', 'ie', text)
        text = re.sub(r'\betc\.', 'etc', text)
        
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        
        # Restore abbreviations
        sentences = [s.replace('Dr', 'Dr.').replace('Mr', 'Mr.').replace('Mrs', 'Mrs.')
                    .replace('eg', 'e.g.').replace('ie', 'i.e.').replace('etc', 'etc.')
                    for s in sentences]
        
        return [s.strip() for s in sentences if s.strip()]
    
    def extract_structured_content(self, text: str) -> Dict[str, Any]:
        """
        Extract structured content from text
        
        Returns:
            Dictionary with headers, bullet points, code blocks, etc.
        """
        content = {
            'headers': [],
            'paragraphs': [],
            'bullet_points': [],
            'code_blocks': [],
            'equations': []
        }
        
        # Extract headers (markdown style)
        headers = re.findall(r'^#{1,6}\s+(.+)$', text, re.MULTILINE)
        content['headers'] = headers
        
        # Extract bullet points
        bullets = re.findall(r'^\s*[-*•]\s+(.+)$', text, re.MULTILINE)
        content['bullet_points'] = bullets
        
        # Extract numbered lists
        numbered = re.findall(r'^\s*\d+\.\s+(.+)$', text, re.MULTILINE)
        content['bullet_points'].extend(numbered)
        
        # Extract code blocks
        content['code_blocks'] = self.extract_code_blocks(text)
        
        # Extract equations (LaTeX style)
        equations = re.findall(r'\$\$(.+?)\$\$', text, re.DOTALL)
        content['equations'] = equations
        
        # Split into paragraphs
        paragraphs = text.split('\n\n')
        content['paragraphs'] = [p.strip() for p in paragraphs if p.strip()]
        
        return content
    
    def normalize_technical_terms(self, text: str) -> str:
        """Normalize technical terms for consistency"""
        # Common variations
        normalizations = {
            r'\bprocess\s+id\b': 'PID',
            r'\boperating\s+system\b': 'OS',
            r'\bcentral\s+processing\s+unit\b': 'CPU',
            r'\brandom\s+access\s+memory\b': 'RAM',
            r'\btransmission\s+control\s+protocol\b': 'TCP',
            r'\binternet\s+protocol\b': 'IP',
            r'\bdomain\s+name\s+system\b': 'DNS',
        }
        
        for pattern, replacement in normalizations.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text


def clean_slide_text(text: str) -> str:
    """Clean text extracted from slides"""
    cleaner = AdvancedTextCleaner()
    
    # Remove common slide artifacts
    text = re.sub(r'Slide\s+\d+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Page\s+\d+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)  # Slide numbers alone
    
    # Apply general cleaning
    text = cleaner.clean_text(text)
    
    return text


def clean_pdf_text(text: str) -> str:
    """Clean text extracted from PDFs"""
    cleaner = AdvancedTextCleaner()
    
    # Remove headers/footers (common patterns)
    text = re.sub(r'^.*?(?:Chapter|Section)\s+\d+.*?$', '', text, flags=re.MULTILINE)
    
    # Remove page numbers
    text = re.sub(r'^\s*-?\s*\d+\s*-?\s*$', '', text, flags=re.MULTILINE)
    
    # Fix hyphenation across lines
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    
    # Apply general cleaning
    text = cleaner.clean_text(text, aggressive=True)
    
    return text
