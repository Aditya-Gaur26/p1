"""
OCR (Optical Character Recognition) module for extracting text from images
Supports diagrams, charts, and images in slides/documents
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
import numpy as np

sys.path.append(str(Path(__file__).parent.parent.parent))

# Try to import OCR libraries
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


class OCRProcessor:
    """OCR processor for extracting text from images"""
    
    def __init__(self, tesseract_cmd: Optional[str] = None):
        """
        Initialize OCR processor
        
        Args:
            tesseract_cmd: Path to tesseract executable (optional)
        """
        if not TESSERACT_AVAILABLE:
            raise ImportError("pytesseract not installed. Run: pip install pytesseract")
        
        # Set tesseract command if provided
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        
        self.available = TESSERACT_AVAILABLE
    
    def preprocess_image(self, image: Image.Image, enhance: bool = True) -> Image.Image:
        """
        Preprocess image for better OCR results
        
        Args:
            image: PIL Image object
            enhance: Apply enhancement techniques
        
        Returns:
            Preprocessed PIL Image
        """
        if not CV2_AVAILABLE or not enhance:
            return image
        
        # Convert PIL to numpy array
        img_array = np.array(image)
        
        # Convert to grayscale if color
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Apply thresholding
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(binary)
        
        # Convert back to PIL
        return Image.fromarray(denoised)
    
    def extract_text(self, image: Image.Image, 
                    preprocess: bool = True,
                    lang: str = 'eng') -> Dict[str, Any]:
        """
        Extract text from image using OCR
        
        Args:
            image: PIL Image object
            preprocess: Apply preprocessing
            lang: Language for OCR
        
        Returns:
            Dictionary with extracted text and metadata
        """
        if not self.available:
            return {
                'text': '',
                'confidence': 0.0,
                'words': [],
                'error': 'Tesseract not available'
            }
        
        try:
            # Preprocess if requested
            if preprocess:
                image = self.preprocess_image(image)
            
            # Extract text
            text = pytesseract.image_to_string(image, lang=lang)
            
            # Get detailed data
            data = pytesseract.image_to_data(image, lang=lang, output_type=pytesseract.Output.DICT)
            
            # Calculate average confidence
            confidences = [float(c) for c in data['conf'] if c != '-1']
            avg_confidence = np.mean(confidences) if confidences else 0.0
            
            # Extract words with confidence
            words = []
            for i, word in enumerate(data['text']):
                if word.strip() and data['conf'][i] != '-1':
                    words.append({
                        'text': word,
                        'confidence': float(data['conf'][i]),
                        'bbox': {
                            'x': data['left'][i],
                            'y': data['top'][i],
                            'width': data['width'][i],
                            'height': data['height'][i]
                        }
                    })
            
            return {
                'text': text.strip(),
                'confidence': avg_confidence,
                'words': words,
                'word_count': len(words),
                'language': lang
            }
        
        except Exception as e:
            return {
                'text': '',
                'confidence': 0.0,
                'words': [],
                'error': str(e)
            }
    
    def detect_text_regions(self, image: Image.Image) -> List[Dict[str, Any]]:
        """
        Detect text regions in image
        
        Args:
            image: PIL Image object
        
        Returns:
            List of text region dictionaries
        """
        if not CV2_AVAILABLE:
            return []
        
        try:
            # Convert to numpy array
            img_array = np.array(image)
            
            # Convert to grayscale
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            # Detect edges
            edges = cv2.Canny(gray, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours for text-like regions
            text_regions = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter by size and aspect ratio
                aspect_ratio = w / h if h > 0 else 0
                area = w * h
                
                if 0.1 < aspect_ratio < 10 and area > 100:
                    text_regions.append({
                        'x': int(x),
                        'y': int(y),
                        'width': int(w),
                        'height': int(h),
                        'area': int(area),
                        'aspect_ratio': float(aspect_ratio)
                    })
            
            return text_regions
        
        except Exception as e:
            print(f"⚠️  Error detecting text regions: {e}")
            return []
    
    def is_diagram(self, image: Image.Image, text_threshold: float = 0.1) -> bool:
        """
        Determine if image is likely a diagram/chart vs text
        
        Args:
            image: PIL Image object
            text_threshold: Threshold for text density
        
        Returns:
            True if image appears to be a diagram
        """
        try:
            # Extract text
            result = self.extract_text(image, preprocess=True)
            text = result['text']
            
            # Calculate text density
            total_pixels = image.width * image.height
            text_length = len(text.strip())
            text_density = text_length / (total_pixels / 1000)  # Normalize
            
            # Low text density suggests diagram
            is_diagram = text_density < text_threshold
            
            return is_diagram
        
        except Exception:
            return False
    
    def extract_from_diagram(self, image: Image.Image) -> Dict[str, Any]:
        """
        Extract information from diagrams/charts
        
        Args:
            image: PIL Image object
        
        Returns:
            Dictionary with extracted information
        """
        # Detect text regions
        regions = self.detect_text_regions(image)
        
        # Extract text from each region
        region_texts = []
        for region in regions:
            try:
                # Crop region
                x, y, w, h = region['x'], region['y'], region['width'], region['height']
                cropped = image.crop((x, y, x + w, y + h))
                
                # Extract text
                result = self.extract_text(cropped, preprocess=True)
                if result['text']:
                    region_texts.append({
                        'text': result['text'],
                        'region': region,
                        'confidence': result['confidence']
                    })
            except Exception:
                continue
        
        # Extract overall text
        overall = self.extract_text(image, preprocess=True)
        
        return {
            'is_diagram': self.is_diagram(image),
            'overall_text': overall['text'],
            'overall_confidence': overall['confidence'],
            'text_regions': region_texts,
            'region_count': len(regions)
        }


class DiagramClassifier:
    """Classify diagram types"""
    
    DIAGRAM_TYPES = {
        'flowchart': ['flow', 'flowchart', 'process', 'decision', 'start', 'end'],
        'state_diagram': ['state', 'transition', 'initial', 'final'],
        'network': ['network', 'router', 'switch', 'node', 'topology', 'lan', 'wan'],
        'sequence': ['sequence', 'actor', 'message', 'lifeline'],
        'architecture': ['architecture', 'component', 'layer', 'module'],
        'graph': ['graph', 'node', 'edge', 'vertex'],
        'chart': ['chart', 'bar', 'pie', 'line', 'plot'],
        'timeline': ['timeline', 'time', 'schedule', 'gantt'],
        'memory': ['memory', 'address', 'stack', 'heap', 'segment'],
        'protocol': ['protocol', 'packet', 'header', 'tcp', 'udp', 'ip']
    }
    
    def __init__(self):
        """Initialize diagram classifier"""
        pass
    
    def classify(self, text: str, image_info: Optional[Dict] = None) -> List[Tuple[str, float]]:
        """
        Classify diagram type based on text content
        
        Args:
            text: Extracted text from diagram
            image_info: Optional image metadata
        
        Returns:
            List of (diagram_type, confidence) tuples
        """
        text_lower = text.lower()
        scores = []
        
        for diagram_type, keywords in self.DIAGRAM_TYPES.items():
            # Count keyword matches
            matches = sum(1 for keyword in keywords if keyword in text_lower)
            
            if matches > 0:
                # Calculate confidence score
                confidence = min(matches / len(keywords), 1.0)
                scores.append((diagram_type, confidence))
        
        # Sort by confidence
        scores.sort(key=lambda x: x[1], reverse=True)
        
        return scores if scores else [('unknown', 0.0)]


def main():
    """Test OCR functionality"""
    print("=" * 70)
    print("OCR Processor Test".center(70))
    print("=" * 70)
    
    if not TESSERACT_AVAILABLE:
        print("\n❌ Tesseract not available!")
        print("Install with: pip install pytesseract")
        print("Also install Tesseract-OCR: https://github.com/tesseract-ocr/tesseract")
        return
    
    print("\n✓ Tesseract available")
    print(f"✓ OpenCV available: {CV2_AVAILABLE}")
    
    # Test with sample image if available
    processor = OCRProcessor()
    print("\n✅ OCR Processor initialized successfully!")


if __name__ == "__main__":
    main()
