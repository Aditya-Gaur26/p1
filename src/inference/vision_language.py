"""
Vision-Language Integration Module
Uses BLIP and other vision models for image understanding
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from PIL import Image
import torch

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.config import config

# Check BLIP availability
try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
    BLIP_AVAILABLE = True
except ImportError:
    BLIP_AVAILABLE = False

# Check CLIP availability for zero-shot classification
try:
    from transformers import CLIPProcessor, CLIPModel
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False


class ImageCaptioner:
    """Generate captions for images using BLIP"""
    
    def __init__(self, model_name: str = "Salesforce/blip-image-captioning-base"):
        """
        Initialize image captioner
        
        Args:
            model_name: Hugging Face model name
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = None
        self.model = None
        
        if BLIP_AVAILABLE:
            self._load_model()
    
    def _load_model(self):
        """Load BLIP model and processor"""
        try:
            print(f"Loading BLIP model: {self.model_name}...")
            self.processor = BlipProcessor.from_pretrained(self.model_name)
            self.model = BlipForConditionalGeneration.from_pretrained(self.model_name).to(self.device)
            print(f"✅ BLIP model loaded on {self.device}")
        except Exception as e:
            print(f"⚠️  Failed to load BLIP: {e}")
            self.processor = None
            self.model = None
    
    def generate_caption(self, image: Image.Image, max_length: int = 50) -> str:
        """
        Generate caption for image
        
        Args:
            image: PIL Image
            max_length: Maximum caption length
        
        Returns:
            Generated caption
        """
        if not BLIP_AVAILABLE or self.model is None:
            return "[Image: Caption unavailable - BLIP not installed]"
        
        try:
            # Preprocess image
            inputs = self.processor(image, return_tensors="pt").to(self.device)
            
            # Generate caption
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_length=max_length)
            
            # Decode caption
            caption = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            return caption
            
        except Exception as e:
            print(f"⚠️  Caption generation failed: {e}")
            return "[Image: Caption generation failed]"
    
    def generate_conditional_caption(self, image: Image.Image, text_prompt: str, max_length: int = 50) -> str:
        """
        Generate caption conditioned on text prompt
        
        Args:
            image: PIL Image
            text_prompt: Text prompt to condition generation
            max_length: Maximum caption length
        
        Returns:
            Generated caption
        """
        if not BLIP_AVAILABLE or self.model is None:
            return "[Image: Caption unavailable]"
        
        try:
            # Preprocess with text prompt
            inputs = self.processor(image, text_prompt, return_tensors="pt").to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_length=max_length)
            
            caption = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            return caption
            
        except Exception as e:
            print(f"⚠️  Conditional caption failed: {e}")
            return "[Image: Caption generation failed]"
    
    def describe_diagram(self, image: Image.Image) -> str:
        """
        Generate description specifically for diagrams
        
        Args:
            image: PIL Image of diagram
        
        Returns:
            Description of diagram
        """
        # Use conditional generation with diagram-specific prompt
        prompt = "a diagram showing"
        description = self.generate_conditional_caption(image, prompt, max_length=100)
        
        return description
    
    def batch_caption(self, images: List[Image.Image], max_length: int = 50) -> List[str]:
        """
        Generate captions for multiple images
        
        Args:
            images: List of PIL Images
            max_length: Maximum caption length
        
        Returns:
            List of captions
        """
        captions = []
        
        for image in images:
            caption = self.generate_caption(image, max_length)
            captions.append(caption)
        
        return captions


class ImageClassifier:
    """Classify images using CLIP zero-shot classification"""
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        """
        Initialize image classifier
        
        Args:
            model_name: CLIP model name
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = None
        self.model = None
        
        if CLIP_AVAILABLE:
            self._load_model()
    
    def _load_model(self):
        """Load CLIP model"""
        try:
            print(f"Loading CLIP model: {self.model_name}...")
            self.processor = CLIPProcessor.from_pretrained(self.model_name)
            self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
            print(f"✅ CLIP model loaded on {self.device}")
        except Exception as e:
            print(f"⚠️  Failed to load CLIP: {e}")
            self.processor = None
            self.model = None
    
    def classify(self, image: Image.Image, candidate_labels: List[str]) -> Dict[str, float]:
        """
        Classify image into one of the candidate labels
        
        Args:
            image: PIL Image
            candidate_labels: List of possible labels
        
        Returns:
            Dictionary mapping labels to probabilities
        """
        if not CLIP_AVAILABLE or self.model is None:
            return {label: 0.0 for label in candidate_labels}
        
        try:
            # Preprocess
            inputs = self.processor(
                text=candidate_labels,
                images=image,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)[0]
            
            # Create result dict
            results = {label: prob.item() for label, prob in zip(candidate_labels, probs)}
            
            return results
            
        except Exception as e:
            print(f"⚠️  Classification failed: {e}")
            return {label: 0.0 for label in candidate_labels}
    
    def classify_diagram_type(self, image: Image.Image) -> Dict[str, float]:
        """
        Classify type of diagram
        
        Args:
            image: PIL Image of diagram
        
        Returns:
            Dictionary of diagram types and probabilities
        """
        diagram_types = [
            "flowchart diagram",
            "state diagram",
            "network diagram",
            "sequence diagram",
            "architecture diagram",
            "graph chart",
            "table",
            "text document",
            "screenshot",
            "photo"
        ]
        
        return self.classify(image, diagram_types)
    
    def is_educational_content(self, image: Image.Image) -> bool:
        """
        Check if image is educational content
        
        Args:
            image: PIL Image
        
        Returns:
            True if educational, False otherwise
        """
        labels = ["educational diagram", "presentation slide", "textbook page", "photo", "advertisement"]
        results = self.classify(image, labels)
        
        # Check if any educational category is dominant
        educational_score = results.get("educational diagram", 0) + \
                           results.get("presentation slide", 0) + \
                           results.get("textbook page", 0)
        
        return educational_score > 0.5


class VisionLanguageProcessor:
    """Combined vision-language processing"""
    
    def __init__(self):
        """Initialize vision-language processor"""
        self.captioner = ImageCaptioner() if BLIP_AVAILABLE else None
        self.classifier = ImageClassifier() if CLIP_AVAILABLE else None
    
    def process_image(self, image: Image.Image) -> Dict[str, Any]:
        """
        Complete image processing pipeline
        
        Args:
            image: PIL Image
        
        Returns:
            Dictionary with all extracted information
        """
        result = {
            'size': image.size,
            'mode': image.mode,
            'format': image.format,
        }
        
        # Generate caption
        if self.captioner:
            result['caption'] = self.captioner.generate_caption(image)
            result['diagram_description'] = self.captioner.describe_diagram(image)
        else:
            result['caption'] = "[Caption unavailable]"
            result['diagram_description'] = "[Description unavailable]"
        
        # Classify
        if self.classifier:
            result['diagram_type'] = self.classifier.classify_diagram_type(image)
            result['is_educational'] = self.classifier.is_educational_content(image)
        else:
            result['diagram_type'] = {}
            result['is_educational'] = True  # Assume educational by default
        
        return result
    
    def process_batch(self, images: List[Image.Image]) -> List[Dict[str, Any]]:
        """
        Process multiple images
        
        Args:
            images: List of PIL Images
        
        Returns:
            List of processing results
        """
        results = []
        
        for image in images:
            result = self.process_image(image)
            results.append(result)
        
        return results
    
    def extract_text_alternative(self, image: Image.Image) -> str:
        """
        Generate text alternative for image (for accessibility)
        
        Args:
            image: PIL Image
        
        Returns:
            Text description
        """
        if self.captioner:
            caption = self.captioner.generate_caption(image)
            
            # Get diagram type if available
            if self.classifier:
                diagram_types = self.classifier.classify_diagram_type(image)
                top_type = max(diagram_types.items(), key=lambda x: x[1])[0]
                
                return f"[{top_type.capitalize()}: {caption}]"
            
            return f"[Image: {caption}]"
        
        return "[Image: Description unavailable]"


def main():
    """Test vision-language processing"""
    print("=" * 70)
    print("Vision-Language Processor Test".center(70))
    print("=" * 70)
    
    print(f"\n✅ BLIP Available: {BLIP_AVAILABLE}")
    print(f"✅ CLIP Available: {CLIP_AVAILABLE}")
    
    if not BLIP_AVAILABLE:
        print("\n⚠️  Install BLIP: pip install transformers torch pillow")
        return
    
    # Create test image (simple diagram)
    print("\n📷 Creating test image...")
    from PIL import Image, ImageDraw, ImageFont
    
    # Create a simple flowchart-like image
    img = Image.new('RGB', (400, 300), color='white')
    draw = ImageDraw.Draw(img)
    
    # Draw boxes
    draw.rectangle([(100, 50), (300, 100)], outline='black', width=2)
    draw.text((150, 65), "Start", fill='black')
    
    draw.rectangle([(100, 130), (300, 180)], outline='black', width=2)
    draw.text((140, 145), "Process", fill='black')
    
    draw.rectangle([(100, 210), (300, 260)], outline='black', width=2)
    draw.text((160, 225), "End", fill='black')
    
    # Draw arrows
    draw.line([(200, 100), (200, 130)], fill='black', width=2)
    draw.line([(200, 180), (200, 210)], fill='black', width=2)
    
    # Process image
    print("\n🔍 Processing image...")
    processor = VisionLanguageProcessor()
    result = processor.process_image(img)
    
    print("\n📊 Results:")
    print(f"  Caption: {result.get('caption', 'N/A')}")
    print(f"  Diagram Description: {result.get('diagram_description', 'N/A')}")
    print(f"  Is Educational: {result.get('is_educational', 'N/A')}")
    
    if result.get('diagram_type'):
        print(f"\n  Diagram Type Classification:")
        for dtype, score in sorted(result['diagram_type'].items(), key=lambda x: x[1], reverse=True)[:3]:
            print(f"    - {dtype}: {score:.3f}")
    
    print(f"\n  Text Alternative: {processor.extract_text_alternative(img)}")
    
    print("\n✅ Vision-language test complete!")


if __name__ == "__main__":
    main()
