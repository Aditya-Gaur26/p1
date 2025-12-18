"""
Advanced Dataset Creator using LLM-based question generation
Supports: OpenAI GPT-4, Anthropic Claude, or local models via Ollama
"""

import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.config import config
from src.utils.helpers import save_jsonl, load_json, generate_id


class AdvancedDatasetCreator:
    """Generate high-quality Q&A pairs using LLM"""
    
    def __init__(self, llm_provider: str = "openai", model: str = None):
        """
        Initialize dataset creator
        
        Args:
            llm_provider: 'openai', 'anthropic', 'ollama', 'groq', or 'github'
            model: Model name (e.g., 'gpt-4', 'claude-3-sonnet', 'llama3:8b', 'gpt-4o-mini')
        """
        self.llm_provider = llm_provider.lower()
        self.model = model
        self.client = None
        
        self._setup_llm_client()
    
    def _setup_llm_client(self):
        """Setup LLM client based on provider"""
        
        if self.llm_provider == "openai":
            try:
                from openai import OpenAI
                self.client = OpenAI()  # Reads OPENAI_API_KEY from environment
                self.model = self.model or "gpt-4o-mini"  # Cheaper option
                print(f"✓ Using OpenAI: {self.model}")
            except ImportError:
                print("❌ OpenAI not installed. Run: pip install openai")
                sys.exit(1)
        
        elif self.llm_provider == "anthropic":
            try:
                from anthropic import Anthropic
                self.client = Anthropic()  # Reads ANTHROPIC_API_KEY from environment
                self.model = self.model or "claude-3-5-sonnet-20241022"
                print(f"✓ Using Anthropic: {self.model}")
            except ImportError:
                print("❌ Anthropic not installed. Run: pip install anthropic")
                sys.exit(1)
        
        elif self.llm_provider == "ollama":
            try:
                from ollama import Client
                self.client = Client()
                self.model = self.model or "llama3.2:latest"
                print(f"✓ Using Ollama: {self.model}")
            except ImportError:
                print("❌ Ollama not installed. Run: pip install ollama")
                sys.exit(1)
        
        elif self.llm_provider == "groq":
            try:
                from groq import Groq
                self.client = Groq()  # Reads GROQ_API_KEY from environment
                self.model = self.model or "llama-3.3-70b-versatile"
                print(f"✓ Using Groq: {self.model}")
            except ImportError:
                print("❌ Groq not installed. Run: pip install groq")
                sys.exit(1)
        
        elif self.llm_provider == "github":
            try:
                from openai import OpenAI
                import os
                token = os.getenv("GITHUB_TOKEN")
                if not token:
                    print("❌ GITHUB_TOKEN not set!")
                    sys.exit(1)
                self.client = OpenAI(
                    base_url="https://models.inference.ai.azure.com",
                    api_key=token
                )
                self.model = self.model or "gpt-4o-mini"
                print(f"✓ Using GitHub Models: {self.model} (FREE)")
            except ImportError:
                print("❌ OpenAI not installed. Run: pip install openai")
                sys.exit(1)
        
        else:
            raise ValueError(f"Unknown LLM provider: {self.llm_provider}")
    
    def generate_questions(self, text_chunk: str, num_questions: int = 5) -> List[Dict[str, str]]:
        """
        Generate diverse, high-quality questions from a text chunk
        
        Returns list of dicts with 'question', 'answer', 'type', 'difficulty'
        """
        
        prompt = f"""You are an expert educator creating exam questions from a Computer Science textbook.

Given the following text excerpt from a textbook, generate {num_questions} high-quality questions that test deep understanding.

TEXT EXCERPT:
{text_chunk[:2000]}  

REQUIREMENTS:
1. Create DIVERSE question types:
   - Definition questions (What is X?)
   - Explanation questions (Explain how X works)
   - Comparison questions (Compare X vs Y)
   - Application questions (When would you use X?)
   - Analysis questions (Why does X happen?)
   - Problem-solving questions (Calculate/Design X)

2. Each question must be answerable from the text above
3. Vary difficulty levels: easy, medium, hard
4. Use specific terminology from the text
5. Make questions exam-worthy and practical

OUTPUT FORMAT (JSON array):
[
  {{
    "question": "What is the purpose of the transport layer in the OSI model?",
    "answer": "The transport layer provides end-to-end communication...",
    "type": "explanation",
    "difficulty": "medium"
  }},
  ...
]

Generate exactly {num_questions} questions. Return ONLY the JSON array, no other text."""
in ["openai", "github"]
        try:
            response = self._call_llm(prompt)
            questions = self._parse_json_response(response)
            
            if not questions or len(questions) == 0:
                print(f"⚠️  No questions generated, using fallback")
                return self._fallback_questions(text_chunk)
            
            return questions[:num_questions]
        
        except Exception as e:
            print(f"⚠️  LLM generation failed: {e}, using fallback")
            return self._fallback_questions(text_chunk)
    
    def _call_llm(self, prompt: str) -> str:
        """Call LLM API based on provider"""
        
        if self.llm_provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert educator creating exam questions."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2000
            )
            return response.choices[0].message.content
        
        elif self.llm_provider == "anthropic":
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                temperature=0.7,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.content[0].text
        
        elif self.llm_provider == "ollama":
            response = self.client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert educator."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response['message']['content']
        
        elif self.llm_provider == "groq":
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert educator creating exam questions."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2000
            )
            return response.choices[0].message.content
    
    def _parse_json_response(self, response: str) -> List[Dict]:
        """Extract JSON from LLM response"""
        # Remove markdown code blocks if present
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0]
        elif "```" in response:
            response = response.split("```")[1].split("```")[0]
        
        response = response.strip()
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # Try to find JSON array in text
            import re
            match = re.search(r'\[.*\]', response, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            return []
    
    def _fallback_questions(self, text_chunk: str) -> List[Dict]:
        """Simple fallback if LLM fails"""
        return [{
            "question": "Explain the concept described in the text.",
            "answer": text_chunk[:500],
            "type": "explanation",
            "difficulty": "medium"
        }]
    
    def create_training_examples(self, chunk: str, source: str) -> List[Dict[str, Any]]:
        """Create training examples from a text chunk"""
        
        questions = self.generate_questions(chunk, num_questions=3)
        
        training_examples = []
        for qa in questions:
            example = {
                "id": generate_id(qa['question'] + chunk[:50]),
                "instruction": qa['question'],
                "input": "",
                "output": qa['answer'],
                "source": source,
                "question_type": qa.get('type', 'general'),
                "difficulty": qa.get('difficulty', 'medium'),
                "text": f"### Instruction:\n{qa['question']}\n\n### Response:\n{qa['answer']}"
            }
            training_examples.append(example)
        
        return training_examples


def process_books_with_llm(llm_provider: str = "openai", model: str = None, 
                           max_chunks: int = None):
    """
    Process books using LLM to generate high-quality dataset
    
    Args:
        llm_provider: 'openai', 'anthropic', 'ollama', or 'groq'
        model: Specific model name
        max_chunks: Limit number of chunks to process (for testing/cost control)
    """
    
    print("=" * 70)
    print("Advanced Dataset Creation with LLM".center(70))
    print("=" * 70)
    
    # Initialize creator
    creator = AdvancedDatasetCreator(llm_provider, model)
    
    # Load processed books
    books_dir = config.data_dir / "processed" / "books"
    combined_file = books_dir / "all_pdfs_combined.json"
    
    if not combined_file.exists():
        print("❌ No processed books found. Run extract_pdfs.py first.")
        return
    
    data = load_json(combined_file)
    print(f"\n📚 Found {data['total_files']} books with {sum(f['num_chunks'] for f in data['files'])} chunks")
    
    # Process chunks
    all_training_data = []
    total_chunks = sum(f['num_chunks'] for f in data['files'])
    
    if max_chunks:
        print(f"⚠️  Limiting to {max_chunks} chunks for testing/cost control")
        total_chunks = min(total_chunks, max_chunks)
    
    chunks_processed = 0
    
    with tqdm(total=total_chunks, desc="Generating Q&A pairs") as pbar:
        for file_data in data['files']:
            filename = file_data['filename']
            
            for i, chunk in enumerate(file_data['chunks'], 1):
                if max_chunks and chunks_processed >= max_chunks:
                    break
                
                if len(chunk) < 200:  # Skip very short chunks
                    pbar.update(1)
                    continue
                
                source = f"{filename} - Section {i}"
                
                try:
                    examples = creator.create_training_examples(chunk, source)
                    all_training_data.extend(examples)
                    
                    pbar.set_postfix({
                        'questions': len(all_training_data),
                        'source': filename[:20]
                    })
                    
                    # Rate limiting
                    if llm_provider in ["openai", "anthropic", "groq"]:
                        time.sleep(0.5)  # Avoid rate limits
                
                except Exception as e:
                    print(f"\n⚠️  Error processing chunk {i} from {filename}: {e}")
                
                chunks_processed += 1
                pbar.update(1)
            
            if max_chunks and chunks_processed >= max_chunks:
                break
    
    if not all_training_data:
        print("\n❌ No training data generated!")
        return
    
    # Shuffle and split
    import random
    random.shuffle(all_training_data)
    
    split_idx = int(len(all_training_data) * 0.9)
    train_data = all_training_data[:split_idx]
    val_data = all_training_data[split_idx:]
    
    # Save with LLM-specific naming (keeps old dataset intact)
    output_dir = config.data_dir / "processed"
    save_jsonl(train_data, output_dir / "train_llm.jsonl")
    save_jsonl(val_data, output_dir / "val_llm.jsonl")
    
    # Print statistics
    print(f"\n" + "=" * 70)
    print("✅ Advanced Dataset Created!".center(70))
    print("=" * 70)
    print(f"\n📊 Statistics:")
    print(f"   Total Q&A pairs: {len(all_training_data)}")
    print(f"   Training set: {len(train_data)}")
    print(f"   Validation set: {len(val_data)}")
    
    # Analyze question types
    type_counts = {}
    for item in all_training_data:
        qtype = item.get('question_type', 'general')
        type_counts[qtype] = type_counts.get(qtype, 0) + 1
    
    print(f"\n📋 Question Types:")
    for qtype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"   {qtype}: {count}")
    
    # Analyze difficulty
    diff_counts = {}
    for item in all_training_data:
        diff = item.get('difficulty', 'medium')
        diff_counts[diff] = diff_counts.get(diff, 0) + 1
    
    print(f"\n🎯 Difficulty Distribution:")
    for diff, count in sorted(diff_counts.items()):
        print(f"   {diff}: {count}")
    
    print(f"\n📁 New LLM-generated dataset saved to:")
    print(f"   {output_dir / 'train_llm.jsonl'}")
    print(f"   {output_dir / 'val_llm.jsonl'}")
    print(f"\n💡 Old keyword-based dataset preserved at:")
    print(f"   {output_dir / 'train.jsonl'}")
    print(f"   {output_dir / 'val.jsonl'}")
    
    # Estimate cost (for API providers)
    if llm_provider == "openai":
        est_tokens = chunks_processed * 3000  # Rough estimate
        est_cost = (est_tokens / 1000000) * 0.15  # gpt-4o-mini pricing
        print(f"\n💰 Estimated API cost: ${est_cost:.2f}")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create advanced dataset with LLM")
    parser.add_argument("--provider", type=str, default="openai",
                       choices=["openai", "anthropic", "ollama", "groq", "github"],
                       help="LLM provider to use")
    parser.add_argument("--model", type=str, default=None,
                       help="Specific model name (e.g., gpt-4o-mini, llama3.2)")
    parser.add_argument("--max-chunks", type=int, default=None,
                       help="Max chunks to process (for testing/cost control)")
    parser.add_argument("--test", action="store_true",
                       help="Test mode: process only 10 chunks")
    
    args = parser.parse_args()
    
    if args.test:
        print("🧪 TEST MODE: Processing only 10 chunks")
        args.max_chunks = 10
    
    # Check for API keys
    if args.provider == "openai":
        import os
        if not os.getenv("OPENAI_API_KEY"):
            print("❌ OPENAI_API_KEY environment variable not set!")
            print("   Set it with: $env:OPENAI_API_KEY='your-key-here'")
            return
    
    elif args.provider == "anthropic":
        import os
        if not os.getenv("ANTHROPIC_API_KEY"):
            print("❌ ANTHROPIC_API_KEY environment variable not set!")
            print("   Set it with: $env:ANTHROPIC_API_KEY='your-key-here'")
            return
    
    elif args.provider == "groq":
        import os
        if not os.getenv("GROQ_API_KEY"):
            print("❌ GROQ_API_KEY environment variable not set!")
            print("   Set it with: $env:GROQ_API_KEY='your-key-here'")
            return
    
    elif args.provider == "github":
        import os
        if not os.getenv("GITHUB_TOKEN"):
            print("❌ GITHUB_TOKEN environment variable not set!")
            print("   Get token from: https://github.com/settings/tokens")
            print("   Set it with: $env:GITHUB_TOKEN='ghp_...'")
            return
    
    elif args.provider == "ollama":
        print("ℹ️  Make sure Ollama is running: ollama serve")
    
    process_books_with_llm(args.provider, args.model, args.max_chunks)


if __name__ == "__main__":
    main()
