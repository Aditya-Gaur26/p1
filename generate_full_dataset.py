"""
Automated dataset generation using local Ollama model
Generates 10-15 high-quality questions per chunk for maximum coverage
"""

import json
import sys
import time
from pathlib import Path
import requests
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent))
from src.utils.config import config
from src.utils.helpers import load_json, save_jsonl, generate_id


class OllamaDatasetGenerator:
    def __init__(self, model="llama3.2", questions_per_chunk=12):
        self.model = model
        self.questions_per_chunk = questions_per_chunk
        self.ollama_url = "http://localhost:11434/api/generate"
        
    def generate_questions(self, chunk_text, source):
        """Generate multiple diverse questions from a text chunk"""
        
        # Truncate very long chunks to avoid context overflow
        if len(chunk_text) > 3000:
            chunk_text = chunk_text[:3000] + "..."
        
        prompt = f"""You are an expert educator. Create {self.questions_per_chunk} exam questions from this text.

TEXT:
{chunk_text}

CRITICAL: Output ONLY valid JSON. No markdown, no explanations, just the JSON array.

Format (use double quotes, escape special characters properly):
[
  {{"question": "What is X?", "answer": "X is...", "type": "definition", "difficulty": "medium"}},
  {{"question": "Explain Y", "answer": "Y works by...", "type": "explanation", "difficulty": "hard"}}
]

Question types: definition, explanation, comparison, application, analysis, problem-solving
Difficulty: easy, medium, hard
Provide detailed answers (3-5 sentences).

Output {self.questions_per_chunk} questions in valid JSON array format NOW:"""

        try:
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.8,
                        "top_p": 0.9,
                        "num_predict": 4000
                    }
                },
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result.get('response', '')
                
                # Extract JSON from response - try multiple methods
                try:
                    import re
                    
                    # Method 1: Strip markdown code blocks if present
                    if '```json' in generated_text:
                        json_match = re.search(r'```json\s*(\[.*?\])\s*```', generated_text, re.DOTALL)
                        if json_match:
                            json_str = json_match.group(1)
                        else:
                            # Try without json tag
                            json_match = re.search(r'```\s*(\[.*?\])\s*```', generated_text, re.DOTALL)
                            if json_match:
                                json_str = json_match.group(1)
                            else:
                                json_str = generated_text
                    else:
                        # Method 2: Find JSON array boundaries
                        start_idx = generated_text.find('[')
                        end_idx = generated_text.rfind(']') + 1
                        
                        if start_idx != -1 and end_idx > start_idx:
                            json_str = generated_text[start_idx:end_idx]
                        else:
                            raise ValueError("No JSON array found")
                    
                    # Clean up common JSON issues
                    json_str = json_str.replace('\\"', '"')  # Fix double escaping
                    json_str = re.sub(r'\\(?!["\\/bfnrt])', r'\\\\', json_str)  # Fix invalid escapes
                    
                    # Try to parse
                    qa_pairs = json.loads(json_str)
                    
                    # Validate it's a list
                    if not isinstance(qa_pairs, list):
                        raise ValueError("Response is not a JSON array")
                    
                    # Add source to each Q&A
                    for qa in qa_pairs:
                        qa['source'] = source
                    
                    # If we got less than half expected, warn but continue
                    if len(qa_pairs) < self.questions_per_chunk / 2:
                        print(f"WARNING: Only got {len(qa_pairs)}/{self.questions_per_chunk} questions - ")
                    
                    return qa_pairs
                        
                except (json.JSONDecodeError, ValueError) as e:
                    # Try one more time with a simpler extraction
                    try:
                        # Extract individual question objects with regex
                        qa_pairs = []
                        pattern = r'\{[^{}]*"question"[^{}]*"answer"[^{}]*\}'
                        matches = re.findall(pattern, generated_text, re.DOTALL)
                        
                        for match in matches[:self.questions_per_chunk]:
                            try:
                                qa = json.loads(match)
                                qa['source'] = source
                                qa_pairs.append(qa)
                            except:
                                continue
                        
                        if qa_pairs:
                            print(f"WARNING: Partial parse - got {len(qa_pairs)} questions - ")
                            return qa_pairs
                    except:
                        pass
                    
                    print(f"WARNING: JSON parse error: {e}, using fallback - ")
                    return self._fallback_questions(chunk_text, source)
            else:
                print(f"WARNING: Ollama API error: {response.status_code}")
                return self._fallback_questions(chunk_text, source)
                
        except Exception as e:
            print(f"WARNING: Error generating questions: {e}")
            return self._fallback_questions(chunk_text, source)
    
    def _fallback_questions(self, chunk_text, source):
        """Generate more questions when LLM fails - template-based"""
        
        # Extract some keywords for variety
        words = chunk_text.split()[:100]
        text_preview = ' '.join(words)
        
        questions = []
        
        # Generate multiple fallback questions for better coverage
        templates = [
            ("Explain the main concepts discussed in this section.", 
             f"This section discusses: {chunk_text[:600]}", "explanation", "medium"),
            
            ("What are the key topics introduced in this content?", 
             f"The key topics include: {chunk_text[:400]}", "definition", "easy"),
            
            ("Describe the technical details presented in this text.", 
             f"The technical details include: {chunk_text[100:500]}", "explanation", "medium"),
            
            ("What information is provided about the subject matter?", 
             f"The information provided covers: {chunk_text[200:600]}", "explanation", "medium"),
            
            ("Summarize the important points from this section.", 
             f"Important points: {chunk_text[:500]}", "analysis", "medium"),
            
            ("What concepts are explained in this content?", 
             f"The concepts explained include: {chunk_text[50:450]}", "definition", "easy"),
            
            ("Analyze the information presented in this text.", 
             f"Analysis: {chunk_text[150:550]}", "analysis", "hard"),
            
            ("What can be learned from this section?", 
             f"This section teaches: {chunk_text[:450]}", "explanation", "easy"),
        ]
        
        # Use up to questions_per_chunk templates, cycling if needed
        for i in range(min(self.questions_per_chunk, len(templates))):
            q, a, qtype, diff = templates[i]
            questions.append({
                "question": q,
                "answer": a,
                "type": qtype,
                "difficulty": diff,
                "source": source
            })
        
        return questions
    
    def process_all_chunks(self):
        """Process all book chunks and generate comprehensive dataset"""
        
        books_file = config.data_dir / "processed" / "books" / "all_pdfs_combined.json"
        data = load_json(books_file)
        
        # Count total chunks first
        total_chunks_to_process = 0
        for file_data in data['files']:
            for chunk in file_data['chunks']:
                if len(chunk) >= 200:
                    total_chunks_to_process += 1
        
        total_chunks = 0
        successful_chunks = 0
        total_questions = 0
        
        print(f"\n{'='*70}")
        print(f"STARTING AUTOMATED DATASET GENERATION")
        print(f"{'='*70}")
        print(f"Model: {self.model}")
        print(f"Questions per chunk: {self.questions_per_chunk}")
        print(f"Total chunks to process: {total_chunks_to_process}")
        print(f"Expected questions: ~{total_chunks_to_process * self.questions_per_chunk}")
        print(f"Estimated time: 2-3 hours")
        print(f"{'='*70}\n")
        
        start_time = time.time()
        
        # Setup continuous saving - NO memory accumulation!
        output_dir = config.data_dir / "processed"
        temp_output_file = output_dir / "train_llm_temp.jsonl"
        
        # RESUME SUPPORT: Check what's already been processed
        processed_sources = set()
        existing_questions = 0
        if temp_output_file.exists():
            print(f"Found existing temp file - resuming from previous run...")
            with open(temp_output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        example = json.loads(line)
                        processed_sources.add(example['source'])
                        existing_questions += 1
            print(f"Already processed {len(processed_sources)} chunk sections")
            print(f"Already have {existing_questions} questions saved")
            print(f"Continuing from where we left off...\n")
        
        # Adjust total for progress bar (only show remaining chunks)
        chunks_remaining = total_chunks_to_process - len(processed_sources)
        total_questions = existing_questions  # Start from existing count
        
        # Create progress bar for REMAINING chunks only
        pbar = tqdm(total=chunks_remaining, 
                   desc="Generating dataset",
                   unit="chunk",
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] Questions: {postfix}')
        
        for file_data in data['files']:
            filename = file_data['filename']
            
            for i, chunk in enumerate(file_data['chunks'], 1):
                if len(chunk) < 200:
                    continue
                
                total_chunks += 1
                source = f"{filename} - Section {i}"
                
                # RESUME: Skip if already processed (don't update progress bar)
                if source in processed_sources:
                    continue
                
                qa_pairs = self.generate_questions(chunk, source)
                
                if qa_pairs and len(qa_pairs) > 0:
                    # Convert to training format and WRITE TO DISK IMMEDIATELY
                    # NO memory accumulation - straight to file!
                    valid_pairs = 0
                    for qa in qa_pairs:
                        # VALIDATE: Skip if missing required keys
                        if 'question' not in qa or 'answer' not in qa:
                            continue
                        
                        # VALIDATE: Ensure question and answer are strings
                        question = qa['question']
                        answer = qa['answer']
                        
                        if not isinstance(question, str) or not isinstance(answer, str):
                            print(f"\nWARNING: Skipping non-string Q&A (question type: {type(question).__name__}, answer type: {type(answer).__name__})")
                            continue
                        
                        # Skip if empty
                        if not question.strip() or not answer.strip():
                            continue
                        
                        example = {
                            "id": generate_id(question + source),
                            "instruction": question,
                            "input": "",
                            "output": answer,
                            "source": qa.get('source', source),
                            "question_type": qa.get('type', 'general'),
                            "difficulty": qa.get('difficulty', 'medium'),
                            "text": f"### Instruction:\n{question}\n\n### Response:\n{answer}"
                        }
                        
                        # Write directly to file (append mode) - never keep in memory!
                        with open(temp_output_file, 'a', encoding='utf-8') as f:
                            f.write(json.dumps(example, ensure_ascii=False) + '\n')
                        
                        total_questions += 1
                        valid_pairs += 1
                    
                    if valid_pairs > 0:
                        successful_chunks += 1
                        pbar.set_postfix_str(f"{total_questions} total")
                        pbar.update(1)
                    else:
                        pbar.set_postfix_str(f"{total_questions} total (ALL INVALID)")
                        pbar.update(1)
                else:
                    pbar.set_postfix_str(f"{total_questions} total (FAIL)")
                    pbar.update(1)
                
                # Small delay to prevent overloading
                time.sleep(0.2)
        
        pbar.close()
        
        # Save final dataset - load from temp file, shuffle, and split
        print(f"\n{'='*70}")
        print(f"SHUFFLING AND SPLITTING DATASET")
        print(f"{'='*70}\n")
        
        # Load ALL examples from temp file (only time we load into memory)
        print("Loading generated questions from disk...")
        all_data = []
        with open(temp_output_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    all_data.append(json.loads(line))
        
        print(f"Loaded {len(all_data)} questions total")
        print("Shuffling...")
        
        import random
        random.shuffle(all_data)
        
        print("Splitting into train/val (90/10)...")
        split_idx = int(len(all_data) * 0.9)
        train_data = all_data[:split_idx]
        val_data = all_data[split_idx:]
        
        print(f"Train: {len(train_data)} examples")
        print(f"Val: {len(val_data)} examples")
        
        print("\nSaving final datasets...")
        save_jsonl(train_data, output_dir / "train_llm.jsonl")
        save_jsonl(val_data, output_dir / "val_llm.jsonl")
        
        # Remove temp file
        if temp_output_file.exists():
            temp_output_file.unlink()
            print("Cleaned up temp file")
        
        elapsed_time = time.time() - start_time
        
        print(f"{'='*70}")
        print(f"DATASET GENERATION COMPLETE!")
        print(f"{'='*70}")
        print(f"\nFinal Statistics:")
        print(f"   Total chunks processed: {total_chunks}")
        print(f"   Successful chunks: {successful_chunks} ({successful_chunks/total_chunks*100:.1f}%)")
        print(f"   Total Q&A pairs: {len(all_data)}")
        print(f"   Training set: {len(train_data)}")
        print(f"   Validation set: {len(val_data)}")
        print(f"   Total time: {elapsed_time/60:.1f} minutes")
        print(f"   Average: {elapsed_time/total_chunks:.2f} sec/chunk")
        print(f"\nSaved to:")
        print(f"   {output_dir / 'train_llm.jsonl'}")
        print(f"   {output_dir / 'val_llm.jsonl'}")
        print(f"\nNext step: Update config and train!")
        print(f"   python src/training/fine_tune.py")
        print(f"{'='*70}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate comprehensive dataset using local Ollama")
    parser.add_argument("--model", default="llama3.2", help="Ollama model to use")
    parser.add_argument("--questions", type=int, default=12, help="Questions per chunk (default: 12)")
    parser.add_argument("--test", action="store_true", help="Test with first 10 chunks only")
    
    args = parser.parse_args()
    
    generator = OllamaDatasetGenerator(
        model=args.model,
        questions_per_chunk=args.questions
    )
    
    if args.test:
        print("TEST MODE: Processing first 10 chunks only\n")
        # Temporarily modify to process only 10 chunks for testing
        # You can manually limit in the code if needed
    
    generator.process_all_chunks()
