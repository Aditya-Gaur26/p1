"""
LLM-Based Question and Answer Generation
Uses OpenAI/Ollama/Qwen to generate high-quality Q&A pairs

MUCH BETTER than rule-based templates!
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import time

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.config import config


class LLMQuestionGenerator:
    """
    Generate diverse, high-quality questions using LLMs
    Supports: OpenAI API, Ollama (local), or HuggingFace
    """
    
    def __init__(self, provider: str = "ollama", model: str = "llama3.2:3b"):
        """
        Initialize LLM question generator
        
        Args:
            provider: "openai", "ollama", or "huggingface"
            model: Model name (e.g., "gpt-4", "llama3.2:3b", "meta-llama/Llama-3-8b")
        """
        self.provider = provider
        self.model = model
        
        if provider == "openai":
            self._init_openai()
        elif provider == "ollama":
            self._init_ollama()
        elif provider == "huggingface":
            self._init_huggingface()
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    def _init_openai(self):
        """Initialize OpenAI API"""
        try:
            import openai
            # Set API key from environment or config
            openai.api_key = config.get('openai_api_key', '')
            if not openai.api_key:
                raise ValueError("OpenAI API key not set! Set OPENAI_API_KEY env variable")
            self.client = openai.OpenAI()
            print(f"✓ Initialized OpenAI API with model: {self.model}")
        except ImportError:
            raise ImportError("Install openai: pip install openai")
    
    def _init_ollama(self):
        """Initialize Ollama (local LLM)"""
        try:
            import requests
            # Test Ollama connection
            response = requests.get("http://localhost:11434/api/tags")
            if response.status_code == 200:
                available_models = [m['name'] for m in response.json().get('models', [])]
                if self.model not in available_models:
                    print(f"⚠️  Model {self.model} not found. Available: {available_models}")
                    print(f"   Download with: ollama pull {self.model}")
                else:
                    print(f"✓ Initialized Ollama with model: {self.model}")
            else:
                raise ConnectionError("Ollama not running. Start with: ollama serve")
        except ImportError:
            raise ImportError("Install requests: pip install requests")
        except Exception as e:
            raise ConnectionError(f"Ollama connection failed: {e}\nStart Ollama: ollama serve")
    
    def _init_huggingface(self):
        """Initialize HuggingFace model"""
        try:
            from transformers import pipeline
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                device_map="auto",
                max_new_tokens=512
            )
            print(f"✓ Initialized HuggingFace model: {self.model}")
        except ImportError:
            raise ImportError("Install transformers: pip install transformers")
    
    def generate(self, prompt: str, max_tokens: int = 500, temperature: float = 0.7) -> str:
        """Generate text using configured LLM"""
        
        if self.provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content
        
        elif self.provider == "ollama":
            import requests
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens
                    }
                }
            )
            return response.json()['response']
        
        elif self.provider == "huggingface":
            result = self.pipe(prompt, max_new_tokens=max_tokens, temperature=temperature)
            return result[0]['generated_text']
    
    def extract_topics(self, content: str) -> List[str]:
        """
        Extract topics from content using LLM
        MUCH better than keyword matching!
        """
        
        prompt = f"""Extract the main technical topics/concepts from this text about Operating Systems or Computer Networks.
Return ONLY a Python list of topics (no explanation).

Text:
{content[:800]}

Output format: ["topic1", "topic2", "topic3"]
"""
        
        response = self.generate(prompt, max_tokens=100, temperature=0.3)
        
        # Parse response to extract list
        try:
            # Try to find JSON list in response
            import re
            match = re.search(r'\[.*?\]', response, re.DOTALL)
            if match:
                topics = eval(match.group(0))  # Safe here since we control the prompt
                return topics[:5]  # Limit to 5 topics
        except:
            pass
        
        # Fallback to line-based parsing
        topics = [line.strip(' -•*') for line in response.split('\n') if line.strip()]
        return topics[:5]
    
    def generate_questions(self, content: str, num_questions: int = 5) -> List[Dict[str, str]]:
        """
        Generate diverse questions from content
        Returns: [{"question": "...", "type": "conceptual", "difficulty": "medium"}, ...]
        """
        
        prompt = f"""You are an expert professor creating exam questions.

Content from course material:
{content[:1000]}

Generate {num_questions} diverse questions that test understanding of this content.
Include different types: conceptual, procedural, comparison, application, troubleshooting.

For each question, provide:
1. The question
2. Question type (conceptual/procedural/comparison/application/troubleshooting)
3. Difficulty (easy/medium/hard)

Format as JSON list:
[
  {{"question": "...", "type": "conceptual", "difficulty": "medium"}},
  ...
]
"""
        
        response = self.generate(prompt, max_tokens=600, temperature=0.7)
        
        # Parse JSON response
        try:
            import re
            match = re.search(r'\[.*\]', response, re.DOTALL)
            if match:
                questions = json.loads(match.group(0))
                return questions
        except:
            pass
        
        # Fallback: extract questions as simple list
        questions = []
        for line in response.split('\n'):
            if '?' in line:
                questions.append({
                    "question": line.strip(' -•*0123456789.'),
                    "type": "conceptual",
                    "difficulty": "medium"
                })
        
        return questions[:num_questions]
    
    def enhance_answer(self, question: str, raw_answer: str) -> str:
        """
        Enhance answer with better structure, examples, and reasoning
        """
        
        prompt = f"""You are a teaching assistant improving an answer for students.

Question: {question}

Current answer (from course material):
{raw_answer[:800]}

Improve this answer by:
1. Adding a brief introduction
2. Organizing into clear points
3. Adding "Step-by-step:" for procedural content
4. Including "Example:" where helpful
5. Keeping all technical details from original

Output the improved answer (no extra commentary):
"""
        
        enhanced = self.generate(prompt, max_tokens=600, temperature=0.5)
        return enhanced.strip()
    
    def add_reasoning_chain(self, question: str, answer: str) -> str:
        """
        Add chain-of-thought reasoning to answer
        Makes model learn to think step-by-step
        """
        
        prompt = f"""Add step-by-step reasoning to this answer.

Question: {question}

Answer: {answer[:600]}

Rewrite the answer with "Let me explain step by step:" followed by numbered steps.
Keep all technical details. Make it pedagogical.

Output:
"""
        
        reasoning_answer = self.generate(prompt, max_tokens=700, temperature=0.5)
        return reasoning_answer.strip()
    
    def generate_comparison_question(self, topic1: str, topic2: str, content: str) -> Dict[str, str]:
        """Generate comparison question between two topics"""
        
        prompt = f"""Create a comparison question for an Operating Systems or Networks course.

Topics to compare: {topic1} vs {topic2}

Context from course material:
{content[:600]}

Generate:
1. A specific comparison question
2. A detailed answer highlighting key differences and similarities

Format as JSON:
{{"question": "...", "answer": "..."}}
"""
        
        response = self.generate(prompt, max_tokens=500, temperature=0.6)
        
        try:
            import re
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                return json.loads(match.group(0))
        except:
            pass
        
        # Fallback
        return {
            "question": f"What is the difference between {topic1} and {topic2}?",
            "answer": content[:500]
        }
    
    def generate_refusal_examples(self, num_examples: int = 10) -> List[Dict[str, str]]:
        """
        Generate "I don't know" examples for out-of-scope questions
        CRITICAL for preventing hallucinations!
        """
        
        prompt = f"""Generate {num_examples} questions that are OUT OF SCOPE for an Operating Systems and Computer Networks course.

Examples:
- Questions about web development frameworks
- Questions about quantum computing
- Questions about biology
- General knowledge questions
- Current events

For each question, the answer should be:
"This topic is outside the scope of the Operating Systems and Computer Networks course materials. I can only answer questions based on the provided lecture content."

Format as JSON list:
[
  {{"question": "...", "answer": "This topic is outside the scope..."}},
  ...
]
"""
        
        response = self.generate(prompt, max_tokens=800, temperature=0.8)
        
        try:
            import re
            match = re.search(r'\[.*\]', response, re.DOTALL)
            if match:
                examples = json.loads(match.group(0))
                return examples[:num_examples]
        except:
            pass
        
        # Fallback
        refusal = "This topic is outside the scope of the Operating Systems and Computer Networks course materials."
        out_of_scope = [
            "What is React.js?",
            "Explain quantum entanglement",
            "Who won the World Cup?",
            "What is photosynthesis?",
            "Explain blockchain mining"
        ]
        
        return [{"question": q, "answer": refusal} for q in out_of_scope[:num_examples]]


def generate_llm_enhanced_dataset(
    content_chunks: List[Dict[str, str]],
    llm_generator: LLMQuestionGenerator,
    questions_per_chunk: int = 3
) -> List[Dict[str, Any]]:
    """
    Generate training dataset using LLM
    
    Args:
        content_chunks: [{"content": "...", "source": "..."}, ...]
        llm_generator: LLMQuestionGenerator instance
        questions_per_chunk: Number of questions per content chunk
        
    Returns:
        List of training examples
    """
    
    print("\n🤖 Generating LLM-enhanced Q&A pairs...")
    print(f"   Provider: {llm_generator.provider} | Model: {llm_generator.model}")
    
    training_data = []
    
    for i, chunk in enumerate(content_chunks):
        content = chunk['content']
        source = chunk['source']
        
        # Skip short content
        if len(content) < 100:
            continue
        
        print(f"   Processing chunk {i+1}/{len(content_chunks)}...", end='\r')
        
        try:
            # 1. Extract topics using LLM
            topics = llm_generator.extract_topics(content)
            
            # 2. Generate questions using LLM
            questions_data = llm_generator.generate_questions(content, questions_per_chunk)
            
            # 3. Create Q&A pairs
            for q_data in questions_data:
                question = q_data.get('question', '')
                if not question:
                    continue
                
                # 4. Enhance answer
                enhanced_answer = llm_generator.enhance_answer(question, content)
                
                # 5. Add reasoning chain for complex questions
                if q_data.get('type') in ['procedural', 'comparison', 'application']:
                    enhanced_answer = llm_generator.add_reasoning_chain(question, enhanced_answer)
                
                training_data.append({
                    "instruction": question,
                    "input": "",
                    "output": enhanced_answer,
                    "source": source,
                    "topics": topics,
                    "question_type": q_data.get('type', 'general'),
                    "difficulty": q_data.get('difficulty', 'medium'),
                    "text": f"### Instruction:\n{question}\n\n### Response:\n{enhanced_answer}"
                })
            
            # Rate limiting (for API calls)
            if llm_generator.provider == "openai":
                time.sleep(0.5)  # Avoid rate limits
        
        except Exception as e:
            print(f"\n⚠️  Error processing chunk {i+1}: {e}")
            continue
    
    print(f"\n✓ Generated {len(training_data)} LLM-enhanced examples")
    
    # Add refusal examples
    print("\n🚫 Adding refusal examples...")
    refusal_examples = llm_generator.generate_refusal_examples(15)
    
    for ref in refusal_examples:
        training_data.append({
            "instruction": ref['question'],
            "input": "",
            "output": ref['answer'],
            "source": "out_of_scope",
            "topics": [],
            "question_type": "refusal",
            "difficulty": "n/a",
            "text": f"### Instruction:\n{ref['question']}\n\n### Response:\n{ref['answer']}"
        })
    
    print(f"✓ Added {len(refusal_examples)} refusal examples")
    
    return training_data


def main():
    """Demo: LLM-enhanced question generation"""
    
    print("=" * 70)
    print("LLM-Enhanced Question Generation Demo".center(70))
    print("=" * 70)
    
    # Example: Use Ollama with local Llama model
    # Change to "openai" + "gpt-4" if you have OpenAI API key
    generator = LLMQuestionGenerator(provider="ollama", model="llama3.2:3b")
    
    # Sample content
    sample_content = """
    Process scheduling is a fundamental concept in operating systems. The scheduler 
    decides which process runs on the CPU and for how long. Common algorithms include:
    
    1. First-Come First-Served (FCFS): Processes are executed in the order they arrive
    2. Shortest Job First (SJF): Process with shortest execution time runs first
    3. Round Robin (RR): Each process gets a fixed time quantum
    4. Priority Scheduling: Processes have priorities, highest priority runs first
    
    Preemptive scheduling allows interrupting running processes, while non-preemptive
    waits for processes to complete. The choice depends on system requirements like
    response time, throughput, and fairness.
    """
    
    # Extract topics
    print("\n1. Extracting topics...")
    topics = generator.extract_topics(sample_content)
    print(f"   Topics: {topics}")
    
    # Generate questions
    print("\n2. Generating questions...")
    questions = generator.generate_questions(sample_content, num_questions=3)
    for i, q in enumerate(questions, 1):
        print(f"\n   Q{i}: {q['question']}")
        print(f"   Type: {q['type']} | Difficulty: {q['difficulty']}")
    
    # Enhance answer
    print("\n3. Enhancing answer...")
    if questions:
        enhanced = generator.enhance_answer(questions[0]['question'], sample_content)
        print(f"   {enhanced[:200]}...")
    
    # Generate refusal examples
    print("\n4. Generating refusal examples...")
    refusals = generator.generate_refusal_examples(3)
    for i, ref in enumerate(refusals, 1):
        print(f"\n   Q{i}: {ref['question']}")
        print(f"   A: {ref['answer'][:80]}...")
    
    print("\n" + "=" * 70)
    print("✓ Demo complete!")
    print("\nTo use in your pipeline:")
    print("  1. Edit src/data_processing/create_dataset.py")
    print("  2. Import: from src.data_processing.llm_augmentation import LLMQuestionGenerator")
    print("  3. Replace rule-based generation with LLM generation")
    print("=" * 70)


if __name__ == "__main__":
    main()
