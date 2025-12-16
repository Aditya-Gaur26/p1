"""
Model loader for fine-tuned Qwen3 model
"""

import sys
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.config import config


class ModelLoader:
    """Load and manage fine-tuned model"""
    
    def __init__(self, base_model_name: str = None, adapter_path: str = None):
        model_config = config.get_model_config()
        
        self.base_model_name = base_model_name or model_config['model']['base_model']
        self.adapter_path = Path(adapter_path or model_config['model']['adapter_path'])
        self.device = model_config['model'].get('device', 'auto')
        
        self.tokenizer = None
        self.model = None
        self.generation_config = model_config.get('generation', {})
    
    def load(self, use_adapter: bool = True):
        """Load model and tokenizer"""
        
        print(f"Loading model: {self.base_model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            trust_remote_code=True
        )
        
        # Load adapter if fine-tuned model exists
        if use_adapter and self.adapter_path.exists():
            print(f"Loading fine-tuned adapter from: {self.adapter_path}")
            self.model = PeftModel.from_pretrained(self.model, str(self.adapter_path))
            print("✓ Fine-tuned model loaded")
        else:
            if use_adapter:
                print("⚠️  Fine-tuned adapter not found, using base model only")
            else:
                print("✓ Base model loaded (no adapter)")
        
        self.model.eval()
        
        print("✓ Model ready for inference")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate response from prompt"""
        
        if self.model is None:
            raise ValueError("Model not loaded. Call load() first.")
        
        # Merge generation config with kwargs
        gen_config = {**self.generation_config, **kwargs}
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=gen_config.get('max_new_tokens', 512),
                temperature=gen_config.get('temperature', 0.7),
                top_p=gen_config.get('top_p', 0.9),
                top_k=gen_config.get('top_k', 50),
                repetition_penalty=gen_config.get('repetition_penalty', 1.1),
                do_sample=gen_config.get('do_sample', True),
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode output
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the new generated text (remove prompt)
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        
        return generated_text
    
    def format_prompt(self, instruction: str, context: str = None) -> str:
        """Format prompt for model"""
        
        prompts_config = config.get_model_config().get('prompts', {})
        system_prompt = prompts_config.get('system_prompt', '')
        
        if context:
            template = prompts_config.get('context_template', '''### Context:
{context}

### Question:
{question}

### Answer:''')
            
            prompt = template.format(context=context, question=instruction)
        else:
            template = prompts_config.get('instruction_template', '''### Instruction:
{instruction}

### Response:''')
            
            prompt = template.format(instruction=instruction)
        
        if system_prompt:
            prompt = f"{system_prompt}\n\n{prompt}"
        
        return prompt


def main():
    """Test model loader"""
    print("=" * 60)
    print("Testing Model Loader".center(60))
    print("=" * 60)
    
    # Load model
    loader = ModelLoader()
    loader.load(use_adapter=True)
    
    # Test generation
    test_prompt = "What is a process in operating systems?"
    formatted_prompt = loader.format_prompt(test_prompt)
    
    print(f"\nPrompt: {test_prompt}")
    print("\nGenerating response...")
    
    response = loader.generate(formatted_prompt, max_new_tokens=200)
    
    print(f"\nResponse:\n{response}")


if __name__ == "__main__":
    main()
