"""
Test the fine-tuned model with inference
"""

import sys
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.config import config


def load_finetuned_model(model_path: str = "./models/fine_tuned"):
    """Load the fine-tuned model with LoRA adapter"""
    
    print("Loading fine-tuned model...")
    print(f"Model path: {model_path}\n")
    
    # Get base model name from config
    training_config = config.get_training_config()
    base_model_name = training_config['model']['name']
    
    # Configure quantization (same as training)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load base model
    print(f"Loading base model: {base_model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )
    
    # Load LoRA adapter
    print(f"Loading LoRA adapter from: {model_path}")
    model = PeftModel.from_pretrained(model, model_path)
    
    print("✓ Model loaded successfully\n")
    
    return model, tokenizer


def generate_response(model, tokenizer, instruction: str, max_new_tokens: int = 512):
    """Generate response for a given instruction"""
    
    # Format the prompt (same format as training data)
    prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_length = inputs['input_ids'].shape[1]
    
    # Generate
    print("Generating response...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Calculate token usage
    total_tokens = outputs[0].shape[0]
    output_tokens = total_tokens - input_length
    
    # Extract only the response part (after "### Response:")
    if "### Response:" in response:
        response = response.split("### Response:")[1].strip()
    
    # Return response and token info
    token_info = {
        'input_tokens': input_length,
        'output_tokens': output_tokens,
        'total_tokens': total_tokens
    }
    
    return response, token_info


def main():
    """Interactive inference loop"""
    
    # Load model
    model, tokenizer = load_finetuned_model()
    
    print("=" * 70)
    print("Fine-tuned Model Inference".center(70))
    print("=" * 70)
    print("\nAsk questions about Operating Systems or Computer Networks!")
    print("Type 'exit' or 'quit' to stop.\n")
    print("-" * 70)
    
    while True:
        # Get user input
        instruction = input("\n📝 Your question: ").strip()
        
        if instruction.lower() in ['exit', 'quit', 'q']:
            print("\n👋 Goodbye!")
            break
        
        if not instruction:
            continue
        
        # Generate response
        print("\n🤖 Model response:")
        print("-" * 70)
        response, token_info = generate_response(model, tokenizer, instruction)
        print(response)
        print("-" * 70)
        print(f"\n📊 Token usage: {token_info['input_tokens']} input + {token_info['output_tokens']} output = {token_info['total_tokens']} total tokens")


if __name__ == "__main__":
    # Example questions for quick testing
    example_questions = [
        "What is a protocol?",
        "Explain the OSI model",
        "What is the difference between TCP and UDP?",
        "What is a deadlock in operating systems?",
        "Explain virtual memory"
    ]
    
    print("\n💡 Example questions you can ask:")
    for i, q in enumerate(example_questions, 1):
        print(f"   {i}. {q}")
    print()
    
    main()
