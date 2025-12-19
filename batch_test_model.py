"""
Batch test the fine-tuned model on a set of questions and save results
"""

import sys
import json
import torch
from pathlib import Path
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm

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


def batch_test_questions(questions_file: str, output_file: str, model_path: str = "./models/fine_tuned"):
    """Test model on a batch of questions and save results"""
    
    # Load questions
    print(f"Loading questions from: {questions_file}")
    with open(questions_file, 'r', encoding='utf-8') as f:
        questions_data = json.load(f)
    
    print(f"Loaded {len(questions_data)} questions\n")
    
    # Load model
    model, tokenizer = load_finetuned_model(model_path)
    
    # Process each question
    results = []
    total_input_tokens = 0
    total_output_tokens = 0
    
    print("=" * 70)
    print("BATCH TESTING IN PROGRESS".center(70))
    print("=" * 70)
    print()
    
    for i, item in enumerate(tqdm(questions_data, desc="Processing questions"), 1):
        question = item['question']
        
        # Generate response
        try:
            model_answer, token_info = generate_response(model, tokenizer, question)
            
            # Accumulate token statistics
            total_input_tokens += token_info['input_tokens']
            total_output_tokens += token_info['output_tokens']
            
            # Store result
            result = {
                'question_id': i,
                'question': question,
                'expected_answer': item.get('answer', ''),
                'model_answer': model_answer,
                'topic': item.get('topic', ''),
                'difficulty': item.get('difficulty', ''),
                'token_usage': token_info
            }
            results.append(result)
            
        except Exception as e:
            print(f"\n❌ Error processing question {i}: {str(e)}")
            results.append({
                'question_id': i,
                'question': question,
                'expected_answer': item.get('answer', ''),
                'model_answer': f"ERROR: {str(e)}",
                'topic': item.get('topic', ''),
                'difficulty': item.get('difficulty', ''),
                'token_usage': {'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0}
            })
    
    # Calculate statistics
    avg_input_tokens = total_input_tokens / len(questions_data)
    avg_output_tokens = total_output_tokens / len(questions_data)
    avg_total_tokens = (total_input_tokens + total_output_tokens) / len(questions_data)
    
    # Create summary
    summary = {
        'test_date': datetime.now().isoformat(),
        'model_path': model_path,
        'total_questions': len(questions_data),
        'total_input_tokens': total_input_tokens,
        'total_output_tokens': total_output_tokens,
        'total_tokens': total_input_tokens + total_output_tokens,
        'avg_input_tokens_per_question': round(avg_input_tokens, 2),
        'avg_output_tokens_per_question': round(avg_output_tokens, 2),
        'avg_total_tokens_per_question': round(avg_total_tokens, 2)
    }
    
    # Save results
    output_data = {
        'summary': summary,
        'results': results
    }
    
    print("\n" + "=" * 70)
    print("BATCH TESTING COMPLETE".center(70))
    print("=" * 70)
    print(f"\n📊 Summary:")
    print(f"   Total questions: {summary['total_questions']}")
    print(f"   Total tokens used: {summary['total_tokens']:,}")
    print(f"   - Input tokens: {summary['total_input_tokens']:,}")
    print(f"   - Output tokens: {summary['total_output_tokens']:,}")
    print(f"\n   Average per question:")
    print(f"   - Input: {summary['avg_input_tokens_per_question']:.1f} tokens")
    print(f"   - Output: {summary['avg_output_tokens_per_question']:.1f} tokens")
    print(f"   - Total: {summary['avg_total_tokens_per_question']:.1f} tokens")
    
    # Save to file
    print(f"\n💾 Saving results to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Results saved successfully!")
    print(f"\nTo view results:")
    print(f"  python -m json.tool {output_file}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch test fine-tuned model")
    parser.add_argument("--questions", type=str, default="test_questions.json",
                       help="Path to questions JSON file")
    parser.add_argument("--output", type=str, default="test_results_def.json",
                       help="Path to save results JSON")
    parser.add_argument("--model", type=str, default="./models/fine_tuned",
                       help="Path to fine-tuned model")
    
    args = parser.parse_args()
    
    try:
        batch_test_questions(args.questions, args.output, args.model)
    except Exception as e:
        print(f"\n❌ Batch testing failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
