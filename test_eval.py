"""
Test evaluation step to find crash issue
"""

import sys
import torch
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

sys.path.append(str(Path(__file__).parent))
from src.utils.config import config


def tokenize_function(examples, tokenizer, max_length=2048):
    """Tokenize text examples"""
    result = tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    result["labels"] = result["input_ids"].copy()
    return result


def main():
    print("=" * 60)
    print("Testing Evaluation Step".center(60))
    print("=" * 60)
    
    # Load configuration
    training_config = config.get_training_config()
    model_name = training_config['model']['name']
    
    print(f"\nLoading model: {model_name}")
    
    # Configure quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        cache_dir=training_config['model']['cache_dir']
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir=training_config['model']['cache_dir']
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare model
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    
    # Setup LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    
    print("✓ Model loaded with LoRA")
    
    # Load validation dataset
    train_file = Path(training_config['data']['train_file'])
    val_file = Path(training_config['data']['val_file'])
    
    dataset = load_dataset(
        'json',
        data_files={
            'train': str(train_file),
            'validation': str(val_file)
        }
    )
    
    print(f"✓ Loaded validation dataset: {len(dataset['validation'])} samples")
    
    # Tokenize
    print("\nTokenizing validation set...")
    tokenized_dataset = dataset['validation'].map(
        lambda examples: tokenize_function(examples, tokenizer, max_length=2048),
        batched=True,
        remove_columns=dataset['validation'].column_names,
    )
    
    print(f"✓ Tokenized: {len(tokenized_dataset)} samples")
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Training arguments (minimal for eval only)
    training_args = TrainingArguments(
        output_dir="./test_eval_output",
        per_device_eval_batch_size=1,
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    
    print("\n" + "=" * 60)
    print("Running evaluation (this is what happens at step 100)...")
    print("=" * 60 + "\n")
    
    try:
        # This is what crashes at step 100
        eval_results = trainer.evaluate()
        print("\n✅ Evaluation completed successfully!")
        print(f"Results: {eval_results}")
    except Exception as e:
        print(f"\n❌ Evaluation FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    if success:
        print("\n" + "=" * 60)
        print("✅ No issue found - evaluation works fine!".center(60))
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("❌ Found the crash issue!".center(60))
        print("=" * 60)
