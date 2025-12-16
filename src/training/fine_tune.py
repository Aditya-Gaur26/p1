"""
Fine-tune Qwen3 model using LoRA/PEFT
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
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import argparse

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.config import config


def load_model_and_tokenizer(model_name: str, training_config: dict):
    """Load base model and tokenizer with quantization"""
    
    # Configure quantization
    bnb_config = None
    if training_config.get('quantization', {}).get('load_in_8bit', False):
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.float16
        )
    elif training_config.get('quantization', {}).get('load_in_4bit', False):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type=training_config['quantization'].get('bnb_4bit_quant_type', 'nf4'),
            bnb_4bit_use_double_quant=True
        )
    
    print(f"Loading model: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=training_config['model'].get('trust_remote_code', True),
        cache_dir=training_config['model'].get('cache_dir', './models/base')
    )
    
    # Set pad token if not exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=training_config['model'].get('trust_remote_code', True),
        cache_dir=training_config['model'].get('cache_dir', './models/base'),
        torch_dtype=torch.bfloat16 if training_config['training'].get('bf16', False) else torch.float16
    )
    
    # Prepare model for training
    if bnb_config:
        model = prepare_model_for_kbit_training(model)
    
    print("✓ Model and tokenizer loaded successfully")
    
    return model, tokenizer


def setup_lora(model, lora_config: dict):
    """Setup LoRA configuration"""
    
    peft_config = LoraConfig(
        r=lora_config.get('r', 16),
        lora_alpha=lora_config.get('lora_alpha', 32),
        target_modules=lora_config.get('target_modules', ["q_proj", "k_proj", "v_proj", "o_proj"]),
        lora_dropout=lora_config.get('lora_dropout', 0.1),
        bias=lora_config.get('bias', 'none'),
        task_type=lora_config.get('task_type', 'CAUSAL_LM')
    )
    
    model = get_peft_model(model, peft_config)
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"✓ LoRA configured")
    print(f"  Trainable params: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    print(f"  Total params: {total_params:,}")
    
    return model


def load_training_data(train_file: Path, val_file: Path):
    """Load training and validation datasets"""
    
    if not train_file.exists():
        raise FileNotFoundError(f"Training file not found: {train_file}")
    
    # Load datasets
    dataset = load_dataset(
        'json',
        data_files={
            'train': str(train_file),
            'validation': str(val_file) if val_file.exists() else None
        }
    )
    
    print(f"✓ Loaded datasets:")
    print(f"  Training samples: {len(dataset['train'])}")
    if 'validation' in dataset:
        print(f"  Validation samples: {len(dataset['validation'])}")
    
    return dataset


def train_model(args):
    """Main training function"""
    
    print("=" * 60)
    print("Fine-tuning Qwen3 with LoRA".center(60))
    print("=" * 60)
    
    # Load configuration
    training_config = config.get_training_config()
    
    # Override with command line arguments
    if args.model:
        training_config['model']['name'] = args.model
    
    model_name = training_config['model']['name']
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name, training_config)
    
    # Setup LoRA
    model = setup_lora(model, training_config['lora'])
    
    # Load datasets
    train_file = Path(training_config['data']['train_file'])
    val_file = Path(training_config['data']['val_file'])
    dataset = load_training_data(train_file, val_file)
    
    # Training arguments
    train_args = training_config['training']
    training_args = TrainingArguments(
        output_dir=train_args['output_dir'],
        num_train_epochs=train_args['num_train_epochs'],
        per_device_train_batch_size=train_args['per_device_train_batch_size'],
        per_device_eval_batch_size=train_args.get('per_device_eval_batch_size', 4),
        gradient_accumulation_steps=train_args['gradient_accumulation_steps'],
        gradient_checkpointing=train_args.get('gradient_checkpointing', True),
        learning_rate=train_args['learning_rate'],
        weight_decay=train_args.get('weight_decay', 0.001),
        warmup_ratio=train_args.get('warmup_ratio', 0.03),
        lr_scheduler_type=train_args.get('lr_scheduler_type', 'cosine'),
        optim=train_args.get('optim', 'paged_adamw_8bit'),
        max_grad_norm=train_args.get('max_grad_norm', 0.3),
        fp16=train_args.get('fp16', False),
        bf16=train_args.get('bf16', True),
        logging_steps=train_args.get('logging_steps', 10),
        save_steps=train_args.get('save_steps', 100),
        save_total_limit=train_args.get('save_total_limit', 3),
        evaluation_strategy=train_args.get('evaluation_strategy', 'steps'),
        eval_steps=train_args.get('eval_steps', 100),
        load_best_model_at_end=train_args.get('load_best_model_at_end', True),
        metric_for_best_model=train_args.get('metric_for_best_model', 'eval_loss'),
        report_to=["tensorboard"],
        logging_dir=f"{train_args['output_dir']}/logs",
        seed=training_config.get('seed', 42),
    )
    
    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset.get('validation'),
        tokenizer=tokenizer,
        max_seq_length=train_args.get('max_seq_length', 2048),
        dataset_text_field=train_args.get('dataset_text_field', 'text'),
        packing=train_args.get('packing', False),
    )
    
    # Start training
    print("\n🚀 Starting training...")
    print(f"Model: {model_name}")
    print(f"Output: {train_args['output_dir']}\n")
    
    trainer.train()
    
    # Save final model
    print("\n💾 Saving final model...")
    trainer.save_model()
    tokenizer.save_pretrained(train_args['output_dir'])
    
    print("\n✅ Training completed successfully!")
    print(f"📁 Model saved to: {train_args['output_dir']}")
    print(f"\nTo view training logs:")
    print(f"  tensorboard --logdir {train_args['output_dir']}/logs")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen3 model")
    parser.add_argument("--model", type=str, help="Model name (overrides config)")
    parser.add_argument("--config", type=str, help="Path to config file")
    
    args = parser.parse_args()
    
    try:
        train_model(args)
    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
