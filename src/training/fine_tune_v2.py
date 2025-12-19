"""
Continue fine-tuning from already fine-tuned model (Version 2)
Loads models/fine_tuned and trains further on new dataset
Saves to models/fine_tuned_v2
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
    TrainerCallback
)
from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training
import argparse
import os
import json

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.config import config


class SafeCheckpointCallback(TrainerCallback):
    """Custom callback to handle checkpoint loading safely"""
    
    def on_save(self, args, state, control, **kwargs):
        """Called after checkpoint save - clean up unsafe files if save_only_model=True"""
        if args.save_only_model:
            checkpoint_folder = os.path.join(
                args.output_dir, 
                f"checkpoint-{state.global_step}"
            )
            # Remove any .pt/.pth files that might have been saved
            for file in Path(checkpoint_folder).glob("*.pt"):
                if file.exists():
                    file.unlink()
                    print(f"  Removed unsafe file: {file.name}")
            for file in Path(checkpoint_folder).glob("*.pth"):
                if file.exists():
                    file.unlink()
                    print(f"  Removed unsafe file: {file.name}")


def load_finetuned_model_for_training(finetuned_model_path: str, training_config: dict):
    """Load already fine-tuned model for continued training"""
    
    print(f"Loading fine-tuned model from: {finetuned_model_path}")
    
    # Get base model name from config
    base_model_name = training_config['model']['name']
    
    # Configure quantization (same as original training)
    bnb_config = None
    if training_config.get('quantization', {}).get('load_in_4bit', False):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type=training_config['quantization'].get('bnb_4bit_quant_type', 'nf4'),
            bnb_4bit_use_double_quant=True
        )
    
    # Load tokenizer from fine-tuned model
    tokenizer = AutoTokenizer.from_pretrained(finetuned_model_path)
    
    # Set pad token if not exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print(f"Loading base model: {base_model_name}")
    # Load base model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        cache_dir=training_config['model'].get('cache_dir', './models/base'),
        torch_dtype=torch.bfloat16 if training_config['training'].get('bf16', False) else torch.float16
    )
    
    # Prepare model for training
    if bnb_config:
        model = prepare_model_for_kbit_training(model)
    
    print(f"Loading LoRA adapter from: {finetuned_model_path}")
    # Load the fine-tuned LoRA adapter
    model = PeftModel.from_pretrained(model, finetuned_model_path, is_trainable=True)
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print("Fine-tuned model loaded and ready for continued training")
    print(f"  Trainable params: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    print(f"  Total params: {total_params:,}")
    
    return model, tokenizer


def tokenize_function(examples, tokenizer, max_length=2048):
    """Tokenize text examples"""
    result = tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding=False,  # Dynamic padding in data collator
    )
    result["labels"] = result["input_ids"].copy()
    return result


def load_training_data(train_file: Path, val_file: Path, tokenizer, max_eval_samples=None):
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
    
    # Limit validation samples for faster evaluation
    if max_eval_samples and 'validation' in dataset and len(dataset['validation']) > max_eval_samples:
        dataset['validation'] = dataset['validation'].select(range(max_eval_samples))
    
    print(f"Loaded datasets:")
    print(f"  Training samples: {len(dataset['train'])}")
    if 'validation' in dataset:
        print(f"  Validation samples: {len(dataset['validation'])}")
    
    # Tokenize datasets
    print("\nTokenizing datasets...")
    tokenized_dataset = dataset.map(
        lambda examples: tokenize_function(examples, tokenizer, max_length=2048),
        batched=True,
        remove_columns=dataset['train'].column_names,
    )
    
    return tokenized_dataset


def train_model_v2(args):
    """Main training function for version 2 (continued fine-tuning)"""
    
    print("=" * 60)
    print("Continue Fine-tuning (V2)".center(60))
    print("=" * 60)
    
    # Load configuration
    training_config = config.get_training_config()
    
    # V2 specific paths
    finetuned_model_path = args.source_model or "./models/fine_tuned"
    output_dir = args.output or "./models/fine_tuned_v2"
    train_file = Path(args.train_data or "./data/processed/train_finetune2.jsonl")
    val_file = Path(args.val_data or "./data/processed/val_finetune2.jsonl")
    
    print(f"\nSource model: {finetuned_model_path}")
    print(f"Output directory: {output_dir}")
    print(f"Training data: {train_file}")
    print(f"Validation data: {val_file}\n")
    
    # Load already fine-tuned model
    model, tokenizer = load_finetuned_model_for_training(finetuned_model_path, training_config)
    
    # Load new training datasets
    max_eval_samples = training_config['training'].get('max_eval_samples')
    dataset = load_training_data(train_file, val_file, tokenizer, max_eval_samples)
    
    # Training arguments (same as V1 but different output dir)
    train_args = training_config['training']
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=train_args['num_train_epochs'],
        per_device_train_batch_size=train_args['per_device_train_batch_size'],
        per_device_eval_batch_size=train_args['per_device_eval_batch_size'],
        gradient_accumulation_steps=train_args['gradient_accumulation_steps'],
        gradient_checkpointing=train_args['gradient_checkpointing'],
        learning_rate=train_args['learning_rate'],
        weight_decay=train_args['weight_decay'],
        warmup_ratio=train_args['warmup_ratio'],
        lr_scheduler_type=train_args['lr_scheduler_type'],
        optim=train_args['optim'],
        max_grad_norm=train_args['max_grad_norm'],
        fp16=train_args['fp16'],
        bf16=train_args['bf16'],
        logging_steps=train_args['logging_steps'],
        save_strategy="steps",
        save_steps=train_args.get('save_steps', 200),
        save_total_limit=train_args.get('save_total_limit', 2),
        save_safetensors=True,
        save_only_model=True,  # CVE-2025-32434 mitigation
        eval_strategy="steps",
        eval_steps=train_args['eval_steps'],
        load_best_model_at_end=False,
        report_to=["tensorboard"],
        logging_dir=f"{output_dir}/logs",
        seed=training_config.get('seed', 42),
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )
    
    # Data collator with dynamic padding
    from transformers import DataCollatorForLanguageModeling
    
    class DataCollatorForCausalLM(DataCollatorForLanguageModeling):
        def __call__(self, features):
            batch = super().__call__(features)
            labels = batch["labels"].clone()
            labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
            return batch
    
    data_collator = DataCollatorForCausalLM(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Initialize custom callback for safe checkpointing
    safe_checkpoint_callback = SafeCheckpointCallback()
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset.get('validation'),
        data_collator=data_collator,
        callbacks=[safe_checkpoint_callback],
    )
    
    # Start training
    print("\n" + "="*60)
    print("Starting continued training (V2)...")
    print(f"Source: {finetuned_model_path}")
    print(f"Output: {output_dir}")
    print(f"Safe mode: Optimizer states won't be saved (CVE-2025-32434 mitigation)")
    
    # Auto-detect and resume from latest checkpoint in V2 output dir
    output_path = Path(output_dir)
    checkpoint_dirs = [d for d in output_path.glob("checkpoint-*") if d.is_dir()]
    
    resume_checkpoint = None
    if checkpoint_dirs:
        latest_checkpoint = max(checkpoint_dirs, key=lambda x: int(x.name.split("-")[1]))
        required_files = ["adapter_model.safetensors", "trainer_state.json"]
        checkpoint_valid = all((latest_checkpoint / f).exists() for f in required_files)
        
        if checkpoint_valid:
            resume_checkpoint = str(latest_checkpoint)
            print(f"Resuming from V2 checkpoint: {latest_checkpoint.name}")
            print(f"   Note: Optimizer state will be reset")
        else:
            print(f"Found checkpoint {latest_checkpoint.name} but missing required files")
            print(f"Starting fresh V2 training")
    else:
        print("Starting fresh V2 training from V1 model")
    
    print("="*60 + "\n")
    
    # Train with auto-resume
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    
    # Save final model
    print("\nSaving final V2 model...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    print("\nV2 Training completed successfully!")
    print(f"V2 Model saved to: {output_dir}")
    print(f"\nTo view training logs:")
    print(f"  tensorboard --logdir {output_dir}/logs")


def main():
    parser = argparse.ArgumentParser(description="Continue fine-tuning from V1 model (V2)")
    parser.add_argument("--source-model", type=str, default="./models/fine_tuned",
                       help="Path to V1 fine-tuned model (default: ./models/fine_tuned)")
    parser.add_argument("--output", type=str, default="./models/fine_tuned_v2",
                       help="Output directory for V2 model (default: ./models/fine_tuned_v2)")
    parser.add_argument("--train-data", type=str, default="./data/processed/train_finetune2.jsonl",
                       help="Training data for V2")
    parser.add_argument("--val-data", type=str, default="./data/processed/val_finetune2.jsonl",
                       help="Validation data for V2")
    
    args = parser.parse_args()
    
    try:
        train_model_v2(args)
    except Exception as e:
        print(f"\nERROR: V2 Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
