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
    BitsAndBytesConfig,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
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
    
    print(f"✓ Loaded datasets:")
    print(f"  Training samples: {len(dataset['train'])}")
    if 'validation' in dataset:
        print(f"  Validation samples: {len(dataset['validation'])}")
    
    # Tokenize datasets
    print("\n⚙️  Tokenizing datasets...")
    tokenized_dataset = dataset.map(
        lambda examples: tokenize_function(examples, tokenizer, max_length=2048),
        batched=True,
        remove_columns=dataset['train'].column_names,
    )
    
    return tokenized_dataset


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
    max_eval_samples = training_config['training'].get('max_eval_samples')
    dataset = load_training_data(train_file, val_file, tokenizer, max_eval_samples)
    
    # Training arguments
    train_args = training_config['training']
    training_args = TrainingArguments(
        output_dir=train_args['output_dir'],
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
        save_strategy="steps",  # Enable checkpointing
        save_steps=train_args.get('save_steps', 200),
        save_total_limit=train_args.get('save_total_limit', 2),
        save_safetensors=True,  # Save model weights in safetensors format
        save_only_model=True,   # FIX: Skip optimizer/scheduler states to avoid CVE-2025-32434
        eval_strategy="steps",
        eval_steps=train_args['eval_steps'],
        load_best_model_at_end=False,
        report_to=["tensorboard"],
        logging_dir=f"{train_args['output_dir']}/logs",
        seed=training_config.get('seed', 42),
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )
    
    # Data collator with dynamic padding
    from transformers import DataCollatorForLanguageModeling
    
    # Custom data collator that pads and sets padding tokens in labels to -100
    class DataCollatorForCausalLM(DataCollatorForLanguageModeling):
        def __call__(self, features):
            # Pad sequences dynamically to the longest in the batch
            batch = super().__call__(features)
            # Replace padding token id in labels with -100 to ignore in loss
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
    
    # Initialize trainer - using standard Trainer instead of SFTTrainer
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
    print("🚀 Starting training...")
    print(f"Model: {model_name}")
    print(f"Output: {train_args['output_dir']}")
    print(f"⚠️  Safe mode: Optimizer states won't be saved (CVE-2025-32434 mitigation)")
    
    # Auto-detect and resume from latest checkpoint
    output_dir = Path(train_args['output_dir'])
    checkpoint_dirs = [d for d in output_dir.glob("checkpoint-*") if d.is_dir()]
    
    resume_checkpoint = None
    if checkpoint_dirs:
        # Sort by step number and get latest
        latest_checkpoint = max(checkpoint_dirs, key=lambda x: int(x.name.split("-")[1]))
        
        # Verify checkpoint has required files
        required_files = ["adapter_model.safetensors", "trainer_state.json"]
        checkpoint_valid = all((latest_checkpoint / f).exists() for f in required_files)
        
        if checkpoint_valid:
            resume_checkpoint = str(latest_checkpoint)
            print(f"📂 Resuming from checkpoint: {latest_checkpoint.name}")
            print(f"   Note: Optimizer state will be reset (training from checkpoint model only)")
        else:
            print(f"⚠️  Found checkpoint {latest_checkpoint.name} but missing required files")
            print(f"🎯 Starting fresh training instead")
    else:
        print("🎯 Starting fresh training")
    
    print("="*60 + "\n")
    
    # Train with auto-resume
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    
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
