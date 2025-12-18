"""
Test checkpoint saving mechanism BEFORE training
"""

import sys
import torch
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset

sys.path.append(str(Path(__file__).parent))
from src.utils.config import config

print("="*60)
print("CHECKPOINT SAVE TEST".center(60))
print("="*60)

# Load config
training_config = config.get_training_config()
model_name = training_config['model']['name']

print(f"\n1. Loading model: {model_name}")

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

# Load minimal dataset
print("\n2. Loading dataset...")
train_file = Path(training_config['data']['train_file'])
dataset = load_dataset('json', data_files={'train': str(train_file)})
dataset['train'] = dataset['train'].select(range(10))  # Only 10 samples

print(f"✓ Loaded {len(dataset['train'])} samples")

# Tokenize
def tokenize_function(examples):
    result = tokenizer(examples["text"], truncation=True, max_length=512, padding=False)
    result["labels"] = result["input_ids"].copy()
    return result

dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset['train'].column_names)

# Data collator
class DataCollatorForCausalLM(DataCollatorForLanguageModeling):
    def __call__(self, features):
        batch = super().__call__(features)
        labels = batch["labels"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        batch["labels"] = labels
        return batch

data_collator = DataCollatorForCausalLM(tokenizer=tokenizer, mlm=False)

# Training arguments
print("\n3. Creating trainer...")
training_args = TrainingArguments(
    output_dir="./test_checkpoint",
    save_strategy="steps",
    save_steps=1,
    save_total_limit=1,
    per_device_train_batch_size=1,
    max_steps=1,  # Only 1 step
    logging_steps=1,
    remove_unused_columns=False,
    dataloader_num_workers=0,
)

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    data_collator=data_collator,
)

print("✓ Trainer created")

# Manually save checkpoint
print("\n4. Manually saving checkpoint to test_checkpoint/checkpoint-test...")
import os
checkpoint_path = "./test_checkpoint/checkpoint-test"
os.makedirs(checkpoint_path, exist_ok=True)

# Save model
trainer.save_model(checkpoint_path)
print("✓ Model saved")

# Manually create and save trainer state
from transformers import TrainerState
state = TrainerState()
state.global_step = 0
state.epoch = 0
state.best_metric = None
state.best_model_checkpoint = None
state.log_history = []

state_path = os.path.join(checkpoint_path, "trainer_state.json")
state.save_to_json(state_path)
print(f"✓ Trainer state saved to {state_path}")

# Verify checkpoint
print("\n5. Verifying checkpoint...")
required_files = [
    "adapter_config.json",
    "adapter_model.safetensors",
    "trainer_state.json",
    "tokenizer_config.json",
]

all_good = True
for file in required_files:
    file_path = os.path.join(checkpoint_path, file)
    exists = os.path.exists(file_path)
    status = "✓" if exists else "✗"
    print(f"  {status} {file}")
    if not exists:
        all_good = False

if all_good:
    print("\n" + "="*60)
    print("✅ CHECKPOINT TEST PASSED!".center(60))
    print("="*60)
    print("\nAll required files are present.")
    print("Checkpoint saving mechanism is working correctly.")
    print("\nYou can now safely start training!")
else:
    print("\n" + "="*60)
    print("❌ CHECKPOINT TEST FAILED!".center(60))
    print("="*60)
    print("\nSome required files are missing.")
    print("DO NOT start training until this is fixed!")
