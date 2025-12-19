# Fine-tuning Version 2 (V2) - Continued Training

This directory contains scripts for **continued fine-tuning** from the already trained V1 model.

## Overview

- **V1 Training**: Base model → Fine-tuned on `train_llm.jsonl` → Saved to `models/fine_tuned`
- **V2 Training**: V1 model → Further fine-tuned on `train_finetune2.jsonl` → Saved to `models/fine_tuned_v2`

## Key Differences from V1

| Aspect | V1 (fine_tune.py) | V2 (fine_tune_v2.py) |
|--------|-------------------|----------------------|
| **Source** | Base Qwen2.5-1.5B-Instruct | V1 fine-tuned model |
| **Training Data** | train_llm.jsonl (10,608) | train_finetune2.jsonl (varies) |
| **Validation Data** | val_llm.jsonl (1,179) | val_finetune2.jsonl (varies) |
| **Output Directory** | models/fine_tuned | models/fine_tuned_v2 |
| **Purpose** | Initial domain adaptation | Continued refinement |

## How V2 Works

1. **Loads V1 LoRA adapter** onto the base model
2. **Enables training mode** (`is_trainable=True`)
3. **Continues training** on new dataset
4. **Saves updated adapter** to separate directory

This approach allows you to:
- Build on top of V1 knowledge
- Add new training examples without retraining from scratch
- Keep V1 model intact for comparison
- Iterate on improvements

## Usage

### Method 1: PowerShell Script (Recommended)
```powershell
powershell -ExecutionPolicy Bypass -File train_v2.ps1
```

### Method 2: Batch File
```cmd
train_v2.bat
```

### Method 3: Direct Python
```bash
# Use defaults (V1 → V2 with train_finetune2.jsonl)
python src/training/fine_tune_v2.py

# Custom paths
python src/training/fine_tune_v2.py \
  --source-model ./models/fine_tuned \
  --output ./models/fine_tuned_v2 \
  --train-data ./data/processed/train_finetune2.jsonl \
  --val-data ./data/processed/val_finetune2.jsonl
```

## Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--source-model` | `./models/fine_tuned` | Path to V1 fine-tuned model |
| `--output` | `./models/fine_tuned_v2` | Output directory for V2 |
| `--train-data` | `./data/processed/train_finetune2.jsonl` | V2 training data |
| `--val-data` | `./data/processed/val_finetune2.jsonl` | V2 validation data |

## Training Configuration

V2 uses the same hyperparameters as V1 (from `configs/training_config.yaml`):
- **LoRA**: r=32, alpha=64
- **Quantization**: 4-bit NF4
- **Optimizer**: paged_adamw_8bit
- **Learning Rate**: 5e-5
- **Batch Size**: 1 (grad_accum=16)
- **Epochs**: 2
- **Safe Checkpointing**: Enabled (CVE-2025-32434 protection)

## Output Structure

```
models/
├── fine_tuned/              ← V1 model (preserved)
│   ├── adapter_model.safetensors
│   ├── adapter_config.json
│   └── ...
│
└── fine_tuned_v2/           ← V2 model (new)
    ├── adapter_model.safetensors
    ├── adapter_config.json
    ├── checkpoint-*/
    └── logs/
```

## Checkpointing

V2 supports resuming:
- Checkpoints saved every 200 steps (configurable)
- Automatic resume from latest V2 checkpoint
- Safe checkpoint format (no optimizer states)

If training is interrupted:
```bash
# Just re-run - it will auto-resume from latest V2 checkpoint
python src/training/fine_tune_v2.py
```

## Testing V2 Model

Use the batch test script with V2 model:

```bash
python batch_test_model.py \
  --model ./models/fine_tuned_v2 \
  --questions test_questions.json \
  --output test_results_v2.json
```

## Comparison: V1 vs V2

To compare V1 and V2 performance:

```bash
# Test V1
python batch_test_model.py --model ./models/fine_tuned --output results_v1.json

# Test V2  
python batch_test_model.py --model ./models/fine_tuned_v2 --output results_v2.json

# Compare results
python -c "
import json
v1 = json.load(open('results_v1.json'))
v2 = json.load(open('results_v2.json'))
print(f'V1 avg tokens: {v1[\"summary\"][\"avg_output_tokens_per_question\"]}')
print(f'V2 avg tokens: {v2[\"summary\"][\"avg_output_tokens_per_question\"]}')
"
```

## When to Use V2

Use V2 training when:
- ✅ You have new training examples to add
- ✅ V1 model needs refinement on specific topics
- ✅ You want to iterate without losing V1
- ✅ Adding domain-specific knowledge incrementally

Don't use V2 if:
- ❌ Starting completely fresh (use V1 instead)
- ❌ Datasets are incompatible in format
- ❌ V1 model is corrupted (retrain V1)

## Troubleshooting

### Issue: "Model not found"
**Solution**: Ensure V1 training completed successfully and `models/fine_tuned` exists.

### Issue: "Dataset file not found"
**Solution**: Check that `train_finetune2.jsonl` and `val_finetune2.jsonl` exist in `data/processed/`.

### Issue: "Out of memory"
**Solution**: V2 uses same memory as V1. If failing, reduce `per_device_train_batch_size` in config.

### Issue: "Different model architecture"
**Solution**: V2 source model must be from same base model (Qwen2.5-1.5B-Instruct).

## Advanced: Multiple Iterations

You can chain further:

```bash
# V3 from V2
python src/training/fine_tune_v2.py \
  --source-model ./models/fine_tuned_v2 \
  --output ./models/fine_tuned_v3 \
  --train-data ./data/processed/train_finetune3.jsonl
```

## Security

V2 maintains same security as V1:
- ✅ Safe checkpoint format (safetensors only)
- ✅ No optimizer state files (.pt/.pth)
- ✅ Protected against CVE-2025-32434

---

**Questions?** Check the main project documentation or open an issue.
