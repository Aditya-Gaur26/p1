# V2 Training Quick Start

## 🎯 What This Does

Continues training from your **already fine-tuned V1 model** on new data while keeping V1 completely safe and unchanged.

## ✅ Safety Guarantees

1. **V1 Model Protected**: Read-only during training, restored after
2. **Separate Output**: V2 saves to `models/fine_tuned_v2` (never touches V1)
3. **Safe Checkpointing**: Uses safetensors format only (no .pt/.pth files)
4. **Auto-Resume**: Picks up from latest V2 checkpoint if interrupted

## 🚀 Start V2 Training

```powershell
# Recommended: High-priority training with power management
powershell -ExecutionPolicy Bypass -File start_training_v2.ps1
```

**OR** manual:
```bash
python src/training/fine_tune_v2.py
```

## 📋 Before You Start

Run safety check:
```bash
python verify_v2_safety.py
```

Should see:
```
✅ ALL SAFETY CHECKS PASSED

V2 Training Safety Guarantees:
  ✓ V1 model will NOT be modified
  ✓ V2 saves to completely separate directory
  ✓ Checkpoints use safe format (safetensors only)
  ✓ Auto-resume works for V2 checkpoints
  ✓ Both models can coexist independently
```

## 📂 Directory Structure

```
models/
├── fine_tuned/              ← V1 model (PROTECTED, not modified)
│   ├── adapter_model.safetensors
│   └── adapter_config.json
│
└── fine_tuned_v2/           ← V2 model (NEW, created during training)
    ├── adapter_model.safetensors
    ├── adapter_config.json
    ├── checkpoint-200/      ← Safe checkpoints
    ├── checkpoint-400/
    └── logs/
```

## ⚙️ How It Works

1. **Loads V1**: Reads V1 model from `models/fine_tuned` (read-only)
2. **Enables Training**: Sets LoRA adapter to trainable mode
3. **Trains on New Data**: Uses `train_finetune2.jsonl` dataset
4. **Saves to V2**: Outputs to `models/fine_tuned_v2` directory
5. **Checkpoints Safely**: Creates checkpoints every 200 steps (configurable)

## 🔄 Resume Training

If training is interrupted, just **re-run the same command**:

```powershell
powershell -ExecutionPolicy Bypass -File start_training_v2.ps1
```

It will automatically:
- Detect the latest V2 checkpoint
- Resume from that point
- Continue training seamlessly

## 🧪 Test After Training

```bash
# Test V2 model
python batch_test_model.py --model ./models/fine_tuned_v2 --output results_v2.json

# Compare V1 vs V2
python batch_test_model.py --model ./models/fine_tuned --output results_v1.json
python batch_test_model.py --model ./models/fine_tuned_v2 --output results_v2.json
```

## ⚠️ Troubleshooting

### "V1 model not found"
**Solution**: Run V1 training first:
```powershell
powershell -ExecutionPolicy Bypass -File start_training.ps1
```

### "Dataset not found"
**Solution**: Ensure `train_finetune2.jsonl` and `val_finetune2.jsonl` exist in `data/processed/`

### "Out of memory"
**Solution**: Same GPU requirements as V1 (4GB RTX 3050). If failing:
- Close other GPU applications
- Reduce `per_device_train_batch_size` in config

### "V1 model modified"
**Solution**: This shouldn't happen! V1 is set to read-only during training. Check `verify_v2_safety.py`

## 📊 Checkpointing Details

- **Format**: SafeTensors only (CVE-2025-32434 safe)
- **Frequency**: Every 200 steps (configurable in `configs/training_config.yaml`)
- **Storage**: Only model weights saved (no optimizer states)
- **Location**: `models/fine_tuned_v2/checkpoint-*/`
- **Resume**: Automatic from latest checkpoint

## 🎓 Advanced: Custom Datasets

To train on different data:

```bash
python src/training/fine_tune_v2.py \
  --source-model ./models/fine_tuned \
  --output ./models/fine_tuned_v2_custom \
  --train-data ./data/custom_train.jsonl \
  --val-data ./data/custom_val.jsonl
```

## 🔗 Related Files

- **Training Script**: `src/training/fine_tune_v2.py`
- **Launcher**: `start_training_v2.ps1`
- **Safety Check**: `verify_v2_safety.py`
- **Documentation**: `FINETUNE_V2_GUIDE.md`

---

**Ready?** Run: `powershell -ExecutionPolicy Bypass -File start_training_v2.ps1` 🚀
