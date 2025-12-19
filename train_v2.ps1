# Continue Fine-tuning V2 Script
# Loads the V1 fine-tuned model and continues training on new dataset
# Saves to models/fine_tuned_v2

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "         Continue Fine-tuning (V2 Training)                " -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Source Model: models/fine_tuned (V1)" -ForegroundColor Yellow
Write-Host "Output Model: models/fine_tuned_v2 (V2)" -ForegroundColor Yellow
Write-Host "Dataset: train_finetune2.jsonl / val_finetune2.jsonl" -ForegroundColor Yellow
Write-Host ""
Write-Host "Starting V2 training..." -ForegroundColor Green
Write-Host ""

# Run the V2 fine-tuning script
python src/training/fine_tune_v2.py

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "V2 Training Complete!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan
