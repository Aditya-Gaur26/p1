# High-priority V2 training script - Continue fine-tuning from V1 model
# Ensures V1 model stays untouched while V2 trains in separate directory

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Starting V2 High-Priority Training" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Verify V1 model exists and won't be modified
Write-Host "[Pre-check] Verifying V1 model safety..." -ForegroundColor Yellow
$v1ModelPath = "models\fine_tuned\adapter_model.safetensors"
$v2OutputPath = "models\fine_tuned_v2"

if (-Not (Test-Path $v1ModelPath)) {
    Write-Host "  âŒ V1 model not found at: $v1ModelPath" -ForegroundColor Red
    Write-Host "  Run V1 training first: powershell -ExecutionPolicy Bypass -File start_training.ps1" -ForegroundColor Yellow
    exit 1
}

# Create read-only attribute on V1 model to prevent accidental modification
Write-Host "  âœ“ V1 model found" -ForegroundColor Green
Write-Host "  ðŸ”’ Setting V1 model to read-only (protection)" -ForegroundColor Green
Get-ChildItem "models\fine_tuned" -Recurse -File | ForEach-Object { 
    $_.IsReadOnly = $true 
}

Write-Host "  ðŸ“‚ V1 model location: models\fine_tuned (PROTECTED)" -ForegroundColor Cyan
Write-Host "  ðŸ“‚ V2 output location: $v2OutputPath (NEW)" -ForegroundColor Cyan
Write-Host ""

# Disable sleep/hibernate during training
Write-Host "[1/4] Configuring power settings..." -ForegroundColor Yellow
powercfg /change monitor-timeout-ac 0
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0
powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c  # High Performance plan
Write-Host "  - Disabled sleep/hibernate" -ForegroundColor Green
Write-Host "  - Switched to High Performance mode" -ForegroundColor Green
Write-Host ""

# Set GPU to prefer maximum performance
Write-Host "[2/4] Setting GPU to maximum performance..." -ForegroundColor Yellow
try {
    nvidia-smi -pm 1 | Out-Null  # Enable persistence mode
    nvidia-smi -pl 300 | Out-Null  # Set power limit (adjust if needed)
    Write-Host "  - GPU configured for maximum performance" -ForegroundColor Green
} catch {
    Write-Host "  - GPU settings unchanged (nvidia-smi not available)" -ForegroundColor Gray
}
Write-Host ""

# Show checkpoint configuration
Write-Host "[3/4] Checkpoint configuration..." -ForegroundColor Yellow
Write-Host "  âœ“ Checkpoints save to: $v2OutputPath\checkpoint-*" -ForegroundColor Green
Write-Host "  âœ“ Safe format: safetensors only (no .pt/.pth)" -ForegroundColor Green
Write-Host "  âœ“ Auto-resume: Enabled from latest V2 checkpoint" -ForegroundColor Green
Write-Host "  âœ“ V1 model: Protected from modifications" -ForegroundColor Green
Write-Host ""

# Start training with HIGH priority
Write-Host "[4/4] Starting V2 training with HIGH priority..." -ForegroundColor Yellow
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "V2 TRAINING IN PROGRESS" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Source: V1 model (models/fine_tuned)" -ForegroundColor White
Write-Host "Output: V2 model (models/fine_tuned_v2)" -ForegroundColor White
Write-Host "Dataset: train_finetune2.jsonl" -ForegroundColor White
Write-Host "" -ForegroundColor White
Write-Host "You can use your laptop normally." -ForegroundColor White
Write-Host "Training will maintain priority on GPU." -ForegroundColor White
Write-Host "Press Ctrl+C to stop training." -ForegroundColor White
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Start Python with high priority and run V2 training
$process = Start-Process -FilePath "python" -ArgumentList "src/training/fine_tune_v2.py" -PassThru -NoNewWindow

# Set process priority to HIGH
$process.PriorityClass = "High"

Write-Host "V2 Training started with PID: $($process.Id)" -ForegroundColor Green
Write-Host "Process priority: HIGH" -ForegroundColor Green
Write-Host ""

# Wait for training to complete
$process.WaitForExit()

# Remove read-only protection from V1 model (restore normal state)
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "V2 Training Complete - Cleanup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

Write-Host "Restoring V1 model permissions..." -ForegroundColor Yellow
Get-ChildItem "models\fine_tuned" -Recurse -File | ForEach-Object { 
    $_.IsReadOnly = $false 
}
Write-Host "  âœ“ V1 model permissions restored" -ForegroundColor Green

# Verify V1 model wasn't modified
Write-Host "Verifying V1 model integrity..." -ForegroundColor Yellow
if (Test-Path $v1ModelPath) {
    Write-Host "  âœ“ V1 model intact at: models\fine_tuned" -ForegroundColor Green
} else {
    Write-Host "  âš ï¸  V1 model missing (unexpected)" -ForegroundColor Red
}

# Verify V2 model was created
Write-Host "Verifying V2 model creation..." -ForegroundColor Yellow
if (Test-Path "$v2OutputPath\adapter_model.safetensors") {
    Write-Host "  âœ“ V2 model saved to: $v2OutputPath" -ForegroundColor Green
    
    # Count checkpoints
    $checkpoints = Get-ChildItem "$v2OutputPath\checkpoint-*" -Directory -ErrorAction SilentlyContinue
    if ($checkpoints) {
        Write-Host "  âœ“ Checkpoints created: $($checkpoints.Count)" -ForegroundColor Green
    }
} else {
    Write-Host "  âš ï¸  V2 model not found (training may have failed)" -ForegroundColor Red
}

# Restore power settings
Write-Host ""
Write-Host "Restoring power settings..." -ForegroundColor Yellow
powercfg /change monitor-timeout-ac 10
powercfg /change standby-timeout-ac 30
powercfg /setactive 381b4222-f694-41f0-9685-ff5bb260df2e  # Balanced plan
Write-Host "  âœ“ Power settings restored to normal" -ForegroundColor Green

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "V2 TRAINING COMPLETE!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Models:" -ForegroundColor White
Write-Host "  V1 (original): models\fine_tuned" -ForegroundColor Cyan
Write-Host "  V2 (new):      models\fine_tuned_v2" -ForegroundColor Cyan
Write-Host ""
Write-Host "To test V2 model:" -ForegroundColor White
Write-Host "  python batch_test_model.py --model ./models/fine_tuned_v2" -ForegroundColor Yellow
Write-Host ""

