# High-priority training script for Qwen fine-tuning
# Ensures training gets maximum GPU/CPU resources

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Starting High-Priority Training" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Disable sleep/hibernate during training
Write-Host "[1/3] Configuring power settings..." -ForegroundColor Yellow
powercfg /change monitor-timeout-ac 0
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0
powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c  # High Performance plan
Write-Host "  - Disabled sleep/hibernate" -ForegroundColor Green
Write-Host "  - Switched to High Performance mode" -ForegroundColor Green
Write-Host ""

# Set GPU to prefer maximum performance
Write-Host "[2/3] Setting GPU to maximum performance..." -ForegroundColor Yellow
# Note: This requires NVIDIA GPU - will fail gracefully if not available
try {
    nvidia-smi -pm 1 | Out-Null  # Enable persistence mode
    nvidia-smi -pl 300 | Out-Null  # Set power limit (adjust if needed)
    Write-Host "  - GPU configured for maximum performance" -ForegroundColor Green
} catch {
    Write-Host "  - GPU settings unchanged (nvidia-smi not available)" -ForegroundColor Gray
}
Write-Host ""

# Start training with HIGH priority
Write-Host "[3/3] Starting training with HIGH priority..." -ForegroundColor Yellow
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "TRAINING IN PROGRESS" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "You can use your laptop normally." -ForegroundColor White
Write-Host "Training will maintain priority on GPU." -ForegroundColor White
Write-Host "Press Ctrl+C to stop training." -ForegroundColor White
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Start Python with high priority and run training
$process = Start-Process -FilePath "python" -ArgumentList "src/training/fine_tune.py" -PassThru -NoNewWindow

# Set process priority to HIGH
$process.PriorityClass = "High"

Write-Host "Training started with PID: $($process.Id)" -ForegroundColor Green
Write-Host "Process priority: HIGH" -ForegroundColor Green
Write-Host ""

# Wait for training to complete
$process.WaitForExit()

# Restore power settings after training
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Training Complete - Restoring Settings" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
powercfg /change monitor-timeout-ac 10
powercfg /change standby-timeout-ac 30
powercfg /setactive 381b4222-f694-41f0-9685-ff5bb260df2e  # Balanced plan

Write-Host "Power settings restored to normal" -ForegroundColor Green
