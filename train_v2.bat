@echo off
REM Continue Fine-tuning V2 - Batch Script
REM Loads V1 model and trains on new dataset

echo ============================================================
echo          Continue Fine-tuning (V2 Training)
echo ============================================================
echo.
echo Source Model: models/fine_tuned (V1)
echo Output Model: models/fine_tuned_v2 (V2)
echo Dataset: train_finetune2.jsonl / val_finetune2.jsonl
echo.
echo Starting V2 training...
echo.

python src\training\fine_tune_v2.py

echo.
echo ============================================================
echo V2 Training Complete!
echo ============================================================
pause
