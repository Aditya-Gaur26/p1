@echo off
REM Train the model

echo ============================================================
echo Fine-tuning Qwen3 Model
echo ============================================================
echo.

call venv\Scripts\activate.bat

echo Starting training...
echo This may take several hours depending on your GPU.
echo.
echo Monitor progress with:
echo   tensorboard --logdir models/fine_tuned/logs
echo.

python src/training/fine_tune.py

echo.
echo ============================================================
echo Training Complete!
echo ============================================================
echo.
echo Next: Run test.bat to test the system
echo.

pause
