@echo off
REM Test the system interactively

echo ============================================================
echo Testing Fine-tuned Model
echo ============================================================
echo.

call venv\Scripts\activate.bat

echo Starting interactive Q&A mode...
echo.
echo Type your questions about Operating Systems and Networks
echo Type 'quit' or 'exit' to stop
echo.

python src/inference/query_processor.py --interactive

pause
