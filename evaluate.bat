@echo off
REM Evaluate the model on test questions

echo ============================================================
echo Evaluating Model Performance
echo ============================================================
echo.

call venv\Scripts\activate.bat

echo Running evaluation on end-semester questions...
echo.

python src/evaluation/evaluate_model.py --test-file data/evaluation/endsem_questions.json

echo.
echo Results saved to outputs/results/
echo.

pause
