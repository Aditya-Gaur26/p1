@echo off
REM Windows Batch Script for Quick Project Setup
REM Usage: run_all.bat

echo ============================================================
echo Operating Systems and Networks - Fine-tuning Pipeline
echo ============================================================
echo.

REM Check if virtual environment exists
if not exist "venv\" (
    echo Creating virtual environment...
    python -m venv venv
    echo ✓ Virtual environment created
    echo.
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
echo.

REM Check if requirements are installed
echo Checking dependencies...
pip show transformers >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies...
    pip install -r requirements.txt
    echo ✓ Dependencies installed
) else (
    echo ✓ Dependencies already installed
)
echo.

REM Run setup
echo Running project setup...
python setup.py
echo.

echo ============================================================
echo Setup Complete!
echo ============================================================
echo.
echo Next Steps:
echo 1. Add your course materials to data/raw/slides and data/raw/books
echo 2. Edit .env file with your API keys (optional)
echo 3. Run: process_data.bat (to process materials)
echo 4. Run: train.bat (to fine-tune model)
echo 5. Run: test.bat (to test the system)
echo.

pause
