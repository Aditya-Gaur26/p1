@echo off
REM Test Multimedia Features
REM Tests image extraction, OCR, vision-language, and figure generation

echo ========================================
echo Testing Multimedia Features
echo ========================================
echo.

REM Activate virtual environment if it exists
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
)

echo Running multimedia test suite...
echo.

python test_multimedia.py

echo.
if %ERRORLEVEL% EQU 0 (
    echo ========================================
    echo Tests completed successfully!
    echo ========================================
) else (
    echo ========================================
    echo Some tests failed. See output above.
    echo ========================================
)

pause
