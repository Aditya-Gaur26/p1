@echo off
REM Process all course materials

echo ============================================================
echo Processing Course Materials
echo ============================================================
echo.

call venv\Scripts\activate.bat

echo Step 1: Extracting from slides...
python src/data_processing/extract_slides.py
echo.

echo Step 2: Extracting from PDFs...
python src/data_processing/extract_pdfs.py
echo.

echo Step 3: Creating training dataset...
python src/data_processing/create_dataset.py
echo.

echo Step 4: Building vector database...
python src/data_processing/build_vectordb.py
echo.

echo ============================================================
echo Data Processing Complete!
echo ============================================================
echo.
echo Next: Run train.bat to fine-tune the model
echo.

pause
