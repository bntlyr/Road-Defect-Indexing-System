@echo off
echo Starting Road Defect Indexing System...
echo.

:: Check if virtual environment exists
if not exist venv (
    echo Virtual environment not found!
    echo Please run setup.bat first.
    pause
    exit /b 1
)

:: Activate virtual environment and start the application
call venv\Scripts\activate
if errorlevel 1 (
    echo Failed to activate virtual environment!
    pause
    exit /b 1
)

:: Start the application
python -m src.app
if errorlevel 1 (
    echo Application failed to start!
    pause
    exit /b 1
)

pause 
