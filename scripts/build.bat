@echo off
echo Building Road Defect System...

:: Activate virtual environment if it exists
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
) else (
    echo Virtual environment not found. Creating one...
    python -m venv venv
    call venv\Scripts\activate.bat
)

:: Install required packages
echo Installing required packages...
pip install -r requirements.txt
pip install pyinstaller pillow

:: Run the build script
echo Running build script...
python build.py

:: Deactivate virtual environment
call venv\Scripts\deactivate.bat

echo.
echo Build process completed!
echo The executable can be found in the 'dist' directory.
pause 