@echo off
echo Road Defect Indexing System Setup
echo ================================
echo.

:: Create and activate virtual environment
echo Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo Failed to create virtual environment!
    pause
    exit /b 1
)

:: Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo Failed to activate virtual environment!
    pause
    exit /b 1
)

:: Navigate to models directory
echo Changing directory to src/app/models...
cd src\app\models
if errorlevel 1 (
    echo Failed to change directory!
    pause
    exit /b 1
)

:: Clone YOLOv10 repository
echo Cloning YOLOv10 repository...
git clone https://github.com/THU-MIG/yolov10.git
if errorlevel 1 (
    echo Failed to clone YOLOv10 repository!
    pause
    exit /b 1
)

:: Install YOLOv10
echo Installing YOLOv10...
cd yolov10
pip install .
if errorlevel 1 (
    echo Failed to install YOLOv10!
    pause
    exit /b 1
)

:: Return to root directory
cd ..\..\..\..

:: Install PyTorch with CUDA 12.1
echo Installing PyTorch with CUDA 12.1...
pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
if errorlevel 1 (
    echo Failed to install PyTorch!
    pause
    exit /b 1
)

:: Install other requirements
echo Installing other requirements...
pip install -r requirements.txt
if errorlevel 1 (
    echo Failed to install requirements!
    pause
    exit /b 1
)

echo.
echo Setup completed successfully!
echo.
echo To start the application:
echo 1. Run 'start.bat' or
echo 2. Activate the virtual environment: venv\Scripts\activate
echo 3. Run: python src/app/main.py
echo.
pause
