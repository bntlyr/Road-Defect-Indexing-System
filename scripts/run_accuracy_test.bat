@echo off
echo ============================================
echo RDIS Severity Calculator Accuracy Test
echo ============================================
echo.

:: Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python and try again.
    pause
    exit /b 1
)

:: Check if test directory exists
set "TEST_DIR=C:\Users\rafab\OneDrive\Desktop\accuracy test"
if not exist "%TEST_DIR%" (
    echo WARNING: Test directory not found: %TEST_DIR%
    echo.
    echo Setting up test directory...
    python test_data_utility.py --setup
    echo.
    echo Please add your test images and annotations, then run this script again.
    pause
    exit /b 0
)

echo Test directory found: %TEST_DIR%
echo.

:: Validate test data first
echo Validating test data...
python test_data_utility.py --validate
if errorlevel 1 (
    echo.
    echo ERROR: Test data validation failed
    echo Please fix the issues and try again.
    pause
    exit /b 1
)

echo.
echo Test data validation passed!
echo.

:: Ask user which test to run
echo Choose test type:
echo 1. Quick test with enhanced visualizations
echo 2. Comprehensive test (detailed analysis)
echo 3. Comprehensive test with visualization plots
echo 4. Quick test results only (faster, no images)
echo.
set /p "CHOICE=Enter your choice (1-4): "
echo.
echo You selected: %CHOICE%

if %CHOICE%==1 (
    echo Running quick test with enhanced visualizations...
    echo Note: Confidence thresholds are NOT shown in visualizations
    python test_rdis_accuracy.py
) else if %CHOICE%==2 (
    echo Running comprehensive test...
    python test_severity_calculator_accuracy.py --no_comparisons
) else if %CHOICE%==3 (
    echo Running comprehensive test with visualization plots...
    python test_severity_calculator_accuracy.py --visualize
) else if %CHOICE%==4 (
    echo Running quick test results only...
    python test_rdis_results_only.py
) else (
    echo Invalid choice. Running enhanced test by default...
    python test_rdis_accuracy.py
)

echo.
echo ============================================
echo Test completed!
echo Check the generated files for detailed results.
echo ============================================
pause
