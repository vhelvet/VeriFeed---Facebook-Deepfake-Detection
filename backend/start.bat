@echo off
REM ============================================
REM VERIFEED PRODUCTION STARTUP SCRIPT (Windows)
REM ============================================

echo ==========================================
echo Starting Verifeed Production Server
echo ==========================================

REM Check if .env exists
if not exist .env (
    echo ERROR: .env file not found!
    echo Please copy .env.example to .env and configure it.
    pause
    exit /b 1
)

REM Check if venv exists
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install/update dependencies
echo Installing dependencies...
pip install --upgrade pip
pip install -r requirements.txt

REM Check if model exists
if not exist models\model_acc_84.17_e8.pt (
    echo WARNING: Model file not found in models\ directory!
    echo Please place your trained model in the models\ directory.
)

REM Create logs directory
if not exist logs mkdir logs

echo.
echo ==========================================
echo SECURITY CHECKS
echo ==========================================

REM Check FLASK_DEBUG
findstr /C:"FLASK_DEBUG=True" .env >nul 2>&1
if %errorlevel% equ 0 (
    echo WARNING: FLASK_DEBUG is enabled! Disable for production!
    set /p continue="Continue anyway? (y/N): "
    if /i not "%continue%"=="y" exit /b 1
) else (
    echo OK: FLASK_DEBUG is disabled
)

REM Check API keys
findstr /C:"API_KEYS=" .env | findstr /V /C:"API_KEYS=$" >nul 2>&1
if %errorlevel% neq 0 (
    echo WARNING: API_KEYS not configured!
    set /p continue="Continue anyway? (y/N): "
    if /i not "%continue%"=="y" exit /b 1
) else (
    echo OK: API_KEYS configured
)

echo ==========================================
echo.

REM Start the server
echo Starting server...
python deepfake_prediction_secure.py

pause