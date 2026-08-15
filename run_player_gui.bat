@echo off
cd /d "%~dp0"

echo Checking dependencies...
python -m pip install -q onnxruntime-directml opencv-python numpy
if errorlevel 1 (
    echo Failed to install packages.
    pause
    exit /b 1
)

echo.
echo Starting Player Video GUI...
python player_video_gui.py
pause
