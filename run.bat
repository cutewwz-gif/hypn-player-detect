@echo off
cd /d "%~dp0"

:: If not admin, relaunch as admin (needed to hear keys while Minecraft is focused)
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo Requesting Administrator permission for global T key listening...
    powershell -NoProfile -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
    exit /b
)

echo Installing / checking dependencies...
python -m pip install -q pynput pyperclip
if errorlevel 1 (
    echo Failed to install packages. Is Python installed and in PATH?
    pause
    exit /b 1
)

echo.
echo Starting... Close this window to quit.
echo.
python beg_clipboard.py
echo.
echo Program ended.
pause
