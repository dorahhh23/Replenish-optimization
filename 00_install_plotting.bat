@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" goto install
where py >nul 2>nul
if errorlevel 1 (
    python -m venv .venv
) else (
    py -3 -m venv .venv
)
if errorlevel 1 goto fail
:install
".venv\Scripts\python.exe" -m pip install -r requirements_plotting.txt
if errorlevel 1 goto fail
echo Ready. Double-click 01_histogram.bat to draw the new histogram.
pause
exit /b 0
:fail
echo Installation failed. Check the message above. Python 3.10 or later is required.
pause
exit /b 1
