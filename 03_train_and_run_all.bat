@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Run 00_install_plotting.bat first.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto fail
".venv\Scripts\python.exe" run_local.py full
if errorlevel 1 goto fail
pause
exit /b 0
:fail
echo The task failed. Check the error message above.
pause
exit /b 1
