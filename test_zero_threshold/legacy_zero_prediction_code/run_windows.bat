@echo off
setlocal
cd /d "%~dp0"
if defined ZERO_PYTHON goto run
if exist "..\.venv\Scripts\python.exe" set "ZERO_PYTHON=%~dp0..\.venv\Scripts\python.exe"
if defined ZERO_PYTHON goto run
if exist "C:\Users\Public\anaconda3\python.exe" set "ZERO_PYTHON=C:\Users\Public\anaconda3\python.exe"
if not defined ZERO_PYTHON set "ZERO_PYTHON=python"
:run
"%ZERO_PYTHON%" -u "%~dp0run_experiment.py" %*
if errorlevel 1 (
  echo Experiment failed. Read the error above.
  pause
  exit /b 1
)
echo Results saved in test_zero_threshold\results.
pause
