@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
if exist "StereoMapper.exe" if exist "runtime\pythonw.exe" (
  start "" "%~dp0StereoMapper.exe"
  exit /b 0
)
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -I -X utf8 "%~dp0check_runtime.py" --quiet
) else (
  cmd /c exit 1
)
if errorlevel 1 (
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1"
  if errorlevel 1 (
    pause
    exit /b 1
  )
)
".venv\Scripts\python.exe" -E -s -X utf8 "%~dp0main.py"
if errorlevel 1 pause
