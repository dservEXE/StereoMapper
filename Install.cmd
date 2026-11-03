@echo off
setlocal
chcp 65001 >nul
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0install-from-github.ps1"
if errorlevel 1 (
  echo.
  echo Installation was not completed. See the message above.
  pause
  exit /b 1
)
