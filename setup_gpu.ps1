$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath "$PSScriptRoot\.venv\Scripts\python.exe")) { & "$PSScriptRoot\setup.ps1" }
& "$PSScriptRoot\.venv\Scripts\python.exe" -m pip install -r "$PSScriptRoot\requirements_gpu.txt"
if ($LASTEXITCODE -ne 0) { throw 'Не удалось установить CUDA-модуль.' }
& "$PSScriptRoot\.venv\Scripts\python.exe" -c "from stereo.gpu_stereo import runtime,gpu_info; runtime(); print(gpu_info())"
if ($LASTEXITCODE -ne 0) { throw 'Не прошла проверка CUDA. Приложение может работать на CPU.' }
