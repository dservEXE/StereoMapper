$ErrorActionPreference = 'Stop'
$taskPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) { & (Join-Path $PSScriptRoot 'setup.ps1') }
& $taskPython -m pip install 'torch>=2.4,<3' --index-url https://download.pytorch.org/whl/cpu
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $taskPython -m pip install -r (Join-Path $PSScriptRoot 'requirements-photo.txt')
exit $LASTEXITCODE
