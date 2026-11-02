$ErrorActionPreference = 'Stop'
$taskRoot = $PSScriptRoot
$taskVenv = Join-Path $taskRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskVenv)) {
    $taskPython = $null
    $taskCandidates = @(
        (Join-Path $env:LOCALAPPDATA 'Programs\Python\Python312\python.exe'),
        (Join-Path $env:LOCALAPPDATA 'Programs\Python\Python313\python.exe'),
        (Join-Path $env:LOCALAPPDATA 'Programs\Python\Python314\python.exe'),
        (Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe')
    )
    $taskCommand = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($taskCommand -and $taskCommand.Source -notlike '*\WindowsApps\*') {
        $taskCandidates = @($taskCommand.Source) + $taskCandidates
    }
    foreach ($taskCandidate in $taskCandidates) {
        if (Test-Path -LiteralPath $taskCandidate) { $taskPython = $taskCandidate; break }
    }
    if (-not $taskPython) {
        Write-Host 'Install Python 3.12 or 3.13 (64-bit), including tkinter, then run start.cmd again.'
        exit 1
    }
    & $taskPython -m venv (Join-Path $taskRoot '.venv')
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
& $taskVenv -m pip uninstall -y opencv-python opencv-python-headless
& $taskVenv -m pip install -r (Join-Path $taskRoot 'requirements.txt')
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host 'Ready. Run start.cmd.'
