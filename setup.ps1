[CmdletBinding()]
param([string]$PythonPath, [switch]$CheckOnly)
$ErrorActionPreference = 'Stop'
$taskRoot = [IO.Path]::GetFullPath($PSScriptRoot)
$taskVenv = Join-Path $taskRoot '.venv\Scripts\python.exe'
$taskProbe = 'import sys,struct,tkinter,venv;sys.exit(0 if struct.calcsize(chr(80))==8 and sys.version_info[:2] in ((3,12),(3,13),(3,14)) else 1)'

function Test-SourcePython([string]$Candidate) {
    if (-not $Candidate -or -not (Test-Path -LiteralPath $Candidate -PathType Leaf)) { return $false }
    try {
        & $Candidate -I -X utf8 -c $taskProbe 2>$null
        return $LASTEXITCODE -eq 0
    } catch { return $false }
}

try {
    if (-not (Test-SourcePython $taskVenv)) {
        $taskCandidates = @()
        if ($PythonPath) { $taskCandidates += $PythonPath }
        foreach ($taskCommandName in @('python.exe', 'python3.exe')) {
            $taskCommand = Get-Command $taskCommandName -ErrorAction SilentlyContinue
            if ($taskCommand -and $taskCommand.Source -notlike '*\WindowsApps\*') {
                $taskCandidates += $taskCommand.Source
            }
        }
        $taskPyLauncher = Get-Command py.exe -ErrorAction SilentlyContinue
        if ($taskPyLauncher) {
            foreach ($taskVersion in @('3.12', '3.13', '3.14')) {
                try {
                    $taskLocated = & $taskPyLauncher.Source "-$taskVersion" -I -c 'import sys; print(sys.executable)' 2>$null
                    if ($LASTEXITCODE -eq 0 -and $taskLocated) { $taskCandidates += [string]$taskLocated }
                } catch { }
            }
        }
        foreach ($taskParent in @((Join-Path $env:LOCALAPPDATA 'Programs\Python'), (Join-Path $env:ProgramFiles 'Python'))) {
            foreach ($taskVersion in @('Python312', 'Python313', 'Python314')) {
                $taskCandidates += Join-Path $taskParent "$taskVersion\python.exe"
            }
        }
        # A local development runtime is allowed, but is never shipped or required.
        $taskCandidates += Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
        $taskPython = $null
        foreach ($taskCandidate in ($taskCandidates | Select-Object -Unique)) {
            if (Test-SourcePython $taskCandidate) { $taskPython = $taskCandidate; break }
        }
        if (-not $taskPython) {
            throw 'Не найден Python 3.12–3.14 x64 с tkinter. Для обычной установки Python не нужен: запустите Install.cmd или скачайте StereoMapper-Setup.exe из GitHub Releases.'
        }
        if ($CheckOnly) {
            Write-Host ('Подходящий Python найден: ' + $taskPython)
            exit 0
        }
        $taskEnvironment = Join-Path $taskRoot '.venv'
        if (Test-Path -LiteralPath $taskEnvironment) {
            $taskResolved = (Resolve-Path -LiteralPath $taskEnvironment).Path
            if ($taskResolved -ne [IO.Path]::GetFullPath((Join-Path $taskRoot '.venv'))) { throw 'Некорректная папка окружения.' }
            $taskItem = Get-Item -LiteralPath $taskResolved -Force
            if (($taskItem.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) { throw '.venv не должна быть ссылкой.' }
            # Preserve broken packages for inspection instead of deleting them.
            $taskBackup = Join-Path $taskRoot ('.venv-backup-' + [guid]::NewGuid().ToString('N'))
            Move-Item -LiteralPath $taskResolved -Destination $taskBackup
        }
        & $taskPython -I -m venv $taskEnvironment
        if ($LASTEXITCODE -ne 0) { throw 'Не удалось создать окружение Python.' }
    }
    if ($CheckOnly) {
        & $taskVenv -I -X utf8 (Join-Path $taskRoot 'check_runtime.py')
        exit $LASTEXITCODE
    }
    Write-Host 'Проверяю и устанавливаю компоненты исходников. Нужен интернет…'
    & $taskVenv -I -m ensurepip --upgrade
    if ($LASTEXITCODE -ne 0) { throw 'Не удалось восстановить pip.' }
    & $taskVenv -I -m pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python-headless
    if ($LASTEXITCODE -ne 0) { throw 'Не удалось убрать несовместимый вариант OpenCV.' }
    # A previous failed download may leave metadata but remove native DLLs.
    & $taskVenv -I -m pip install --disable-pip-version-check --force-reinstall -r (Join-Path $taskRoot 'requirements.txt')
    if ($LASTEXITCODE -ne 0) { throw 'Зависимости не установлены. Проверьте интернет; повторный start.cmd продолжит восстановление.' }
    & $taskVenv -I -X utf8 (Join-Path $taskRoot 'check_runtime.py')
    if ($LASTEXITCODE -ne 0) { throw 'Окружение не прошло проверку после установки.' }
    Write-Host 'Готово. Приложение можно запускать через start.cmd.'
    exit 0
} catch {
    Write-Host ('Не удалось подготовить приложение: ' + $_.Exception.Message) -ForegroundColor Red
    exit 1
}
