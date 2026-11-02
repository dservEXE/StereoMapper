[CmdletBinding()]
param(
    [string]$Destination = (Join-Path $env:LOCALAPPDATA 'Programs\StereoMapper'),
    [switch]$NoShortcut,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding

function Get-AbsoluteDirectory([string]$Value) {
    return [System.IO.Path]::GetFullPath($Value).TrimEnd('\', '/')
}

function Assert-OrdinaryTree([string]$Path) {
    # Do not follow junctions or symbolic links while installing an archive.
    $pending = New-Object 'System.Collections.Generic.Stack[string]'
    $pending.Push($Path)
    while ($pending.Count -gt 0) {
        $current = $pending.Pop()
        $item = Get-Item -LiteralPath $current -Force
        if (($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw "Обнаружена ссылка или перенаправление папки: $current"
        }
        if ($item.PSIsContainer) {
            foreach ($child in Get-ChildItem -LiteralPath $current -Force) {
                $pending.Push($child.FullName)
            }
        }
    }
}

try {
    $sourceDirectory = Get-AbsoluteDirectory $PSScriptRoot
    $targetDirectory = Get-AbsoluteDirectory $Destination
    $sourceExe = Join-Path $sourceDirectory 'StereoMapper.exe'
    $targetExe = Join-Path $targetDirectory 'StereoMapper.exe'
    if (-not (Test-Path -LiteralPath $sourceExe -PathType Leaf)) {
        throw 'Распакуйте архив целиком. Install.cmd должен находиться рядом со StereoMapper.exe.'
    }

    $sameDirectory = $sourceDirectory.Equals($targetDirectory, [System.StringComparison]::OrdinalIgnoreCase)
    if (-not $sameDirectory -and (
        $targetDirectory.StartsWith($sourceDirectory + '\', [System.StringComparison]::OrdinalIgnoreCase) -or
        $sourceDirectory.StartsWith($targetDirectory + '\', [System.StringComparison]::OrdinalIgnoreCase))) {
        throw 'Папка установки и папка распакованного архива не должны находиться одна внутри другой.'
    }

    # Existing studies and preferences belong to this PC and are never replaced.
    $preserve = @('data', 'output', 'screen_board.json')
    if (Test-Path -LiteralPath (Join-Path $targetDirectory 'settings.json')) {
        $preserve += 'settings.json'
    }
    $items = @(Get-ChildItem -LiteralPath $sourceDirectory -Force |
        Where-Object { $preserve -notcontains $_.Name })
    foreach ($item in $items) {
        Assert-OrdinaryTree $item.FullName
    }
    $ancestorPath = $targetDirectory
    while (-not (Test-Path -LiteralPath $ancestorPath)) {
        $ancestorPath = Split-Path -Parent $ancestorPath
        if (-not $ancestorPath) { throw 'Не удалось определить родительскую папку установки.' }
    }
    $ancestor = Get-Item -LiteralPath $ancestorPath -Force
    while ($null -ne $ancestor) {
        if (($ancestor.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw "Папка установки находится внутри ссылки или перенаправления: $($ancestor.FullName)"
        }
        $ancestor = $ancestor.Parent
    }
    if (Test-Path -LiteralPath $targetDirectory) {
        # Check only entries that can be replaced, not the user's studies.
        foreach ($item in $items) {
            $existing = Join-Path $targetDirectory $item.Name
            if (Test-Path -LiteralPath $existing) { Assert-OrdinaryTree $existing }
        }
    }

    if ($DryRun) {
        Write-Host "Проверка пройдена. Источник: $sourceDirectory"
        Write-Host "Папка установки: $targetDirectory"
        Write-Host 'Исследования, калибровка, настройки и размер экранной доски сохраняются.'
        Write-Host 'В режиме проверки файлы и ярлыки не создавались.'
        exit 0
    }

    if (-not $sameDirectory) {
        $running = @(Get-Process -Name 'StereoMapper' -ErrorAction SilentlyContinue |
            Where-Object { $_.Path -and $_.Path.Equals($targetExe, [System.StringComparison]::OrdinalIgnoreCase) })
        if ($running.Count -gt 0) {
            throw 'Перед обновлением закройте установленную программу StereoMapper и повторите установку.'
        }
        New-Item -ItemType Directory -Path $targetDirectory -Force | Out-Null
        Write-Host 'Копирую программу. Для большого архива это может занять несколько минут…'
        foreach ($item in $items) {
            Copy-Item -LiteralPath $item.FullName -Destination $targetDirectory -Recurse -Force
        }
    }

    if (-not $NoShortcut) {
        $desktop = [Environment]::GetFolderPath('Desktop')
        if (-not $desktop -or -not (Test-Path -LiteralPath $desktop -PathType Container)) {
            throw 'Не найдена папка рабочего стола. Программа скопирована; запустите StereoMapper.exe из папки установки.'
        }
        $shell = New-Object -ComObject WScript.Shell
        $shortcut = $shell.CreateShortcut((Join-Path $desktop 'StereoMapper.lnk'))
        $shortcut.TargetPath = $targetExe
        $shortcut.WorkingDirectory = $targetDirectory
        $shortcut.Description = 'Стереокамера: расстояние, глубина и 3D-карта'
        $shortcut.IconLocation = $targetExe + ',0'
        $shortcut.Save()
        Write-Host 'Ярлык StereoMapper создан на рабочем столе.'
    }

    Write-Host ''
    Write-Host 'Готово. Программу можно запускать из папки:'
    Write-Host $targetDirectory
    Write-Host 'После установки флешку можно отключить.'
    exit 0
}
catch {
    Write-Host ''
    Write-Host ('Не удалось завершить установку: ' + $_.Exception.Message) -ForegroundColor Red
    exit 1
}
