[CmdletBinding()]
param([switch]$LibraryOnly)
$ErrorActionPreference = 'Stop'
$script:Repository = 'dservEXE/StereoMapper'
$script:InstallerName = 'StereoMapper-Setup.exe'
$script:MaximumInstallerBytes = [long]4294967296
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

function Test-TrustedDownloadUrl([string]$Url) {
    try { $address = [uri]$Url } catch { return $false }
    return $address.IsAbsoluteUri -and $address.Scheme -ceq 'https' -and -not $address.UserInfo -and $address.Port -eq 443 -and
        $address.Host -in @('api.github.com', 'github.com', 'release-assets.githubusercontent.com', 'objects.githubusercontent.com')
}

function Select-InstallerRelease($Metadata) {
    if ($Metadata.draft -or $Metadata.prerelease -or [string]$Metadata.tag_name -notmatch '^v?[0-9]+\.[0-9]+\.[0-9]+$') {
        throw 'GitHub вернул неподходящий выпуск приложения.'
    }
    $expected = 'https://github.com/' + $script:Repository + '/releases/download/' + $Metadata.tag_name + '/' + $script:InstallerName
    $assets = @($Metadata.assets | Where-Object { $_.name -ceq $script:InstallerName -and $_.state -ceq 'uploaded' })
    if ($assets.Count -ne 1) { throw 'В этом выпуске пока нет полного установщика StereoMapper-Setup.exe. Пакет StereoMapper-Update.exe подходит только уже установленной программе.' }
    $asset = $assets[0]
    if ($asset.browser_download_url -cne $expected -or
        ($asset.size -isnot [int] -and $asset.size -isnot [long]) -or
        $asset.size -le 0 -or $asset.size -gt $script:MaximumInstallerBytes -or
        [string]$asset.digest -notmatch '^sha256:[a-fA-F0-9]{64}$') {
        throw 'Не удалось проверить адрес, размер и контрольную сумму установщика.'
    }
    return [pscustomobject]@{ Url=$expected; Size=[long]$asset.size; Sha256=$asset.digest.Substring(7).ToLowerInvariant(); Version=[string]$Metadata.tag_name }
}

function Open-TrustedResponse([string]$Url) {
    for ($redirect = 0; $redirect -le 5; $redirect++) {
        if (-not (Test-TrustedDownloadUrl $Url)) { throw 'Недопустимый адрес загрузки.' }
        $request = [Net.HttpWebRequest]::Create($Url)
        $request.AllowAutoRedirect = $false
        $request.Timeout = 30000
        $request.ReadWriteTimeout = 30000
        $request.UserAgent = 'StereoMapper-Installer/0.16.0'
        $request.Accept = 'application/vnd.github+json'
        $response = $request.GetResponse()
        if ([int]$response.StatusCode -in @(301,302,303,307,308)) {
            $next = [uri]::new([uri]$Url, [string]$response.Headers['Location']).AbsoluteUri
            $response.Close()
            $Url = $next
            continue
        }
        if ([int]$response.StatusCode -ne 200) { $response.Close(); throw 'GitHub не передал файл установщика.' }
        return $response
    }
    throw 'Слишком много перенаправлений при загрузке.'
}

function Read-InstallerMetadata {
    $response = Open-TrustedResponse ('https://api.github.com/repos/' + $script:Repository + '/releases/latest')
    try {
        $stream = $response.GetResponseStream()
        $memory = New-Object IO.MemoryStream
        try {
            $buffer = New-Object byte[] 16384
            while (($read = $stream.Read($buffer, 0, $buffer.Length)) -gt 0) {
                if ($memory.Length + $read -gt 1048576) { throw 'Ответ GitHub слишком большой.' }
                $memory.Write($buffer, 0, $read)
            }
            return Select-InstallerRelease ([Text.Encoding]::UTF8.GetString($memory.ToArray()) | ConvertFrom-Json)
        } finally { $stream.Dispose(); $memory.Dispose() }
    } finally { $response.Close() }
}

function Get-InstallerSha256([string]$Path) {
    $fileStream = [IO.File]::OpenRead($Path)
    $algorithm = [Security.Cryptography.SHA256]::Create()
    try { return [BitConverter]::ToString($algorithm.ComputeHash($fileStream)).Replace('-', '').ToLowerInvariant() }
    finally { $fileStream.Dispose(); $algorithm.Dispose() }
}

function Save-VerifiedInstaller($Release, [string]$Target) {
    # Metadata is selected above before any network download or executable launch.
    $response = Open-TrustedResponse $Release.Url
    try {
        $taskDataStream = $response.GetResponseStream()
        $output = [IO.File]::Open($Target, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
        try {
            $buffer = New-Object byte[] 65536
            $total = [long]0
            $lastPercent = -1
            while (($read = $taskDataStream.Read($buffer, 0, $buffer.Length)) -gt 0) {
                $total += $read
                if ($total -gt $Release.Size) { throw 'Размер загруженного файла не совпал с выпуском.' }
                $output.Write($buffer, 0, $read)
                $percent = [int][math]::Floor(100.0 * $total / $Release.Size)
                if ($percent -ne $lastPercent) {
                    Write-Progress -Activity 'Загружаю установщик StereoMapper' -Status ($percent.ToString() + '%') -PercentComplete $percent
                    $lastPercent = $percent
                }
            }
        } finally { $taskDataStream.Dispose(); $output.Dispose(); Write-Progress -Activity 'Загружаю установщик StereoMapper' -Completed }
    } finally { $response.Close() }
    if ($total -ne $Release.Size -or (Get-InstallerSha256 $Target) -cne $Release.Sha256) {
        throw 'Проверка SHA-256 установщика не прошла. Файл запускаться не будет.'
    }
}

if ($LibraryOnly) { return }
$taskDirectory = $null
try {
    Write-Host 'Ищу последний готовый установщик на GitHub…'
    $taskRelease = Read-InstallerMetadata
    $taskDirectory = Join-Path (Join-Path $env:LOCALAPPDATA 'StereoMapper\installer-downloads') ([guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Path $taskDirectory | Out-Null
    $taskTarget = Join-Path $taskDirectory $script:InstallerName
    Write-Host ('Версия ' + $taskRelease.Version + '. Скачивание полного комплекта может занять несколько минут.')
    Save-VerifiedInstaller $taskRelease $taskTarget
    Write-Host 'Контрольная сумма совпала. Открываю установщик…'
    $taskProcess = Start-Process -FilePath $taskTarget -WindowStyle Normal -Wait -PassThru
    if ($taskProcess.ExitCode -ne 0) { throw ('Установка не завершена, код ' + $taskProcess.ExitCode) }
    exit 0
} catch {
    Write-Host ('Установка не завершена: ' + $_.Exception.Message) -ForegroundColor Red
    Write-Host 'Можно скачать полный StereoMapper-Setup.exe вручную: https://github.com/dservEXE/StereoMapper/releases/latest'
    exit 1
} finally {
    # Remove only our own named download after the installer has exited.
    if ($taskDirectory) {
        $taskDownload = Join-Path $taskDirectory $script:InstallerName
        try {
            if (Test-Path -LiteralPath $taskDownload -PathType Leaf) { [IO.File]::Delete($taskDownload) }
            if (Test-Path -LiteralPath $taskDirectory -PathType Container) { [IO.Directory]::Delete($taskDirectory, $false) }
        } catch { }
    }
}
