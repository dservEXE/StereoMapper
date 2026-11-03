[CmdletBinding()]
param(
    [string]$Product='StereoMapper',
    [string]$Folder='StereoMapper',
    [string]$Executable='StereoMapper.exe',
    [string]$Version,
    [switch]$CodeOnly,
    [string]$Payload,
    [string]$OutputDirectory,
    [switch]$SkipVerify
)
$ErrorActionPreference='Stop'
function Get-TaskSHA256([string]$Path) {
    $taskHashInput=[IO.File]::OpenRead($Path)
    $taskHasher=[Security.Cryptography.SHA256]::Create()
    try { return ([BitConverter]::ToString($taskHasher.ComputeHash($taskHashInput))).Replace('-','') }
    finally { $taskHasher.Dispose(); $taskHashInput.Dispose() }
}
$taskRoot=Split-Path -Parent $PSScriptRoot
if (-not $Version) { $Version=[regex]::Match((Get-Content (Join-Path $taskRoot 'stereo\version.py') -Raw),'"(\d+\.\d+\.\d+)"').Groups[1].Value }
if ($Version -notmatch '^\d+\.\d+\.\d+$') { throw 'Version must be major.minor.patch.' }
foreach ($value in @($Product,$Folder,$Executable)) { if ($value -notmatch '^[A-Za-z0-9._-]+$') { throw 'Invalid product configuration.' } }
if (-not $OutputDirectory) { $OutputDirectory=Join-Path $taskRoot 'dist' }
New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
if (-not $Payload) { $Payload=Join-Path $OutputDirectory 'payload.zip' }
if (-not (Test-Path -LiteralPath $Payload -PathType Leaf)) { throw 'Build the payload ZIP first.' }
$taskPayload=(Resolve-Path -LiteralPath $Payload).Path
$taskHash=Get-TaskSHA256 $taskPayload
$taskBuildInfo=Join-Path $OutputDirectory 'BuildInfo.cs'
$taskCode='internal static class BuildInfo { public const string Name="'+$Product+'", Folder="'+$Folder+'", Executable="'+$Executable+'", Version="'+$Version+'", Hash="'+$taskHash+'"; public const bool CodeOnly='+$CodeOnly.IsPresent.ToString().ToLowerInvariant()+'; }'
[IO.File]::WriteAllText($taskBuildInfo,$taskCode)
$taskCompiler='C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe'
$taskBase=Join-Path $OutputDirectory ('installer-base-'+[guid]::NewGuid().ToString('N')+'.exe')
& $taskCompiler /nologo /target:winexe /platform:x64 /optimize+ /reference:System.Windows.Forms.dll /reference:System.Drawing.dll /reference:System.IO.Compression.dll /reference:System.IO.Compression.FileSystem.dll /reference:Microsoft.CSharp.dll ('/win32icon:'+(Join-Path $PSScriptRoot 'app.ico')) ('/out:'+$taskBase) (Join-Path $PSScriptRoot 'Setup.cs') $taskBuildInfo
if ($LASTEXITCODE -ne 0) { throw 'Installer compilation failed.' }
$taskName=if ($CodeOnly) { $Product+'-Update.exe' } else { $Product+'-Setup.exe' }
$taskOutput=Join-Path $OutputDirectory $taskName
# ZIP is an overlay, not a .NET manifest resource: no 2-GB byte-array allocation.
$taskStream=[IO.File]::Open($taskOutput,[IO.FileMode]::Create,[IO.FileAccess]::Write)
try {
    foreach ($taskFile in @($taskBase,$taskPayload)) {
        $taskInput=[IO.File]::OpenRead($taskFile)
        try { $taskInput.CopyTo($taskStream,1048576) } finally { $taskInput.Dispose() }
    }
    $taskFooter=[Text.Encoding]::ASCII.GetBytes('SMAPZIP1')
    $taskStream.Write($taskFooter,0,$taskFooter.Length)
    $taskLength=[BitConverter]::GetBytes([long](Get-Item -LiteralPath $taskPayload).Length)
    $taskStream.Write($taskLength,0,$taskLength.Length)
} finally { $taskStream.Dispose() }
$taskSetupHash=(Get-TaskSHA256 $taskOutput).ToLowerInvariant()
[IO.File]::WriteAllText($taskOutput+'.sha256',$taskSetupHash+'  '+$taskName+[Environment]::NewLine)
if (-not $SkipVerify) {
    $taskVerify=Join-Path $OutputDirectory ('verify-'+[guid]::NewGuid().ToString('N'))
    $taskResult=Join-Path $OutputDirectory 'installer-verify.txt'
    $taskProcess=Start-Process -FilePath $taskOutput -ArgumentList @('--verify',('"'+$taskVerify+'"'),'--result',('"'+$taskResult+'"')) -WindowStyle Hidden -Wait -PassThru
    if ($taskProcess.ExitCode -ne 0) { throw ('Installer verification failed: '+(Get-Content -LiteralPath $taskResult -Raw -ErrorAction SilentlyContinue)) }
}
Write-Output ('Installer ready: '+$taskOutput)
