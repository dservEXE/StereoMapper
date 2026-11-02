param([string]$Product='Kadr',[string]$Folder='Kadr',[string]$Executable='Kadr.exe',[string]$Version,[switch]$CodeOnly)
$ErrorActionPreference='Stop'
$root=Split-Path -Parent $PSScriptRoot
$dist=Join-Path $root 'dist'
$payload=Join-Path $dist 'payload.zip'
if(-not(Test-Path -LiteralPath $payload)){throw 'Build dist/payload.zip first.'}
$hash=(Get-FileHash -LiteralPath $payload -Algorithm SHA256).Hash
$generated=Join-Path $dist 'BuildInfo.cs'
$code='internal static class BuildInfo { public const string Name="'+$Product+'", Folder="'+$Folder+'", Executable="'+$Executable+'", Version="'+$Version+'", Hash="'+$hash+'"; public const bool CodeOnly='+$CodeOnly.IsPresent.ToString().ToLowerInvariant()+'; }'
[IO.File]::WriteAllText($generated,$code)
$compiler='C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe'
& $compiler /nologo /target:winexe /platform:x64 /optimize+ /reference:System.Windows.Forms.dll /reference:System.Drawing.dll /reference:System.IO.Compression.dll /reference:System.IO.Compression.FileSystem.dll /reference:Microsoft.CSharp.dll ('/resource:'+$payload+',Payload') ('/out:'+(Join-Path $dist ($Product+'-Setup.exe'))) (Join-Path $PSScriptRoot 'Setup.cs') $generated
if($LASTEXITCODE -ne 0){throw 'Installer compilation failed.'}
$verify=Join-Path $dist 'verify'
$process=Start-Process -FilePath (Join-Path $dist ($Product+'-Setup.exe')) -ArgumentList @('--verify',('"'+$verify+'"')) -WindowStyle Hidden -Wait -PassThru
if($process.ExitCode -ne 0){throw 'Installer payload verification failed.'}
