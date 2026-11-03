import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("runtime_probe", ROOT / "check_runtime.py")
probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(probe)


def modules(name):
    if name == "cv2":
        return SimpleNamespace(ximgproc=SimpleNamespace(createRightMatcher=lambda: None,
                                                        createDisparityWLSFilter=lambda: None))
    return SimpleNamespace()


class RuntimeProbeTests(unittest.TestCase):
    def test_complete_runtime_does_not_need_camera(self):
        self.assertTrue(probe.probe_runtime(modules)["ready"])

    def test_incomplete_environment_is_detected_despite_existing_interpreter(self):
        def missing(name):
            if name == "cv2":
                raise ModuleNotFoundError("No module named 'cv2'")
            return modules(name)
        result = probe.probe_runtime(missing)
        self.assertFalse(result["ready"])
        self.assertIn("cv2", " ".join(result["problems"]))

    def test_wrong_opencv_variant_is_rejected(self):
        result = probe.probe_runtime(lambda name: SimpleNamespace() if name == "cv2" else modules(name))
        self.assertFalse(result["ready"])
        self.assertIn("opencv-contrib-python", " ".join(result["problems"]))

    def test_native_dll_failure_is_a_repairable_dependency_problem(self):
        def broken(name):
            if name == "numpy":
                raise OSError("DLL load failed")
            return modules(name)
        self.assertFalse(probe.probe_runtime(broken)["ready"])

    def test_unsupported_architecture_and_version(self):
        with patch.object(probe.struct, "calcsize", return_value=4), patch.object(probe.sys, "version_info", (3, 11, 1)):
            result = probe.probe_runtime(modules)
        self.assertFalse(result["ready"])
        self.assertEqual(len(result["problems"]), 2)


@unittest.skipUnless(os.name == "nt", "Windows startup scripts")
class WindowsBootstrapTests(unittest.TestCase):
    def test_powershell5_metadata_trust_and_download_integrity(self):
        with tempfile.TemporaryDirectory(prefix="stereo-bootstrap-") as temporary:
            folder = Path(temporary)
            check = folder / "checks.ps1"
            installer_script = str(ROOT / "install-from-github.ps1").replace("'", "''")
            check.write_text("""$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
. 'INSTALLER_SCRIPT' -LibraryOnly
function Assert($value, [string]$message) { if (-not $value) { throw $message } }
Assert (Test-TrustedDownloadUrl 'https://github.com/dservEXE/StereoMapper') 'GitHub rejected'
foreach ($bad in @('http://github.com/a','https://github.com.evil.invalid/a','https://user@github.com/a','https://github.com:444/a')) {
 Assert (-not (Test-TrustedDownloadUrl $bad)) ('Untrusted URL accepted: '+$bad)
}
$bytes=[Text.Encoding]::UTF8.GetBytes('verified installer bytes')
$sha=[Security.Cryptography.SHA256]::Create()
$hash=[BitConverter]::ToString($sha.ComputeHash($bytes)).Replace('-','').ToLowerInvariant()
$sha.Dispose()
function Metadata {
 return @{tag_name='v0.13.0';draft=$false;prerelease=$false;assets=@(@{name='StereoMapper-Setup.exe';state='uploaded';size=$bytes.Length;digest=('sha256:'+$hash);browser_download_url='https://github.com/dservEXE/StereoMapper/releases/download/v0.13.0/StereoMapper-Setup.exe'})}
}
$good=Select-InstallerRelease (Metadata)
Assert ($good.Sha256 -ceq $hash) 'Valid release rejected'
for ($i=0; $i -lt 6; $i++) {
 $bad=Metadata
 if ($i -eq 0) { $bad.assets[0].browser_download_url='https://github.com/elsewhere/StereoMapper/releases/download/v0.13.0/StereoMapper-Setup.exe' }
 if ($i -eq 1) { $bad.assets[0].name='StereoMapper-Update.exe' }
 if ($i -eq 2) { $bad.assets[0].digest='sha256:bad' }
 if ($i -eq 3) { $bad.assets[0].size=[long]4294967297 }
 if ($i -eq 4) { $bad.draft=$true }
 if ($i -eq 5) { $bad.assets += $bad.assets[0] }
 $rejected=$false
 try { $null=Select-InstallerRelease $bad } catch { $rejected=$true }
 Assert $rejected ('Invalid release accepted: '+$i)
}
function Open-TrustedResponse([string]$url) {
 $fake=[pscustomobject]@{Stream=(New-Object IO.MemoryStream(,$script:bytes))}
 $fake | Add-Member ScriptMethod GetResponseStream { return $this.Stream }
 $fake | Add-Member ScriptMethod Close { $this.Stream.Dispose() }
 return $fake
}
$target=Join-Path $PSScriptRoot 'valid.exe'
Save-VerifiedInstaller $good $target
Assert ((Get-InstallerSha256 $target) -ceq $hash) 'Download changed'
$good.Sha256='0'*64
$rejected=$false
try { Save-VerifiedInstaller $good (Join-Path $PSScriptRoot 'tampered.exe') } catch { $rejected=$true }
Assert $rejected 'Tampered installer was accepted'
Write-Output 'PASS metadata and SHA-256'
""".replace("INSTALLER_SCRIPT", installer_script), encoding="utf-8-sig")
            result = subprocess.run(["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(check)],
                                    capture_output=True, encoding="utf-8", errors="replace", timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_start_repairs_incomplete_venv_and_skips_repair_when_ready(self):
        compiler = Path(os.environ["WINDIR"]) / "Microsoft.NET/Framework64/v4.0.30319/csc.exe"
        if not compiler.exists():
            self.skipTest("C# compiler unavailable for fake interpreter fixture")
        with tempfile.TemporaryDirectory(prefix="stereo-start-") as temporary:
            folder = Path(temporary) / "Source with spaces"
            venv = folder / ".venv/Scripts"
            fake_bin = folder / "fake-bin"
            venv.mkdir(parents=True)
            fake_bin.mkdir()
            log = folder / "calls.log"
            marker = folder / "ready.marker"
            source = folder / "Fake.cs"
            source.write_text(r'''using System;
using System.IO;
using System.Reflection;
class Fake {
 static int Main(string[] args) {
  string log=Environment.GetEnvironmentVariable("STEREO_TEST_LOG");
  string marker=Environment.GetEnvironmentVariable("STEREO_TEST_MARKER");
  string exe=Path.GetFileName(Assembly.GetExecutingAssembly().Location);
  if (exe.Equals("powershell.exe",StringComparison.OrdinalIgnoreCase)) {
   File.AppendAllText(log,"repair\n");File.WriteAllText(marker,"ready");return 0;
  }
  foreach(string arg in args) {
   if(arg.EndsWith("check_runtime.py")) {File.AppendAllText(log,"probe\n");return File.Exists(marker)?0:1;}
   if(arg.EndsWith("main.py")) {File.AppendAllText(log,"main\n");return File.Exists(marker)?0:9;}
  }
  return 8;
 }
}''', encoding="utf-8")
            fake = venv / "python.exe"
            subprocess.run([str(compiler), "/nologo", "/target:exe", "/platform:x64", f"/out:{fake}", str(source)], check=True, capture_output=True)
            shutil.copy2(fake, fake_bin / "powershell.exe")
            shutil.copy2(ROOT / "start.cmd", folder / "start.cmd")
            environment = os.environ.copy()
            environment.update(STEREO_TEST_LOG=str(log), STEREO_TEST_MARKER=str(marker))
            environment["PATH"] = str(fake_bin) + os.pathsep + environment["PATH"]
            result = subprocess.run(["cmd.exe", "/c", str(folder / "start.cmd")], cwd=folder, env=environment,
                                    capture_output=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(log.read_text().splitlines(), ["probe", "repair", "main"])
            log.write_text("")
            result = subprocess.run(["cmd.exe", "/c", str(folder / "start.cmd")], cwd=folder, env=environment,
                                    capture_output=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(log.read_text().splitlines(), ["probe", "main"])


if __name__ == "__main__":
    unittest.main()
