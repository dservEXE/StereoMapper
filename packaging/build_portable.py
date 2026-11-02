"""Rebuild an offline Windows x64 folder using the tested local environment.

Run from the application's .venv; never copies studies or the source venv config.
"""
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys

SOURCE = Path(__file__).resolve().parents[1]
DESTINATION = SOURCE.parent / "Готовая_программа" / "StereoMapper"
BASE = Path(sys.base_prefix)
SITE = Path(sys.prefix) / "Lib/site-packages"
SKIP = {"__pycache__", ".git", ".pytest_cache"}


def copy_tree(source, target, excluded=()):
    target.mkdir(parents=True, exist_ok=True)
    for item in sorted(source.iterdir()):
        if item.name in SKIP or item.name in excluded:
            continue
        dest = target / item.name
        if item.is_dir():
            copy_tree(item, dest)
        elif item.suffix not in (".pyc", ".pyo"):
            shutil.copy2(item, dest)


def build():
    if not (SITE / "cv2").is_dir() or not (BASE / "pythonw.exe").exists():
        raise RuntimeError("Build with the application's Windows x64 .venv interpreter.")
    if DESTINATION.exists():
        raise RuntimeError(f"Refusing to overwrite an existing release: {DESTINATION}")
    DESTINATION.mkdir(parents=True)
    runtime = DESTINATION / "runtime"
    runtime.mkdir()
    for item in BASE.iterdir():
        if item.is_file() and (item.suffix in (".exe", ".dll") or item.name == "LICENSE.txt"):
            shutil.copy2(item, runtime / item.name)
    copy_tree(BASE / "DLLs", runtime / "DLLs")
    # App-local MSVC CRT from the official redistributable; no system install.
    for item in (SOURCE / "packaging/vc_runtime").glob("*.dll"):
        shutil.copy2(item, runtime / item.name)
    copy_tree(BASE / "Lib", runtime / "Lib", {"site-packages", "test", "idlelib", "ensurepip"})
    copy_tree(BASE / "tcl", runtime / "tcl")
    print("Copying the application libraries, CUDA and photo model...", flush=True)
    excluded = {"pip", "imageio_ffmpeg", "isympy.py"}
    for item in SITE.iterdir():
        if item.name.startswith(("pip-", "imageio_ffmpeg-")):
            excluded.add(item.name)
    copy_tree(SITE, runtime / "Lib/site-packages", excluded)
    copy_tree(SOURCE / "stereo", DESTINATION / "stereo")
    copy_tree(SOURCE / "models", DESTINATION / "models")
    for name in ("main.py", "calibration_poses.png"):
        shutil.copy2(SOURCE / name, DESTINATION / name)
    for name in ("README.md", "SCANNING_GUIDE.md", "CALIBRATION_GUIDE.md", "3D_EDITOR_GUIDE.md", "INTERFACE_GUIDE.md"):
        document = SOURCE / "packaging/docs" / name
        shutil.copy2(document if document.exists() else SOURCE / name, DESTINATION / name)
    for name in ("app_bootstrap.py", "install.ps1", "Install.cmd", "START_HERE_RU.txt"):
        shutil.copy2(SOURCE / "packaging" / name, DESTINATION / name)
    copy_tree(SOURCE / "packaging/licenses", DESTINATION / "licenses")
    compiler = Path("C:/Windows/Microsoft.NET/Framework64/v4.0.30319/csc.exe")
    subprocess.run([str(compiler), "/nologo", "/target:winexe", "/platform:x64", "/optimize+",
                    "/reference:System.Windows.Forms.dll", f"/win32icon:{SOURCE / 'packaging/app.ico'}",
                    f"/out:{DESTINATION / 'StereoMapper.exe'}", str(SOURCE / "packaging/Launcher.cs")], check=True)
    distributions = sorted(({"name": d.metadata["Name"], "version": d.version}
                             for d in importlib.metadata.distributions(path=[str(SITE)])
                             if not d.metadata["Name"].lower().startswith(("pip", "imageio-ffmpeg"))),
                            key=lambda d: d["name"].lower())
    manifest = {"application": "StereoMapper", "version": "0.12.1", "platform": "Windows x64",
                "built_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
                "offline": True, "components": distributions,
                "excluded": ["studies", "camera photos", "calibration", "user settings", "logs", "screen scale", "pip", "diagnostic ffmpeg"]}
    (DESTINATION / "BUILD_INFO.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    files = [f for f in DESTINATION.rglob("*") if f.is_file()]
    print(json.dumps({"directory": str(DESTINATION), "files": len(files),
                      "bytes": sum(f.stat().st_size for f in files),
                      "longest_relative_path": max(len(str(f.relative_to(DESTINATION))) for f in files)}, ensure_ascii=False))


if __name__ == "__main__":
    build()
