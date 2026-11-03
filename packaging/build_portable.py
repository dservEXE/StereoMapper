"""Rebuild an offline Windows x64 folder using the tested local environment.

Run from the application's .venv; never copies studies or the source venv config.
"""
from datetime import datetime, timezone
import argparse
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys

SOURCE = Path(__file__).resolve().parents[1]
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


def build(destination, runtime_source=None, models_source=None):
    destination = Path(destination).resolve()
    version = (SOURCE / 'stereo/version.py').read_text(encoding='utf-8').split('"')[1]
    if not runtime_source and (not (SITE / "cv2").is_dir() or not (BASE / "pythonw.exe").exists()):
        raise RuntimeError("Build with the application's Windows x64 .venv interpreter.")
    if destination.exists():
        raise RuntimeError(f"Refusing to overwrite an existing release: {destination}")
    # Keep the historical local name below; all paths now come from CLI arguments.
    DESTINATION = destination
    destination.mkdir(parents=True)
    runtime = destination / "runtime"
    if runtime_source:
        original = Path(runtime_source).resolve()
        if not (original / 'pythonw.exe').is_file() or (original / 'pyvenv.cfg').exists():
            raise RuntimeError('Runtime source must be a standalone Windows Python, not a venv.')
        if destination.is_relative_to(original):
            raise RuntimeError('Build destination must be outside the runtime source.')
        print('Copying the verified private Python runtime...', flush=True)
        copy_tree(original, runtime)
    else:
        runtime.mkdir()
        for item in BASE.iterdir():
            if item.is_file() and (item.suffix in (".exe", ".dll") or item.name == "LICENSE.txt"):
                shutil.copy2(item, runtime / item.name)
        copy_tree(BASE / "DLLs", runtime / "DLLs")
        copy_tree(BASE / "Lib", runtime / "Lib", {"site-packages", "test", "idlelib", "ensurepip"})
        copy_tree(BASE / "tcl", runtime / "tcl")
        excluded = {"pip", "imageio_ffmpeg", "isympy.py"}
        for item in SITE.iterdir():
            if item.name.startswith(("pip-", "imageio_ffmpeg-")):
                excluded.add(item.name)
        copy_tree(SITE, runtime / "Lib/site-packages", excluded)
    # The redistributable binaries are local build inputs, never user data.
    for item in (SOURCE / "packaging/vc_runtime").glob("*.dll"):
        shutil.copy2(item, runtime / item.name)
    import os
    for name in ('msvcp140.dll', 'msvcp140_1.dll', 'msvcp140_2.dll', 'msvcp140_atomic_wait.dll',
                 'msvcp140_codecvt_ids.dll', 'vcruntime140.dll', 'vcruntime140_1.dll',
                 'vcruntime140_threads.dll', 'concrt140.dll', 'vcomp140.dll'):
        system = Path(os.environ.get('SystemRoot', 'C:/Windows')) / 'System32' / name
        if not (runtime / name).exists() and system.is_file():
            shutil.copy2(system, runtime / name)
    copy_tree(SOURCE / "stereo", DESTINATION / "stereo")
    model = Path(models_source or SOURCE / 'models') / 'depth-anything-v2-small'
    if not model.is_dir():
        raise RuntimeError('Include the local photo model with --models-source, or download it before building.')
    (DESTINATION / 'models/depth-anything-v2-small').mkdir(parents=True)
    for name in ('config.json', 'preprocessor_config.json', 'model.safetensors', 'revision.txt'):
        if (model / name).exists():
            shutil.copy2(model / name, DESTINATION / 'models/depth-anything-v2-small' / name)
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
                             for d in importlib.metadata.distributions(path=[str(runtime / 'Lib/site-packages')])
                             if not d.metadata["Name"].lower().startswith(("pip", "imageio-ffmpeg"))),
                            key=lambda d: d["name"].lower())
    manifest = {"application": "StereoMapper", "version": version, "runtime_schema": 1, "platform": "Windows x64",
                "built_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
                "offline": True, "components": distributions,
                "excluded": ["studies", "camera photos", "calibration", "user settings", "logs", "screen scale", "pip", "diagnostic ffmpeg"]}
    (DESTINATION / "BUILD_INFO.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    files = [f for f in DESTINATION.rglob("*") if f.is_file()]
    print(json.dumps({"directory": str(DESTINATION), "files": len(files),
                      "bytes": sum(f.stat().st_size for f in files),
                      "longest_relative_path": max(len(str(f.relative_to(DESTINATION))) for f in files)}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=SOURCE / 'dist/app')
    parser.add_argument('--runtime-source', type=Path)
    parser.add_argument('--models-source', type=Path)
    args = parser.parse_args()
    build(args.output, args.runtime_source, args.models_source)
