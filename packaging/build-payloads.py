"""Create clean full and code-update payloads. Never collect project user data."""
import argparse
from pathlib import Path
import zipfile

DIRECTORIES = {"runtime", "stereo", "models", "licenses"}
FILES = {"StereoMapper.exe", "main.py", "app_bootstrap.py", "calibration_poses.png", "app.ico",
         "README.md", "SCANNING_GUIDE.md", "CALIBRATION_GUIDE.md", "3D_EDITOR_GUIDE.md",
         "INTERFACE_GUIDE.md", "START_HERE_RU.txt", "BUILD_INFO.json", "Install.cmd", "install.ps1"}


def build(application, output):
    application, output = Path(application).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    critical = ("StereoMapper.exe", "runtime/pythonw.exe", "runtime/Lib/site-packages/cv2/__init__.py", "stereo/version.py")
    for name in critical:
        if not (application / name).is_file():
            raise ValueError(f"Application missing required file: {name}")
    selected = []
    for path in sorted(application.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(application)
        if "__pycache__" in relative.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        if relative.parts[0] in DIRECTORIES or relative.as_posix() in FILES:
            selected.append((path, relative))
    total, done, last = sum(path.stat().st_size for path, _ in selected), 0, -1
    with zipfile.ZipFile(output / "payload.zip", "w", compression=zipfile.ZIP_DEFLATED, compresslevel=5) as full:
        for path, relative in selected:
            full.write(path, relative.as_posix())
            done += path.stat().st_size
            percent = int(10 * done / total) * 10
            if percent != last:
                print(f"Full package: {percent}%", flush=True)
                last = percent
    with zipfile.ZipFile(output / "update-payload.zip", "w", compression=zipfile.ZIP_DEFLATED, compresslevel=5) as update:
        for path, relative in selected:
            if relative.parts[0] == "stereo" and path.suffix == ".py":
                update.write(path, relative.as_posix())
    for filename in ("payload.zip", "update-payload.zip"):
        with zipfile.ZipFile(output / filename) as archive:
            if archive.testzip() is not None:
                raise ValueError(f"Corrupt payload: {filename}")
        print(f"Verified: {filename}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--application", default="dist/app", type=Path)
    parser.add_argument("--output", default="dist", type=Path)
    args = parser.parse_args()
    build(args.application, args.output)
