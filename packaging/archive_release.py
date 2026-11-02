"""Create and validate a clean release from an allowlist, never from user data."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile

SOURCE = Path(__file__).resolve().parents[1]
PACKAGE = SOURCE.parent / "Готовая_программа" / "StereoMapper"
ARCHIVE = PACKAGE.parent / "StereoMapper-0.12.0-Windows-x64.zip"
FOLDERS = {"runtime", "stereo", "models", "licenses"}
FILES = {"StereoMapper.exe", "app_bootstrap.py", "main.py", "Install.cmd", "install.ps1",
         "START_HERE_RU.txt", "README.md", "SCANNING_GUIDE.md", "CALIBRATION_GUIDE.md",
         "calibration_poses.png", "BUILD_INFO.json", "3D_EDITOR_GUIDE.md", "INTERFACE_GUIDE.md"}


def files():
    for path in sorted(PACKAGE.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(PACKAGE)
        if "__pycache__" in relative.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        if relative.parts[0] in FOLDERS or str(relative) in FILES:
            yield path, relative


def build():
    selected = list(files())
    total = sum(path.stat().st_size for path, _ in selected)
    done, mark = 0, -1
    with zipfile.ZipFile(ARCHIVE, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=5, allowZip64=True) as archive:
        for path, relative in selected:
            archive.write(path, "StereoMapper/" + relative.as_posix())
            done += path.stat().st_size
            percent = int(done / total * 10) * 10
            if percent != mark:
                mark = percent
                print(f"Packing {percent}%", flush=True)
    with zipfile.ZipFile(ARCHIVE) as archive:
        names = archive.namelist()
        for name in names:
            parts = Path(name).parts
            assert parts[0] == "StereoMapper"
            assert parts[1] in FOLDERS or (len(parts) == 2 and parts[1] in FILES), name
        assert not any(name.startswith(("StereoMapper/data/", "StereoMapper/output/", "StereoMapper/camera_calibration/")) for name in names)
        assert "StereoMapper/settings.json" not in names
        assert "StereoMapper/portable_selftest.py" not in names
        assert not any("__pycache__" in name for name in names)
        print("Checking archive integrity...", flush=True)
        assert archive.testzip() is None, "CRC check failed"
    digest = hashlib.file_digest(ARCHIVE.open("rb"), "sha256").hexdigest()
    ARCHIVE.with_suffix(".zip.sha256").write_text(f"{digest}  {ARCHIVE.name}\n", encoding="ascii")
    report = {"file": ARCHIVE.name, "bytes": ARCHIVE.stat().st_size, "unpacked_bytes": total,
              "entries": len(selected), "sha256": digest, "crc_verified": True,
              "clean_release": True, "utc": datetime.now(timezone.utc).isoformat()}
    (SOURCE / "tmp/qa/package_archive.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    build()
