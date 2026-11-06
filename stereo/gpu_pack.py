"""Optional PyTorch for NVIDIA cards, downloaded on request next to the application.

The bundled PyTorch runs on the processor only, and with this one the installer would exceed
GitHub's 2 GiB limit for a file. It is the same PyTorch version, so nothing else has to change.
It is placed first on the import path, which stereo/__init__.py does when the folder exists.
"""
import hashlib
from pathlib import Path
import shutil
import subprocess
import urllib.error
import urllib.request
import zipfile

FOLDER = "gpu_runtime"
WHEEL_URL = "https://download-r2.pytorch.org/whl/cu126/torch-2.14.1%2Bcu126-cp312-cp312-win_amd64.whl"
WHEEL_SIZE = 2602768778
WHEEL_SHA256 = "97e3bc87ea74367518b740d912a7f5259c25a3c8115bf5916abee1d53ff115a4"
DOWNLOAD_SIZE_GB = 2.6
BLOCK = 4 * 1024 * 1024
NO_WINDOW = 0x08000000


def gpu_pack_ready(root) -> bool:
    return (Path(root) / FOLDER / "torch" / "__init__.py").is_file()


def nvidia_present() -> bool:
    """Whether the system has a working NVIDIA driver, without importing anything heavy."""
    try:
        return subprocess.run(["nvidia-smi", "-L"], capture_output=True, timeout=10,
                              creationflags=NO_WINDOW).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _download(target, status, opener, url, size) -> None:
    """Continue an interrupted download: 2.6 GB is a lot to fetch twice."""
    done = target.stat().st_size if target.exists() else 0
    if done >= size:
        return
    headers = {"User-Agent": "StereoMapper-Models"}
    if done:
        headers["Range"] = f"bytes={done}-"
    with opener(urllib.request.Request(url, headers=headers), timeout=60) as response:
        if done and getattr(response, "status", 206) != 206:
            done = 0
        last = -1
        with target.open("ab" if done else "wb") as output:
            while block := response.read(BLOCK):
                output.write(block)
                done += len(block)
                percent = done * 100 // size
                if percent != last:
                    last = percent
                    status(f"Скачиваю ускорение для видеокарты: {percent}%")


def _matches(path, expected) -> bool:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(BLOCK):
            digest.update(block)
    return digest.hexdigest() == expected


def install_gpu_pack(root, status=lambda message: None, opener=urllib.request.urlopen,
                     url=WHEEL_URL, size=WHEEL_SIZE, sha256=WHEEL_SHA256) -> None:
    root = Path(root)
    final, partial = root / FOLDER, root / (FOLDER + ".download")
    fresh = root / (FOLDER + ".new")
    if gpu_pack_ready(root):
        return
    try:
        _download(partial, status, opener, url, size)
        status("Проверяю скачанный файл…")
        if partial.stat().st_size != size or not _matches(partial, sha256):
            partial.unlink()
            raise ValueError("контрольная сумма не совпала")
        status("Распаковываю ускорение для видеокарты…")
        shutil.rmtree(fresh, ignore_errors=True)
        with zipfile.ZipFile(partial) as wheel:
            wheel.extractall(fresh)
        shutil.rmtree(final, ignore_errors=True)
        fresh.replace(final)
        partial.unlink()
    except (OSError, ValueError, zipfile.BadZipFile, urllib.error.URLError) as error:
        shutil.rmtree(fresh, ignore_errors=True)
        raise RuntimeError(f"Не удалось установить ускорение для видеокарты: {error}") from error
