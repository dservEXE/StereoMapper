"""Optional accelerator packs, downloaded on request next to the application.

The bundled PyTorch runs on the processor only. A pack adds a PyTorch that can use a graphics
card: CUDA for NVIDIA, DirectML for AMD, Intel and others. Neither fits the installer, as GitHub
limits a file to 2 GiB. A pack is first placed beside the active one and swapped in at the next
start, because loaded libraries cannot be replaced while the program runs. stereo/__init__.py
puts the active folder first on the import path. This module must import no heavy library.
"""
from dataclasses import dataclass
from functools import cache
import hashlib
import http.client
import json
from pathlib import Path
import re
import shutil
import subprocess
import urllib.error
import urllib.request
import zipfile

FOLDER = "gpu_runtime"
PENDING = FOLDER + ".new"
REMOVAL_REQUEST = FOLDER + ".remove"
MARKER = "stereomapper_pack.json"
BLOCK = 4 * 1024 * 1024
NO_WINDOW = 0x08000000
VENDOR_PACKS = (("nvidia", "nvidia"), ("advanced micro", "directml"), ("amd", "directml"), ("ati ", "directml"),
                ("radeon", "directml"), ("intel", "directml"))
INTEGRATED_NAMES = ("radeon(tm) graphics", "radeon graphics", "radeon vega", "vega 8 graphics", "uhd graphics",
                    "hd graphics", "iris", "basic render")
INTEGRATED_RADEON = re.compile(r"radeon(\(tm\))? \d{3}m\b")


@dataclass(frozen=True)
class Wheel:
    url: str
    size: int
    sha256: str


@dataclass(frozen=True)
class AcceleratorPack:
    key: str
    title: str
    summary: str
    wheels: tuple

    @property
    def size_gb(self) -> float:
        return sum(wheel.size for wheel in self.wheels) / 1e9

    @property
    def size_text(self) -> str:
        return f"{self.size_gb:.1f} ГБ".replace(".", ",")


PACKS = {
    "nvidia": AcceleratorPack(
        "nvidia", "NVIDIA (CUDA)", "Самое быстрое на картах NVIDIA.",
        (Wheel("https://download-r2.pytorch.org/whl/cu126/torch-2.14.1%2Bcu126-cp312-cp312-win_amd64.whl",
               2602768778, "97e3bc87ea74367518b740d912a7f5259c25a3c8115bf5916abee1d53ff115a4"),)),
    "directml": AcceleratorPack(
        "directml", "AMD, Intel и другие (DirectML)", "Для любых видеокарт с DirectX 12, в том числе встроенных.",
        (Wheel("https://download-r2.pytorch.org/whl/cpu/torch-2.4.1%2Bcpu-cp312-cp312-win_amd64.whl",
               201717875, "3a570e5c553415cdbddfe679207327b3a3806b21c6adea14fba77684d1619e97"),
         Wheel("https://files.pythonhosted.org/packages/71/f2/eb7d3c601310c758f6aef891773e4a00f7388eeb613d536c0d2681fcb105/"
               "torch_directml-0.2.5.dev240914-cp312-cp312-win_amd64.whl",
               8956338, "ea19d11e33e9450b290311c06f7eb10924dd25c555e504d367b7b437d3eb24d0"))),
}


@dataclass(frozen=True)
class Advice:
    pack: str | None
    text: str


class DownloadInterrupted(RuntimeError):
    """The connection dropped. What arrived is kept, and pressing again continues from there."""


def _read_marker(folder) -> str | None:
    if not (Path(folder) / "torch" / "__init__.py").is_file():
        return None
    try:
        return json.loads((Path(folder) / MARKER).read_text(encoding="utf-8"))["pack"]
    except (OSError, ValueError, KeyError, TypeError):
        # Installed by 0.15.2, before packs had a marker, when NVIDIA was the only one.
        return "nvidia"


def installed_pack(root) -> str | None:
    return _read_marker(Path(root) / FOLDER)


def pending_pack(root) -> str | None:
    """A pack waiting for the next start, when it replaces the active one."""
    return _read_marker(Path(root) / PENDING)


def gpu_pack_ready(root) -> bool:
    return installed_pack(root) is not None or pending_pack(root) is not None


def activate_pending(root) -> Path | None:
    """Finish a swap or removal before PyTorch loads; return the active folder, if any."""
    root = Path(root)
    final, fresh, removal = root / FOLDER, root / PENDING, root / REMOVAL_REQUEST
    try:
        if removal.exists():
            shutil.rmtree(final, ignore_errors=True)
            removal.unlink(missing_ok=True)
        if _read_marker(fresh):
            shutil.rmtree(final, ignore_errors=True)
            if not final.exists():
                fresh.replace(final)
    except OSError:
        pass
    return final if _read_marker(final) else None


def request_removal(root) -> None:
    """Take the pack away at the next start, and drop one still waiting for its turn."""
    root = Path(root)
    shutil.rmtree(root / PENDING, ignore_errors=True)
    if installed_pack(root):
        (root / REMOVAL_REQUEST).write_text("", encoding="utf-8")


@cache
def graphics_adapters() -> tuple:
    """Names of the graphics adapters in this computer; empty if they cannot be read."""
    command = "Get-CimInstance Win32_VideoController | ForEach-Object { $_.Name + ' | ' + $_.AdapterCompatibility }"
    try:
        result = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
                                capture_output=True, timeout=20, creationflags=NO_WINDOW, text=True, errors="replace")
    except (OSError, subprocess.SubprocessError):
        return ()
    return tuple(line.strip() for line in result.stdout.splitlines() if line.strip()) if result.returncode == 0 else ()


def is_integrated(name: str) -> bool:
    """A graphics chip built into the processor, sharing its memory and usually no faster than it."""
    lowered = name.lower()
    return any(word in lowered for word in INTEGRATED_NAMES) or INTEGRATED_RADEON.search(lowered) is not None


def adapter_pack(line: str) -> str | None:
    lowered = line.lower()
    return next((pack for vendor, pack in VENDOR_PACKS if vendor in lowered), None)


def recommended_pack(adapters) -> str | None:
    """NVIDIA wins when present, since it is the fastest and computes stereo depth as well."""
    choices = {adapter_pack(line) for line in adapters if not is_integrated(line)}
    return "nvidia" if "nvidia" in choices else "directml" if "directml" in choices else None


def pack_advice(adapters, present=None) -> Advice:
    """What to tell the owner of this computer, given its adapters and the pack it already has."""
    if not adapters:
        return Advice(None, "Видеокарты определить не удалось. Комплект можно выбрать вручную.")
    names = [line.split(" | ")[0] for line in adapters]
    choice = recommended_pack(adapters)
    if choice is None:
        if all(is_integrated(line) for line in adapters):
            return Advice(None, f"На этом компьютере только встроенная графика ({', '.join(names)}). Обычно она "
                                "не быстрее процессора, поэтому скачивать ничего не нужно.")
        return Advice(None, f"Для этой видеокарты ({', '.join(names)}) подходящего комплекта нет.")
    card = next(line.split(" | ")[0] for line in adapters if not is_integrated(line) and adapter_pack(line) == choice)
    pack = PACKS[choice]
    if choice == present:
        return Advice(None, f"{card}: подходящий комплект уже установлен.")
    return Advice(choice, f"Для этого компьютера ({card}) рекомендуется комплект {pack.title}, {pack.size_text}.")


def _download(target, status, opener, wheel, label) -> None:
    done = target.stat().st_size if target.exists() else 0
    if done >= wheel.size:
        return
    headers = {"User-Agent": "StereoMapper-Models"}
    if done:
        headers["Range"] = f"bytes={done}-"
    with opener(urllib.request.Request(wheel.url, headers=headers), timeout=60) as response:
        if done and getattr(response, "status", 206) != 206:
            done = 0
        last = -1
        with target.open("ab" if done else "wb") as output:
            while block := response.read(BLOCK):
                output.write(block)
                done += len(block)
                percent = done * 100 // wheel.size
                if percent != last:
                    last = percent
                    status(f"Скачиваю {label}: {percent}%")
    if done < wheel.size:
        # A dropped connection ends the read without an error; the next press continues.
        raise DownloadInterrupted("Связь прервалась. Нажмите ещё раз: скачивание продолжится с этого места.")


def _matches(path, expected) -> bool:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(BLOCK):
            digest.update(block)
    return digest.hexdigest() == expected


def install_pack(root, key, status=lambda message: None, opener=urllib.request.urlopen, pack=None) -> None:
    """Download and verify a pack, and leave it ready for the next start."""
    root = Path(root)
    pack = pack or PACKS[key]
    fresh = root / PENDING
    downloads = [root / f"{FOLDER}.{pack.key}.{index}.download" for index in range(len(pack.wheels))]
    try:
        for wheel, target in zip(pack.wheels, downloads):
            _download(target, status, opener, wheel, pack.title)
        status("Проверяю скачанные файлы…")
        for wheel, target in zip(pack.wheels, downloads):
            if target.stat().st_size != wheel.size or not _matches(target, wheel.sha256):
                target.unlink()
                raise ValueError("контрольная сумма не совпала, файл нужно скачать заново")
        status("Распаковываю…")
        shutil.rmtree(fresh, ignore_errors=True)
        for target in downloads:
            with zipfile.ZipFile(target) as archive:
                archive.extractall(fresh)
        (fresh / MARKER).write_text(json.dumps(dict(pack=pack.key)), encoding="utf-8")
        (root / REMOVAL_REQUEST).unlink(missing_ok=True)
        for target in downloads:
            target.unlink()
    except DownloadInterrupted:
        raise
    except (OSError, ValueError, zipfile.BadZipFile, urllib.error.URLError, http.client.HTTPException) as error:
        shutil.rmtree(fresh, ignore_errors=True)
        raise RuntimeError(f"Не удалось установить ускорение: {error}") from error
