"""Optional GitHub updates. Only standard-library networking and file operations."""
import hashlib
import json
import os
from pathlib import Path
import re
import urllib.error
import urllib.parse
import urllib.request
import uuid

REPOSITORY = "dservEXE/StereoMap"
INSTALLER = "StereoMap-Setup.exe"
MAX_PACKAGE = 600 * 1024 * 1024


def version(value):
    if not isinstance(value, str) or not re.fullmatch(r"v?\d+\.\d+\.\d+", value):
        raise ValueError("Некорректная версия релиза")
    return tuple(int(part) for part in value.lstrip("v").split("."))


def select_release(data, current):
    if data.get("draft") or data.get("prerelease"):
        return None
    tag = data.get("tag_name")
    if version(tag) <= version(current):
        return None
    expected = f"https://github.com/{REPOSITORY}/releases/download/{tag}/{INSTALLER}"
    asset = next((item for item in data.get("assets", []) if item.get("name") == INSTALLER and item.get("state") == "uploaded"), None)
    if not asset or asset.get("browser_download_url") != expected or not isinstance(asset.get("size"), int) or not 0 < asset["size"] <= MAX_PACKAGE or not re.fullmatch(r"sha256:[a-fA-F0-9]{64}", asset.get("digest", "")):
        raise ValueError("Пакет обновления не прошёл проверку метаданных")
    return dict(tag=tag, version=tag.lstrip("v"), url=expected, size=asset["size"], sha256=asset["digest"][7:].lower())


def allowed_url(url):
    parsed = urllib.parse.urlsplit(url)
    return parsed.scheme == "https" and not parsed.username and not parsed.password and parsed.hostname in {"api.github.com", "github.com", "release-assets.githubusercontent.com", "objects.githubusercontent.com"}


class Redirects(urllib.request.HTTPRedirectHandler):
    max_redirections = 5

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not allowed_url(newurl):
            raise ValueError("Недопустимый адрес загрузки")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def request(url):
    if not allowed_url(url):
        raise ValueError("Недопустимый адрес загрузки")
    req = urllib.request.Request(url, headers={"User-Agent": "StereoMap-Updater", "Accept": "application/vnd.github+json"})
    return urllib.request.build_opener(Redirects()).open(req, timeout=30)


def check_release(current, opener=request):
    try:
        with opener(f"https://api.github.com/repos/{REPOSITORY}/releases/latest") as response:
            body = response.read(1024 * 1024 + 1)
        if len(body) > 1024 * 1024:
            raise ValueError("Ответ GitHub слишком большой")
        return select_release(json.loads(body), current)
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None
        raise


def update_root():
    return Path(os.environ["LOCALAPPDATA"]) / "StereoMap" / "updates"


def owned_directory(directory, root=None):
    directory, root = Path(directory), Path(root or update_root())
    if not re.fullmatch(r"[a-f0-9]{32}", directory.name) or directory.parent.resolve() != root.resolve() or not directory.is_dir():
        return False
    return all(not item.is_symlink() and not (hasattr(item, "is_junction") and item.is_junction()) for item in (directory, *directory.parents))


def cleanup(directory, root=None):
    directory = Path(directory)
    if not owned_directory(directory, root):
        return
    try:
        for name in (INSTALLER, INSTALLER + ".part"):
            file = directory / name
            if file.is_file() and not file.is_symlink():
                file.unlink()
        if not any(directory.iterdir()):
            directory.rmdir()
    except OSError:
        pass


def download_release(release, progress=lambda percent: None, root=None, opener=request):
    root = Path(root or update_root())
    directory = root / uuid.uuid4().hex
    directory.mkdir(parents=True)
    partial, target = directory / (INSTALLER + ".part"), directory / INSTALLER
    try:
        digest, total, last = hashlib.sha256(), 0, -1
        with opener(release["url"]) as response, partial.open("xb") as output:
            while True:
                block = response.read(64 * 1024)
                if not block:
                    break
                total += len(block)
                if total > release["size"]:
                    raise ValueError("Неверный размер пакета")
                digest.update(block)
                output.write(block)
                percent = total * 100 // release["size"]
                if percent != last:
                    last = percent
                    progress(percent)
        if total != release["size"] or digest.hexdigest() != release["sha256"]:
            raise ValueError("Контрольная сумма пакета не совпала")
        partial.replace(target)
        return target
    except Exception:
        cleanup(directory, root)
        raise
