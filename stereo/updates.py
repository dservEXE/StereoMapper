"""Verified, optional GitHub release updates; independent of camera libraries."""
import hashlib
import json
import os
from pathlib import Path
import re
import urllib.error
import urllib.parse
import urllib.request
import uuid

REPOSITORY = "dservEXE/StereoMapper"
INSTALLER = "StereoMapper-Update.exe"
SETUP = "StereoMapper-Setup.exe"
RUNTIME_ABI = 1
MAX_PACKAGE = 128 * 1024 * 1024
MAX_METADATA = 1024 * 1024


class Cancelled(Exception):
    """The window closed or the user cancelled a pending download."""


def version(value):
    if not isinstance(value, str) or not re.fullmatch(r"v?\d+\.\d+\.\d+", value):
        raise ValueError("Некорректная версия релиза")
    return tuple(int(part) for part in value.lstrip("v").split("."))


def select_release(data, current):
    if not isinstance(data, dict):
        raise ValueError("Некорректный ответ GitHub")
    if data.get("draft") or data.get("prerelease"):
        return None
    tag = data.get("tag_name")
    if version(tag) <= version(current):
        return None
    page = f"https://github.com/{REPOSITORY}/releases/tag/{tag}"
    body = data.get("body") or ""
    if not isinstance(body, str):
        raise ValueError("Некорректное описание релиза")
    marker = re.search(r"^StereoMapper-Runtime:\s*(\d+)\s*$", body, re.MULTILINE)
    # Unknown runtime compatibility requires a full installer, never code alone.
    if not marker or int(marker.group(1)) != RUNTIME_ABI:
        return dict(tag=tag, version=tag.lstrip("v"), requires_setup=True, page=page)
    expected = f"https://github.com/{REPOSITORY}/releases/download/{tag}/{INSTALLER}"
    assets = data.get("assets", [])
    if not isinstance(assets, list):
        raise ValueError("Некорректный список файлов релиза")
    matches = [item for item in assets if isinstance(item, dict) and item.get("name") == INSTALLER and item.get("state") == "uploaded"]
    if len(matches) != 1:
        raise ValueError("В релизе пока нет готового пакета обновления")
    asset = matches[0]
    digest = asset.get("digest")
    if (asset.get("browser_download_url") != expected or type(asset.get("size")) is not int
            or not 0 < asset["size"] <= MAX_PACKAGE or not isinstance(digest, str)
            or not re.fullmatch(r"sha256:[a-fA-F0-9]{64}", digest)):
        raise ValueError("Пакет обновления не прошёл проверку метаданных")
    return dict(tag=tag, version=tag.lstrip("v"), url=expected, size=asset["size"],
                sha256=digest[7:].lower(), requires_setup=False, page=page)


def allowed_url(url):
    try:
        parsed = urllib.parse.urlsplit(url)
        return (parsed.scheme == "https" and not parsed.username and not parsed.password
                and parsed.port in (None, 443) and not parsed.fragment
                and parsed.hostname in {"api.github.com", "github.com", "release-assets.githubusercontent.com", "objects.githubusercontent.com"})
    except (ValueError, TypeError):
        return False


class Redirects(urllib.request.HTTPRedirectHandler):
    max_redirections = 5

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not allowed_url(newurl):
            raise ValueError("Недопустимый адрес загрузки")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def request(url):
    if not allowed_url(url):
        raise ValueError("Недопустимый адрес загрузки")
    req = urllib.request.Request(url, headers={"User-Agent": "StereoMapper-Updater",
        "Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"})
    return urllib.request.build_opener(Redirects()).open(req, timeout=15)


def check_release(current, opener=request):
    try:
        with opener(f"https://api.github.com/repos/{REPOSITORY}/releases/latest") as response:
            body = response.read(MAX_METADATA + 1)
        if len(body) > MAX_METADATA:
            raise ValueError("Ответ GitHub слишком большой")
        return select_release(json.loads(body), current)
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None
        raise


def state_root():
    return Path(os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData/Local")) / "StereoMapper"


def update_root():
    return state_root() / "updates"


def ordinary_path(path):
    return all(not item.is_symlink() and not (hasattr(item, "is_junction") and item.is_junction())
               for item in (Path(path), *Path(path).parents))


def owned_directory(directory, root=None):
    directory, root = Path(directory), Path(root or update_root())
    return bool(re.fullmatch(r"[a-f0-9]{32}", directory.name)
                and directory.parent.resolve() == root.resolve() and directory.is_dir()
                and ordinary_path(directory))


def cleanup(directory, root=None):
    directory = Path(directory)
    if not owned_directory(directory, root):
        return
    try:
        for name in (INSTALLER, INSTALLER + ".part"):
            file = directory / name
            if file.is_file() and ordinary_path(file):
                file.unlink()
        if not any(directory.iterdir()):
            directory.rmdir()
    except OSError:
        pass


def download_release(release, progress=lambda percent: None, root=None, opener=request, cancelled=lambda: False):
    # Revalidate before requests or file creation, even when called without select_release.
    tag = release.get("tag")
    version(tag)
    expected = f"https://github.com/{REPOSITORY}/releases/download/{tag}/{INSTALLER}"
    if (release.get("url") != expected or type(release.get("size")) is not int
            or not 0 < release["size"] <= MAX_PACKAGE or not isinstance(release.get("sha256"), str)
            or not re.fullmatch(r"[a-f0-9]{64}", release["sha256"])):
        raise ValueError("Некорректный пакет обновления")
    if cancelled():
        raise Cancelled()
    root = Path(root or update_root())
    if not ordinary_path(root):
        raise ValueError("Папка обновлений содержит перенаправление")
    directory = root / uuid.uuid4().hex
    directory.mkdir(parents=True)
    partial, target = directory / (INSTALLER + ".part"), directory / INSTALLER
    try:
        digest, total, last = hashlib.sha256(), 0, -1
        with opener(expected) as response, partial.open("xb") as output:
            while True:
                if cancelled():
                    raise Cancelled()
                block = response.read(64 * 1024)
                if not block:
                    break
                total += len(block)
                if total > release["size"]:
                    raise ValueError("Неверный размер пакета")
                digest.update(block)
                output.write(block)
                percent = min(99, total * 100 // release["size"])
                if percent != last:
                    last = percent
                    progress(percent)
        if total != release["size"] or digest.hexdigest() != release["sha256"]:
            raise ValueError("Контрольная сумма пакета не совпала")
        if cancelled():
            raise Cancelled()
        partial.replace(target)
        progress(100)
        return target
    except Exception:
        cleanup(directory, root)
        raise
