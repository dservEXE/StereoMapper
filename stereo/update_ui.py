"""Tk controller; background workers communicate through a queue only."""
import json
import os
from pathlib import Path
from queue import Queue, Empty
import subprocess
import sys
from threading import Thread
import time
import tkinter as tk
from tkinter import ttk
from .version import VERSION
from .updates import check_release, download_release, cleanup, update_root, owned_directory


class UpdateController:
    def __init__(self, app, parent):
        self.app, self.root = app, app.root
        self.file = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "StereoMap" / "updates-settings.json"
        try:
            self.settings = json.loads(self.file.read_text(encoding="utf8"))
        except (OSError, ValueError):
            self.settings = {}
        if not isinstance(self.settings, dict):
            self.settings = {}
        if not isinstance(self.settings.get("last", 0), (int, float)):
            self.settings["last"] = 0
        self.events, self.available, self.prepared = Queue(), None, None
        self.working, self.poll_id = False, None
        self.auto = tk.BooleanVar(value=self.settings.get("auto", True))
        self.text = tk.StringVar(value=f"Версия {VERSION} · обновления необязательные")
        self.frame = ttk.Frame(parent, style="App.TFrame")
        self.frame.pack(fill="x", pady=(0, 10), before=app.workflow)
        self.label = ttk.Label(self.frame, textvariable=self.text, style="App.TLabel")
        self.label.pack(side="left", fill="x", expand=True)
        self.check_button = ttk.Button(self.frame, text="Проверить обновления", command=lambda: self.check(True))
        self.check_button.pack(side="right")
        self.auto_button = ttk.Checkbutton(self.frame, text="Автопроверка", variable=self.auto, command=self.configure)
        self.auto_button.pack(side="right", padx=12)
        self.banner = tk.Frame(parent, background="#c8ff76", padx=16, pady=10)
        self.banner_text = tk.Label(self.banner, text="", background="#c8ff76", foreground="#192611", font=("Segoe UI", 11, "bold"))
        self.banner_text.pack(side="left", fill="x", expand=True)
        self.later_button = ttk.Button(self.banner, text="Позже", command=self.later)
        self.later_button.pack(side="right", padx=8)
        self.install_button = ttk.Button(self.banner, text="Обновить", command=self.install)
        self.install_button.pack(side="right")
        self.root.after(1500, lambda: self.check(False))
        self.schedule_id = self.root.after(6 * 3600000, self.scheduled)
        self.startup_cleanup()

    def save(self):
        self.file.parent.mkdir(parents=True, exist_ok=True)
        self.file.write_text(json.dumps(self.settings), encoding="utf8")

    def configure(self):
        self.settings["auto"] = self.auto.get()
        self.save()

    def later(self):
        if self.available:
            self.settings["dismissed"] = self.available["tag"]
            self.save()
            self.banner.pack_forget()
            self.text.set("Обновление можно открыть кнопкой «Проверить обновления».")

    def scheduled(self):
        self.check(False)
        self.schedule_id = self.root.after(6 * 3600000, self.scheduled)

    def job(self, function):
        self.working = True
        self.check_button.configure(state="disabled")
        self.install_button.configure(state="disabled")
        def work():
            try:
                function()
            except Exception as error:
                self.events.put(("error", str(error)))
            finally:
                self.events.put(("done", None))
        Thread(target=work, daemon=True).start()
        self.poll_id = self.root.after(100, self.poll)

    def check(self, manual=False):
        if self.working or self.app.closing:
            return
        if not manual and (not self.auto.get() or time.time() - self.settings.get("last", 0) < 6 * 3600):
            return
        self.text.set("Проверяем GitHub Releases…")
        self.settings["last"] = time.time()
        self.save()
        self.job(lambda: self.events.put(("release", (check_release(VERSION), manual))))

    def busy(self):
        return self.app.connected or self.app.connection_pending or self.app.calibration_busy

    def install(self):
        if self.working or not self.available:
            return
        if self.busy():
            self.text.set("Сначала остановите камеры и дождитесь завершения обработки.")
            return
        root = self.app.directory.resolve()
        if not (root / "runtime/pythonw.exe").is_file() or not (root / "StereoMapper.exe").is_file():
            self.text.set("Обновление устанавливается в готовую программу. В исходниках используйте Git.")
            return
        release = self.available.copy()
        self.job(lambda: self.events.put(("downloaded", (download_release(release, lambda percent: self.events.put(("progress", percent))), root))))

    def poll(self):
        if self.app.closing:
            return
        try:
            while True:
                kind, value = self.events.get_nowait()
                if kind == "release":
                    self.available, manual = value
                    if self.available:
                        if manual:
                            self.settings["dismissed"] = None
                            self.save()
                        self.text.set("Доступна " + self.available["version"] + " · можно позже")
                        self.banner_text.configure(text="Новая версия " + self.available["version"])
                        if self.settings.get("dismissed") != self.available["tag"]:
                            self.banner.pack(fill="x", pady=(0, 10), before=self.frame)
                    else:
                        self.banner.pack_forget()
                        self.text.set("Установлена актуальная версия " + VERSION)
                elif kind == "progress":
                    self.text.set(f"Загрузка {value}% · проверим SHA-256")
                elif kind == "downloaded":
                    installer, root = value
                    if self.busy():
                        cleanup(installer.parent)
                        self.text.set("Камеры были включены во время загрузки. Остановите их и повторите обновление.")
                    else:
                        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "DETACHED_PROCESS", 0)
                        try:
                            subprocess.Popen([str(installer), "--update", str(os.getpid()), sys.executable, str(root), str(installer.parent)], creationflags=flags, close_fds=True)
                            self.app.close()
                        except OSError as error:
                            cleanup(installer.parent)
                            self.text.set("Не удалось запустить обновление: " + str(error))
                elif kind == "error":
                    self.text.set("Обновление недоступно: " + value)
                elif kind == "done":
                    self.working = False
                    self.check_button.configure(state="normal")
                    self.install_button.configure(state="normal")
        except Empty:
            pass
        if self.working:
            self.poll_id = self.root.after(100, self.poll)

    def startup_cleanup(self):
        # Only the known updater directory and known installer filenames are removed.
        try:
            root = update_root()
            for directory in root.iterdir():
                if owned_directory(directory) and time.time() - directory.stat().st_mtime > 86400:
                    cleanup(directory)
        except OSError:
            pass
        if "--cleanup-update" not in sys.argv:
            return
        try:
            i = sys.argv.index("--cleanup-update")
            directory, pid = Path(sys.argv[i + 1]), int(sys.argv[i + 2])
            if not owned_directory(directory):
                return
            def wait():
                import ctypes
                from ctypes import wintypes
                kernel = ctypes.WinDLL("kernel32", use_last_error=True)
                kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
                kernel.OpenProcess.restype = wintypes.HANDLE
                kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
                kernel.WaitForSingleObject.restype = wintypes.DWORD
                kernel.CloseHandle.argtypes = [wintypes.HANDLE]
                handle = kernel.OpenProcess(0x100000, False, pid)
                if handle:
                    result = kernel.WaitForSingleObject(handle, 60000)
                    kernel.CloseHandle(handle)
                    if result != 0:
                        return
                cleanup(directory)
            Thread(target=wait, daemon=True).start()
        except (ValueError, IndexError, OSError):
            pass
